"""Shared code: tokenizer, vocabulary, TextCNN, BiLSTM, training / evaluation loop."""
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
FAMILIES = ["GPT", "Claude", "Gemini", "Llama", "Qwen", "DeepSeek"]
LABEL2ID = {f: i for i, f in enumerate(FAMILIES)}

import os

torch.set_num_threads(int(os.environ.get("THREADS", "2")))
MAX_OUT = 256  # output tokens fed to the CNN / LSTM (~1,100 characters); see report for the 512-token check

# ------------------------------------------------------------------ data


def load_data():
    df = pd.read_csv(DATA / "dataset.csv", keep_default_na=False)
    splits = json.load(open(DATA / "splits.json"))
    return df, splits


def load_siblings():
    return pd.read_csv(DATA / "siblings.csv", keep_default_na=False)


def split_frames(df, splits):
    return tuple(df[df.prompt_id.isin(splits[k])].reset_index(drop=True) for k in ("train", "val", "test"))


# ------------------------------------------------------------------ tokenizer
# Case is kept and every punctuation mark / newline is its own token, because formatting
# (markdown bullets, '**', headers, line breaks) is a plausible part of an LLM's "fingerprint".
TOKEN_RE = re.compile(r"\n|[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\sA-Za-z\d]")


def tokenize(text: str):
    return ["<nl>" if t == "\n" else t for t in TOKEN_RE.findall(text)]


PAD, UNK, SEP = "<pad>", "<unk>", "<sep>"


class Vocab:
    def __init__(self, token_lists, min_freq=2, max_size=30000):
        c = Counter(t for toks in token_lists for t in toks)
        words = [w for w, n in c.most_common(max_size) if n >= min_freq]
        self.itos = [PAD, UNK, SEP] + [w for w in words if w not in (PAD, UNK, SEP)]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, toks):
        u = self.stoi[UNK]
        return [self.stoi.get(t, u) for t in toks]

    def __len__(self):
        return len(self.itos)


def make_sequences(frame, mode, max_in=96, max_out=MAX_OUT, input_only_len=256, transform=None):
    """mode: 'output' | 'input' | 'both'. transform(text)->text is applied to the output (RQ4 ablations)."""
    seqs = []
    for inp, out in zip(frame["input"], frame["output"]):
        if transform is not None:
            out = transform(out)
        if mode == "output":
            s = tokenize(out)[:max_out]
        elif mode == "input":
            s = tokenize(inp)[:input_only_len]
        elif mode == "both":
            s = tokenize(inp)[:max_in] + [SEP] + tokenize(out)[:max_out]
        else:
            raise ValueError(mode)
        seqs.append(s if s else [UNK])
    return seqs


def to_tensor_batches(ids, labels, batch_size, shuffle, rng=None):
    lens_all = np.array([len(x) for x in ids])
    if shuffle:  # shuffle, then sort inside chunks of 20 batches so batches have similar lengths
        order = rng.permutation(len(ids))
        chunk = batch_size * 20
        order = np.concatenate([c[np.argsort(lens_all[c])] for c in np.split(order, range(chunk, len(order), chunk))])
        batches = [order[s : s + batch_size] for s in range(0, len(order), batch_size)]
        rng.shuffle(batches)
    else:  # bucket by length for faster eval
        order = np.argsort(lens_all)
        batches = [order[s : s + batch_size] for s in range(0, len(order), batch_size)]
    for idx in batches:
        lens = [len(ids[i]) for i in idx]
        L = max(lens)
        x = torch.zeros(len(idx), L, dtype=torch.long)
        for r, i in enumerate(idx):
            x[r, : lens[r]] = torch.tensor(ids[i])
        yield idx, x, torch.tensor(lens), torch.tensor([labels[i] for i in idx])


# ------------------------------------------------------------------ models


class TextCNN(nn.Module):
    """Kim (2014)-style CNN: embedding -> parallel 1-D convs (widths 2..5) -> max-over-time -> FC."""

    def __init__(self, vocab_size, n_classes, emb=128, n_filters=100, widths=(2, 3, 4, 5), dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb, n_filters, w, padding=w - 1) for w in widths])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(widths), n_classes)

    def features(self, x, lens):
        mask = (x != 0).unsqueeze(1)  # B,1,L
        e = self.emb(x).transpose(1, 2)  # B,E,L
        pooled, acts = [], []
        for conv in self.convs:
            h = torch.relu(conv(e))  # B,F,L+w-1
            m = torch.ones(h.size(0), 1, h.size(2), dtype=torch.bool)
            m[:, :, : mask.size(2)] = mask  # windows that start on padding are ignored
            h = h.masked_fill(~m, 0.0)
            acts.append(h)
            pooled.append(h.max(dim=2).values)
        return torch.cat(pooled, dim=1), acts

    def forward(self, x, lens):
        f, _ = self.features(x, lens)
        return self.fc(self.drop(f))


class BiLSTM(nn.Module):
    """Embedding -> 1-layer bidirectional LSTM -> [max-pool ; mean-pool] over valid steps -> FC."""

    def __init__(self, vocab_size, n_classes, emb=128, hidden=96, dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.emb_drop = nn.Dropout(0.2)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * hidden, n_classes)

    def forward(self, x, lens):
        e = self.emb_drop(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(e, lens, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        mask = (x != 0).unsqueeze(2)
        mx = out.masked_fill(~mask, -1e9).max(dim=1).values
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(self.drop(torch.cat([mx, mean], dim=1)))


ARCH = {"cnn": TextCNN, "lstm": BiLSTM}
HPARAMS = {
    "cnn": dict(lr=1e-3, weight_decay=1e-4, batch_size=32, max_epochs=15, patience=3),
    "lstm": dict(lr=2e-3, weight_decay=1e-4, batch_size=32, max_epochs=12, patience=3),
}

# ------------------------------------------------------------------ training


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, ids, labels, batch_size=64):
    model.eval()
    preds = np.zeros(len(ids), dtype=int)
    probs = np.zeros((len(ids), model.fc.out_features), dtype=np.float32)
    for idx, x, lens, _ in to_tensor_batches(ids, labels, batch_size, shuffle=False):
        logits = model(x, lens)
        p = torch.softmax(logits, dim=1).numpy()
        probs[idx] = p
        preds[idx] = p.argmax(1)
    return preds, probs


def metrics(y, p, n_classes=len(FAMILIES)):
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", labels=list(range(n_classes)), zero_division=0)),
        "per_class_f1": [float(v) for v in f1_score(y, p, average=None, labels=list(range(n_classes)), zero_division=0)],
        "confusion": confusion_matrix(y, p, labels=list(range(n_classes))).tolist(),
    }


def train_model(arch, train_seqs, y_train, val_seqs, y_val, seed=0, n_classes=len(FAMILIES), verbose=True, **over):
    """Builds the vocabulary on the training sequences only, trains with early stopping on
    validation macro-F1 and restores the best epoch. Returns (model, vocab, history)."""
    hp = {**HPARAMS[arch], **over}
    set_seed(seed)
    vocab = Vocab(train_seqs)
    tr_ids = [vocab.encode(s) for s in train_seqs]
    va_ids = [vocab.encode(s) for s in val_seqs]
    model = ARCH[arch](len(vocab), n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    lossf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(1, hp["max_epochs"] + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for _, x, lens, yb in to_tensor_batches(tr_ids, y_train, hp["batch_size"], True, rng):
            opt.zero_grad()
            loss = lossf(model(x, lens), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        vp, _ = predict(model, va_ids, y_val)
        vf1 = f1_score(y_val, vp, average="macro")
        vacc = accuracy_score(y_val, vp)
        hist.append({"epoch": ep, "train_loss": tot / n, "val_acc": vacc, "val_macro_f1": vf1, "sec": time.time() - t0})
        if verbose:
            print(f"  [{arch} s{seed}] ep{ep:2d} loss={tot/n:.3f} val_acc={vacc:.3f} val_f1={vf1:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if vf1 > best + 1e-4:
            best, bad = vf1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    return model, vocab, hist


def evaluate(model, vocab, seqs, y):
    p, prob = predict(model, [vocab.encode(s) for s in seqs], y)
    return metrics(y, p, model.fc.out_features), p, prob


def labels_of(frame):
    return frame["family"].map(LABEL2ID).to_numpy()


def save_json(obj, name):
    RESULTS.mkdir(exist_ok=True)
    json.dump(obj, open(RESULTS / name, "w"), indent=1)


def summarize(runs, key):
    v = np.array([r[key] for r in runs])
    return float(v.mean()), float(v.std())
