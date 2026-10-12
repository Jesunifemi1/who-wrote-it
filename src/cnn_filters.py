"""RQ4: which n-grams does the trained CNN respond to? For every class we take the 8 filters with the
largest positive weight into that class's logit and list the n-grams (test set) that maximally
activate them, counted over test responses of that family."""
from collections import Counter, defaultdict

import torch

from common import FAMILIES, RESULTS, TextCNN, Vocab, labels_of, load_data, make_sequences, save_json, split_frames

WIDTHS = (2, 3, 4, 5)


def main():
    df, sp = load_data()
    _, _, te = split_frames(df, sp)
    ck = torch.load(RESULTS / "models" / "cnn_output_s0.pt")
    vocab = Vocab([])
    vocab.itos = ck["itos"]
    vocab.stoi = {w: i for i, w in enumerate(vocab.itos)}
    model = TextCNN(len(vocab), len(FAMILIES))
    model.load_state_dict(ck["state"])
    model.eval()
    seqs = make_sequences(te, "output")
    y = labels_of(te)
    W = model.fc.weight.detach()  # C x (4*100)
    nf = model.convs[0].out_channels
    out = {}
    for c, fam in enumerate(FAMILIES):
        top_filters = torch.argsort(W[c], descending=True)[:8].tolist()
        counts = defaultdict(Counter)
        for s, lab in zip(seqs, y):
            if lab != c:
                continue
            ids = torch.tensor([vocab.encode(s)])
            with torch.no_grad():
                _, acts = model.features(ids, torch.tensor([len(s)]))
            for j in top_filters:
                wi, fi = divmod(j, nf)
                a = acts[wi][0, fi]
                pos = int(a.argmax())
                w = WIDTHS[wi]
                start = pos - (w - 1)  # conv padding = w-1
                gram = [s[k] if 0 <= k < len(s) else "_" for k in range(start, start + w)]
                counts[j][" ".join(gram).replace("<nl>", "\\n")] += 1
        out[fam] = []
        for j in top_filters:
            wi, fi = divmod(j, nf)
            out[fam].append({"filter": j, "width": WIDTHS[wi], "weight": round(float(W[c, j]), 3),
                             "top_ngrams": counts[j].most_common(4)})
        print(fam, [d["top_ngrams"][0] for d in out[fam][:5]])
    save_json(out, "cnn_filters.json")


if __name__ == "__main__":
    main()
