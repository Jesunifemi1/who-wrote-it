"""All CNN / LSTM experiments (RQ1-RQ4). Each job writes results/nn/<job>.json and is skipped if
that file already exists, so the queue can be stopped and resumed.

  python run_nn.py --queue cnn      # all CNN jobs
  python run_nn.py --queue lstm     # all LSTM jobs
  python run_nn.py --only rq1_cnn_output_s0
"""
import argparse
import json
import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from ablations import TRANSFORMS
from common import (FAMILIES, LABEL2ID, RESULTS, evaluate, labels_of, load_data, load_siblings,
                    make_sequences, split_frames, train_model)

OUTDIR = RESULTS / "nn"
MODELDIR = RESULTS / "models"
DF, SPLITS = load_data()
TR, VA, TE = split_frames(DF, SPLITS)


def run_basic(arch, train, val, test, mode="output", seed=0, transform=None, max_out=None, extra_eval=None):
    kw = {} if max_out is None else {"max_out": max_out}
    s_tr = make_sequences(train, mode, transform=transform, **kw)
    s_va = make_sequences(val, mode, transform=transform, **kw)
    s_te = make_sequences(test, mode, transform=transform, **kw)
    t0 = time.time()
    model, vocab, hist = train_model(arch, s_tr, labels_of(train), s_va, labels_of(val), seed=seed)
    m, pred, prob = evaluate(model, vocab, s_te, labels_of(test))
    res = {
        "arch": arch, "mode": mode, "seed": seed, "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "vocab": len(vocab), "params": sum(p.numel() for p in model.parameters()), "epochs": len(hist),
        "best_epoch": int(np.argmax([h["val_macro_f1"] for h in hist]) + 1), "train_sec": time.time() - t0,
        "history": hist, "test": m,
        "test_pred": {"prompt_id": test.prompt_id.tolist(), "y": labels_of(test).tolist(), "pred": pred.tolist()},
    }
    if extra_eval:
        res.update(extra_eval(model, vocab, kw))
    return res, model, vocab


# ---------------------------------------------------------------- RQ1 / RQ2 (main split)


def eval_siblings(model, vocab, kw):
    """RQ3 cross-version: models trained on one version per family, tested on a newer/older sibling."""
    sib = load_siblings()
    sib = sib[sib.prompt_id.isin(SPLITS["test"])].reset_index(drop=True)
    seq = make_sequences(sib, "output", **kw)
    m, pred, _ = evaluate(model, vocab, seq, labels_of(sib))
    per_fam = {f: float((pred[sib.family.values == f] == LABEL2ID[f]).mean()) for f in sib.family.unique()}
    te_pred = evaluate(model, vocab, make_sequences(TE, "output", **kw), labels_of(TE))[1]
    per_fam_primary = {f: float((te_pred[TE.family.values == f] == LABEL2ID[f]).mean()) for f in FAMILIES}
    return {"siblings": {"metrics": m, "recall_by_family": per_fam, "primary_recall_by_family": per_fam_primary,
                         "models": dict(zip(sib.family, sib.model)),
                         "pred_counts": {f: np.bincount(pred[sib.family.values == f], minlength=6).tolist()
                                         for f in sib.family.unique()}}}


def job_main(arch, mode, seed, max_out=None, tag=None):
    name = tag or f"rq1_{arch}_{mode}_s{seed}"
    extra = eval_siblings if mode == "output" else None
    res, model, vocab = run_basic(arch, TR, VA, TE, mode, seed, max_out=max_out, extra_eval=extra)
    if mode == "output" and seed == 0 and max_out is None:
        MODELDIR.mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "itos": vocab.itos}, MODELDIR / f"{arch}_output_s0.pt")
    return name, res


# ---------------------------------------------------------------- RQ2: confounded (non-parallel) collection


def confounded_frames(kind, seed=0, p_pref=0.7):
    """Keep ONE response per prompt. 'random': the family is chosen uniformly (like a fair
    collection). 'topic_biased': prompts are clustered by topic (k-means on TF-IDF of the
    instruction) and each topic cluster 'prefers' one family with prob p_pref, mimicking logs where
    different user communities use different chatbots."""
    rng = np.random.default_rng(seed)
    pids = np.array(sorted(DF.prompt_id.unique()))
    instr = DF[DF.family == "GPT"].set_index("prompt_id").loc[pids, "input"].tolist()
    if kind == "random":
        fam = rng.integers(0, 6, len(pids))
    else:
        X = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True).fit_transform(instr)
        km = KMeans(6, n_init=10, random_state=seed).fit(X)
        # capacity-constrained assignment -> 6 equal-size topic groups (keeps the labels balanced)
        dist = km.transform(X)
        cap = np.full(6, int(np.ceil(len(pids) / 6)))
        cl = np.empty(len(pids), dtype=int)
        for i in np.argsort(dist.min(1) - np.sort(dist, 1)[:, 1]):  # most confident first
            for c in np.argsort(dist[i]):
                if cap[c] > 0:
                    cl[i] = c
                    cap[c] -= 1
                    break
        fam = np.where(rng.random(len(pids)) < p_pref, cl, rng.integers(0, 6, len(pids)))
    chosen = {int(p): FAMILIES[int(f)] for p, f in zip(pids, fam)}
    d = DF[DF.apply(lambda r: chosen[r.prompt_id] == r.family, axis=1)].reset_index(drop=True)
    return split_frames(d, SPLITS)


def job_confound(arch, kind, mode, seed=0):
    tr, va, te = confounded_frames(kind)
    res, _, _ = run_basic(arch, tr, va, te, mode, seed)
    res["label_dist_train"] = tr.family.value_counts().to_dict()
    return f"rq2conf_{arch}_{kind}_{mode}_s{seed}", res


# ---------------------------------------------------------------- RQ3: cross-task and cross-source


def domain_of(task):
    return {"coding": "technical", "math": "technical"}.get(task, task)


DOM = DF[DF.family == "GPT"].set_index("prompt_id").task.map(domain_of)
SRC = DF[DF.family == "GPT"].set_index("prompt_id").source


def _val_split(pids, rng, frac=0.12):
    pids = np.array(sorted(pids))
    rng.shuffle(pids)
    k = int(frac * len(pids))
    return set(pids[k:]), set(pids[:k])


def task_shift_frames(exp, cond, fold):
    """exp 'write': train on technical+qa, test on writing.  exp 'tech': train on qa, test on technical.
    Target prompts are split in two halves; the test half is fixed per fold. 'ood' trains only on
    the source domains; 'id' swaps an equal number of source prompts for the other target half, so
    both conditions have the same training size and the same test set."""
    tgt = "writing" if exp == "write" else "technical"
    src_doms = ["technical", "qa"] if exp == "write" else ["qa"]
    rng = np.random.default_rng(100 + (exp == "tech"))
    tgt_p = np.array(sorted(DOM[DOM == tgt].index))
    rng.shuffle(tgt_p)
    halves = [set(tgt_p[: len(tgt_p) // 2]), set(tgt_p[len(tgt_p) // 2 :])]
    test_p, other_half = halves[fold], halves[1 - fold]
    src_p = set(DOM[DOM.isin(src_doms)].index)
    rng2 = np.random.default_rng(7 + fold)
    if cond == "id":
        drop = set(rng2.choice(sorted(src_p), size=len(other_half), replace=False))
        pool = (src_p - drop) | other_half
    else:
        pool = src_p
    trp, vap = _val_split(pool, np.random.default_rng(11 + fold))
    f = lambda s: DF[DF.prompt_id.isin(s)].reset_index(drop=True)
    return f(trp), f(vap), f(test_p)


def job_taskshift(arch, exp, cond, fold, seed=0):
    tr, va, te = task_shift_frames(exp, cond, fold)
    res, _, _ = run_basic(arch, tr, va, te, "output", seed)
    return f"rq3task_{arch}_{exp}_{cond}_f{fold}", res


def job_loso(arch, source, seed=0):
    test_p = set(SRC[SRC == source].index)
    trp, vap = _val_split(set(SRC.index) - test_p, np.random.default_rng(5))
    f = lambda s: DF[DF.prompt_id.isin(s)].reset_index(drop=True)
    res, _, _ = run_basic(arch, f(trp), f(vap), f(test_p), "output", seed)
    return f"rq3loso_{arch}_{source}", res


# ---------------------------------------------------------------- RQ4: ablations


def job_ablation(arch, name, seed=0):
    if name.startswith("first"):
        k = int(name[5:])
        res, _, _ = run_basic(arch, TR, VA, TE, "output", seed, max_out=k)
    else:
        res, _, _ = run_basic(arch, TR, VA, TE, "output", seed, transform=TRANSFORMS[name])
    return f"rq4abl_{arch}_{name}_s{seed}", res


# ---------------------------------------------------------------- queues

def queue(arch):
    if arch == "lstm":  # ordered by priority: the BiLSTM is ~3x slower than the CNN on CPU
        q = [(job_main, (arch, "output", s)) for s in range(3)]
        q += [(job_main, (arch, "both", 0))]
        q += [(job_main, (arch, "input", s)) for s in range(3)]
        q += [(job_ablation, (arch, a)) for a in ("no_format", "skeleton")]
        q += [(job_taskshift, (arch, e, c, 0)) for e in ("write", "tech") for c in ("ood", "id")]
        q += [(job_main, (arch, "both", s)) for s in (1, 2)]
        q += [(job_ablation, (arch, a)) for a in ("content_only", "first64")]
        q += [(job_taskshift, (arch, e, c, 1)) for e in ("write", "tech") for c in ("ood", "id")]
        return q
    q = []
    q += [(job_main, (arch, "output", s)) for s in range(3)]
    q += [(job_main, (arch, "both", s)) for s in range(3)]
    q += [(job_main, (arch, "input", s)) for s in range(3)]
    q += [(job_ablation, (arch, a)) for a in ["no_format", "skeleton", "content_only", "words_only", "no_open_close", "first64"]]
    q += [(job_taskshift, (arch, e, c, f)) for f in (0, 1) for e in ("write", "tech") for c in ("ood", "id")]
    q += [(job_confound, (arch, k, m)) for k in ("topic_biased", "random") for m in ("input", "output", "both")]
    q += [(job_loso, (arch, s)) for s in ("helpful_base", "koala", "oasst", "selfinstruct", "vicuna")]
    q += [(job_main, (arch, "output", 0, 512, "rq1_cnn_output512_s0"))]
    q += [(job_ablation, (arch, a, 1)) for a in ["no_format", "skeleton", "content_only", "first64"]]
    return q


def job_name(fn, args):
    a = list(args)
    if fn is job_main:
        return a[4] if len(a) > 4 else f"rq1_{a[0]}_{a[1]}_s{a[2]}"
    if fn is job_confound:
        return f"rq2conf_{a[0]}_{a[1]}_{a[2]}_s0"
    if fn is job_taskshift:
        return f"rq3task_{a[0]}_{a[1]}_{a[2]}_f{a[3]}"
    if fn is job_loso:
        return f"rq3loso_{a[0]}_{a[1]}"
    if fn is job_ablation:
        return f"rq4abl_{a[0]}_{a[1]}_s{a[2] if len(a) > 2 else 0}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", choices=["cnn", "lstm"])
    ap.add_argument("--only", default=None)
    ap.add_argument("--reverse", action="store_true", help="walk the queue from the end (for a second worker)")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    jobs = queue(args.queue)
    for fn, a in (jobs[::-1] if args.reverse else jobs):
        name = job_name(fn, a)
        if args.only and name != args.only:
            continue
        if (OUTDIR / f"{name}.json").exists():
            continue
        lock = OUTDIR / f"{name}.lock"  # lets several workers share one queue
        try:
            lock.open("x").close()
        except FileExistsError:
            continue
        print(f"=== {name}", flush=True)
        n2, res = fn(*a)
        assert n2 == name, (n2, name)
        json.dump(res, open(OUTDIR / f"{name}.json", "w"))
        lock.unlink()
        print(f"=== {name}: test acc={res['test']['accuracy']:.3f} macroF1={res['test']['macro_f1']:.3f} "
              f"({res['train_sec']:.0f}s, {res['epochs']} ep)", flush=True)
