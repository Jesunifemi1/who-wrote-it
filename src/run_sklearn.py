"""Non-neural parts: TF-IDF + logistic-regression baseline (RQ1/RQ2), stylometric features,
feature-group ablations, permutation importance, discriminative n-grams and openers (RQ4)."""
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import FAMILIES, LABEL2ID, labels_of, load_data, metrics, save_json, split_frames, tokenize

DF, SPLITS = load_data()
TR, VA, TE = split_frames(DF, SPLITS)
TRV = pd.concat([TR, VA], ignore_index=True)  # sklearn models have no early stopping -> use train+val


def text_of(frame, mode):
    if mode == "output":
        return frame.output.tolist()
    if mode == "input":
        return frame.input.tolist()
    return [i + " <sep> " + o for i, o in zip(frame.input, frame.output)]


def tfidf_lr(train, test, mode, C=4.0):
    vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2),
                          min_df=2, max_features=200000, sublinear_tf=True)
    Xtr = vec.fit_transform(text_of(train, mode))
    clf = LogisticRegression(C=C, max_iter=3000)
    clf.fit(Xtr, labels_of(train))
    pred = clf.predict(vec.transform(text_of(test, mode)))
    return metrics(labels_of(test), pred), vec, clf, pred


# ---------------------------------------------------------------- stylometric features
OPENERS = re.compile(r"^\W*(certainly|sure|of course|absolutely|great question|yes|no|here|i)\b", re.I)
FEATS = {}


def feat(group):
    def deco(fn):
        FEATS[fn.__name__] = (group, fn)
        return fn
    return deco


def _words(t):
    return re.findall(r"[A-Za-z']+", t)


@feat("length")
def n_chars(t): return len(t)
@feat("length")
def n_words(t): return len(_words(t))
@feat("length")
def n_lines(t): return t.count("\n") + 1
@feat("length")
def n_paragraphs(t): return len([p for p in re.split(r"\n\s*\n", t) if p.strip()])
@feat("formatting")
def n_headers(t): return len(re.findall(r"^\s{0,3}#{1,6}\s", t, re.M))
@feat("formatting")
def n_bullets(t): return len(re.findall(r"^\s*[-*+•]\s+", t, re.M))
@feat("formatting")
def n_numbered(t): return len(re.findall(r"^\s*\d{1,3}[.)]\s+", t, re.M))
@feat("formatting")
def n_bold(t): return t.count("**") // 2
@feat("formatting")
def n_code_blocks(t): return t.count("```") // 2
@feat("formatting")
def frac_list_lines(t):
    lines = [l for l in t.split("\n") if l.strip()]
    return np.mean([bool(re.match(r"\s*([-*+•]|\d{1,3}[.)])\s+", l)) for l in lines]) if lines else 0.0
@feat("formatting")
def n_colon_lines(t): return len(re.findall(r":\s*$", t, re.M))
@feat("lexical")
def avg_word_len(t):
    w = _words(t)
    return np.mean([len(x) for x in w]) if w else 0.0
@feat("lexical")
def avg_sent_len(t):
    s = [x for x in re.split(r"(?<=[.!?])\s+|\n+", t) if _words(x)]
    return np.mean([len(_words(x)) for x in s]) if s else 0.0
@feat("lexical")
def ttr_first200(t):  # lexical diversity, length-controlled
    w = [x.lower() for x in _words(t)][:200]
    return len(set(w)) / len(w) if w else 0.0
@feat("lexical")
def rate_I(t):
    w = _words(t)
    return sum(x in ("I", "I'm", "I'd", "I'll", "I've") for x in w) / max(1, len(w))
@feat("lexical")
def rate_you(t):
    w = [x.lower() for x in _words(t)]
    return sum(x in ("you", "your", "you're", "yours") for x in w) / max(1, len(w))
@feat("lexical")
def rate_hedge(t):
    w = [x.lower() for x in _words(t)]
    return sum(x in ("however", "may", "might", "generally", "typically", "often", "important") for x in w) / max(1, len(w))
@feat("punctuation")
def rate_exclaim(t): return t.count("!") / max(1, len(t)) * 1000
@feat("punctuation")
def rate_question(t): return t.count("?") / max(1, len(t)) * 1000
@feat("punctuation")
def rate_colon(t): return t.count(":") / max(1, len(t)) * 1000
@feat("punctuation")
def rate_semicolon(t): return t.count(";") / max(1, len(t)) * 1000
@feat("punctuation")
def rate_dash(t): return (t.count("—") + t.count("–") + t.count(" - ")) / max(1, len(t)) * 1000
@feat("punctuation")
def rate_paren(t): return t.count("(") / max(1, len(t)) * 1000
@feat("punctuation")
def n_emoji(t): return len(re.findall(r"[\U0001F300-\U0001FAFF☀-➿]", t))
@feat("discourse")
def opens_affirmation(t): return float(bool(re.match(r"^\W*(certainly|sure|of course|absolutely|great|yes)\b", t, re.I)))
@feat("discourse")
def opens_restating(t): return float(bool(re.match(r"^\W*(here|the|a|an|to|in|this)\b", t, re.I)))
@feat("discourse")
def has_conclusion(t): return float(bool(re.search(r"\b(in summary|in conclusion|overall|to summarize|in short)\b", t, re.I)))
@feat("discourse")
def offers_more_help(t): return float(bool(re.search(r"(let me know|hope this helps|feel free to|happy to help|if you have any)", t, re.I)))
@feat("discourse")
def ends_with_question(t): return float(t.strip().endswith("?"))


def feature_frame(frame):
    return pd.DataFrame({k: [fn(t) for t in frame.output] for k, (_, fn) in FEATS.items()})


def top_ngrams(names, clf, k=60):
    """Top-k n-grams per family by coefficient, split into lexical (contain a letter) and symbol-only."""
    res = {}
    for i, f in enumerate(FAMILIES):
        grams = [names[j].replace("<nl>", "\\n") for j in np.argsort(-clf.coef_[i])[:k]]
        res[f] = {"lexical": [g for g in grams if re.search(r"[A-Za-z]", g.replace("\\n", ""))],
                  "symbol": [g for g in grams if not re.search(r"[A-Za-z]", g.replace("\\n", ""))]}
    return res


def main():
    out = {}
    # --- TF-IDF baseline, main split, three input modes (RQ1 / RQ2)
    out["tfidf_main"] = {}
    for mode in ("output", "input", "both"):
        m, vec, clf, pred = tfidf_lr(TRV, TE, mode)
        out["tfidf_main"][mode] = m
        print("tfidf", mode, round(m["accuracy"], 3), round(m["macro_f1"], 3), flush=True)
        if mode == "output":
            names = np.array(vec.get_feature_names_out())
            out["top_ngrams"] = top_ngrams(names, clf)

    # --- TF-IDF under the two simulated collection protocols (RQ2)
    import run_nn as R  # reuses the exact same confounded datasets as the CNN
    out["tfidf_confound"] = {}
    for kind in ("topic_biased", "random"):
        tr, va, te = R.confounded_frames(kind)
        trv = pd.concat([tr, va], ignore_index=True)
        out["tfidf_confound"][kind] = {}
        for mode in ("input", "output", "both"):
            m = tfidf_lr(trv, te, mode)[0]
            out["tfidf_confound"][kind][mode] = m
            print("conf", kind, mode, round(m["accuracy"], 3), flush=True)
        out["tfidf_confound"][kind]["majority"] = float(te.family.value_counts(normalize=True).max())

    # --- TF-IDF cross-task / cross-source references (RQ3)
    out["tfidf_taskshift"] = {}
    for exp in ("write", "tech"):
        for cond in ("ood", "id"):
            accs = []
            for fold in (0, 1):
                tr, va, te = R.task_shift_frames(exp, cond, fold)
                accs.append(tfidf_lr(pd.concat([tr, va]), te, "output")[0]["accuracy"])
            out["tfidf_taskshift"][f"{exp}_{cond}"] = accs
            print("task", exp, cond, np.round(accs, 3), flush=True)
    out["tfidf_loso"] = {}
    for s in ("helpful_base", "koala", "oasst", "selfinstruct", "vicuna"):
        trn = DF[DF.source != s]
        ten = DF[DF.source == s]
        out["tfidf_loso"][s] = tfidf_lr(trn, ten, "output")[0]["accuracy"]
    print("loso", out["tfidf_loso"], flush=True)

    # --- stylometric features (RQ4)
    Ftr, Fte = feature_frame(TRV), feature_frame(TE)
    Fall = feature_frame(DF)
    Fall["family"] = DF.family.values
    out["feature_means"] = Fall.groupby("family").mean().loc[FAMILIES].round(4).to_dict(orient="index")
    out["feature_medians"] = Fall.groupby("family").median().loc[FAMILIES].round(4).to_dict(orient="index")
    out["feature_groups"] = {k: g for k, (g, _) in FEATS.items()}
    Fall.to_csv(R.RESULTS / "stylometric_features.csv", index=False)
    ytr, yte = labels_of(TRV), labels_of(TE)
    groups = sorted(set(g for g, _ in FEATS.values()))
    out["feature_clf"] = {}
    for name, cols in [("all", list(FEATS))] + [(g, [k for k, (gg, _) in FEATS.items() if gg == g]) for g in groups] + \
                      [("all_but_" + g, [k for k, (gg, _) in FEATS.items() if gg != g]) for g in groups]:
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
        gb.fit(Ftr[cols], ytr)
        out["feature_clf"][name] = metrics(yte, gb.predict(Fte[cols]))
        print("feat", name, round(out["feature_clf"][name]["accuracy"], 3), flush=True)
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(Ftr, ytr)
    pi = permutation_importance(gb, Fte, yte, n_repeats=10, random_state=0, scoring="accuracy")
    out["perm_importance"] = dict(sorted(zip(Ftr.columns, pi.importances_mean.round(4).tolist()), key=lambda x: -x[1]))
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Ftr, ytr)
    out["feature_clf"]["all_logreg"] = metrics(yte, lr.predict(Fte))

    # --- how each family opens its answer (first 3 words)
    op = {}
    for f in FAMILIES:
        firsts = [" ".join(_words(t)[:2]) for t in DF[DF.family == f].output]
        c = Counter(firsts)
        op[f] = [(k, round(v / len(firsts), 3)) for k, v in c.most_common(6)]
    out["openers"] = op
    save_json(out, "sklearn_results.json")
    print("done")


if __name__ == "__main__":
    import sys
    if "--sweep" in sys.argv:  # RQ2: input-only accuracy as a function of how biased the collection is
        import json
        import run_nn as R
        from common import RESULTS
        res = json.load(open(RESULTS / "sklearn_results.json"))
        sweep = {}
        for p in (0.0, 0.3, 0.5, 0.7, 0.9, 1.0):
            accs = {"input": [], "output": [], "majority": []}
            for seed in range(5):
                tr, va, te = R.confounded_frames("topic_biased", seed=seed, p_pref=p)
                trv = pd.concat([tr, va], ignore_index=True)
                for mode in ("input", "output"):
                    accs[mode].append(tfidf_lr(trv, te, mode)[0]["accuracy"])
                accs["majority"].append(float(te.family.value_counts(normalize=True).max()))
            sweep[str(p)] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in accs.items()}
            print(p, {k: round(v[0], 3) for k, v in sweep[str(p)].items()}, flush=True)
        res["confound_sweep"] = sweep
        save_json(res, "sklearn_results.json")
    elif "--ngrams" in sys.argv:  # refresh only the n-gram table inside an existing results file
        import json
        from common import RESULTS
        res = json.load(open(RESULTS / "sklearn_results.json"))
        _, vec, clf, _ = tfidf_lr(TRV, TE, "output")
        res["top_ngrams"] = top_ngrams(np.array(vec.get_feature_names_out()), clf)
        save_json(res, "sklearn_results.json")
        print({f: v["lexical"][:8] for f, v in res["top_ngrams"].items()})
    else:
        main()
