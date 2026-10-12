"""Aggregate results/nn/*.json + results/sklearn_results.json into results/summary.json and figures/*.png."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FAMILIES, FIGS, RESULTS, load_data

NN = RESULTS / "nn"
FAM_COL = {"GPT": "#0072B2", "Claude": "#D55E00", "Gemini": "#009E73", "Llama": "#CC79A7", "Qwen": "#E69F00", "DeepSeek": "#56B4E9"}
ARCH_COL = {"CNN": "#0072B2", "LSTM": "#D55E00", "TF-IDF+LR": "#8C8C8C"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "axes.titlesize": 10, "axes.titleweight": "bold"})


def load(name):
    p = NN / f"{name}.json"
    return json.load(open(p)) if p.exists() else None


def ms(vals):
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (None, None, 0)


def write_markdown(S):
    """Short, human-readable summary of the headline numbers (results/RESULTS.md)."""
    f = lambda t: "n/a" if not t or t[0] is None else (f"{t[0]:.3f} ± {t[1]:.3f} ({t[2]} seeds)" if t[2] > 1 else f"{t[0]:.3f} (1 run)")
    M = S["main"]
    L = ["# Results at a glance", "", "Test accuracy, 6 classes (chance = 0.167). Prompt-level split: test prompts never seen in training.", "",
         "| View | TextCNN | BiLSTM | TF-IDF + LogReg |", "|---|---|---|---|"]
    for m, lab in [("output", "Output only (RQ1)"), ("input", "Input only (RQ2)"), ("both", "Input + output (RQ2)")]:
        L.append(f"| {lab} | {f(M.get(f'cnn_{m}', {}).get('acc'))} | {f(M.get(f'lstm_{m}', {}).get('acc'))} | {M[f'tfidf_{m}']['acc'][0]:.3f} |")
    L += ["", "RQ4 ablations (test accuracy after retraining on transformed text):", "", "| Text | TextCNN | BiLSTM |", "|---|---|---|"]
    for a in ("no_open_close", "no_format", "words_only", "content_only", "skeleton", "first64"):
        L.append(f"| {a} | {f(S['ablation'].get(f'cnn_{a}'))} | {f(S['ablation'].get(f'lstm_{a}'))} |")
    L += ["", f"Stylometric features only (gradient-boosted trees): {S['features']['clf']['all']:.3f}", "",
          "See report/Project_Report.pdf for RQ2 confound, RQ3 cross-task / cross-source / cross-version results and the full analysis."]
    (RESULTS / "RESULTS.md").write_text("\n".join(L) + "\n")


def main():
    FIGS.mkdir(exist_ok=True)
    S = {}
    sk = json.load(open(RESULTS / "sklearn_results.json"))
    df, splits = load_data()

    # ------------------------------------------------ dataset stats
    feats = pd.read_csv(RESULTS / "stylometric_features.csv")
    S["dataset"] = {
        "rows": len(df), "prompts": int(df.prompt_id.nunique()), "per_family": df.family.value_counts().to_dict(),
        "split_prompts": {k: len(v) for k, v in splits.items()},
        "split_rows": {k: len(v) * 6 for k, v in splits.items()},
        "source_prompts": df[df.family == "GPT"].source.value_counts().to_dict(),
        "task_prompts": df[df.family == "GPT"].task.value_counts().to_dict(),
        "median_words": feats.groupby("family").n_words.median().round(0).to_dict(),
        "median_chars": feats.groupby("family").n_chars.median().round(0).to_dict(),
        "input_median_words": int(df[df.family == "GPT"].input.str.split().str.len().median()),
    }
    from common import MAX_OUT, tokenize
    ntok = df.output.map(lambda t: len(tokenize(t)))
    S["dataset"]["median_tokens"] = ntok.groupby(df.family).median().to_dict()
    S["dataset"]["share_truncated"] = (ntok > MAX_OUT).groupby(df.family).mean().round(3).to_dict()

    # ------------------------------------------------ RQ1 / RQ2 main
    S["main"] = {}
    for arch in ("cnn", "lstm"):
        for mode in ("output", "input", "both"):
            runs = [load(f"rq1_{arch}_{mode}_s{s}") for s in range(3)]
            runs = [r for r in runs if r]
            if not runs:
                continue
            S["main"][f"{arch}_{mode}"] = {
                "acc": ms([r["test"]["accuracy"] for r in runs]), "f1": ms([r["test"]["macro_f1"] for r in runs]),
                "per_class_f1": np.mean([r["test"]["per_class_f1"] for r in runs], 0).round(3).tolist(),
                "confusion_sum": np.sum([r["test"]["confusion"] for r in runs], 0).tolist(),
                "epochs": [r["epochs"] for r in runs], "best_epoch": [r["best_epoch"] for r in runs],
                "params": runs[0]["params"], "vocab": runs[0]["vocab"], "train_sec": ms([r["train_sec"] for r in runs]),
                "n_train": runs[0]["n_train"], "n_test": runs[0]["n_test"],
                "sec_per_epoch": float(np.median([h["sec"] for r in runs for h in r["history"]])),
            }
    for mode, m in sk["tfidf_main"].items():
        S["main"][f"tfidf_{mode}"] = {"acc": (m["accuracy"], 0, 1), "f1": (m["macro_f1"], 0, 1), "per_class_f1": m["per_class_f1"]}
    r512 = load("rq1_cnn_output512_s0")
    if r512:
        S["cnn512"] = {"acc": r512["test"]["accuracy"], "f1": r512["test"]["macro_f1"],
                       "acc_256_s0": load("rq1_cnn_output_s0")["test"]["accuracy"]}

    # ensemble of CNN+LSTM (average probabilities is not stored; majority of seed predictions instead)
    # -> confusion-based summary of where errors go
    for arch in ("cnn", "lstm"):
        k = f"{arch}_output"
        if k in S["main"]:
            C = np.array(S["main"][k]["confusion_sum"], float)
            S["main"][k]["recall"] = (np.diag(C) / C.sum(1)).round(3).tolist()

    # ------------------------------------------------ RQ2 confounded
    S["confound"] = {}
    for kind in ("topic_biased", "random"):
        for mode in ("input", "output", "both"):
            r = load(f"rq2conf_cnn_{kind}_{mode}_s0")
            S["confound"][f"cnn_{kind}_{mode}"] = r["test"]["accuracy"] if r else None
            S["confound"][f"tfidf_{kind}_{mode}"] = sk["tfidf_confound"][kind][mode]["accuracy"]
        S["confound"][f"majority_{kind}"] = sk["tfidf_confound"][kind]["majority"]

    # ------------------------------------------------ RQ3
    S["taskshift"] = {}
    for arch in ("cnn", "lstm"):
        for exp in ("write", "tech"):
            for cond in ("ood", "id"):
                rs = [load(f"rq3task_{arch}_{exp}_{cond}_f{f}") for f in (0, 1)]
                S["taskshift"][f"{arch}_{exp}_{cond}"] = ms([r["test"]["accuracy"] if r else None for r in rs])
    for k, v in sk["tfidf_taskshift"].items():
        S["taskshift"][f"tfidf_{k}"] = ms(v)
    S["loso"] = {"tfidf": sk["tfidf_loso"], "cnn": {}}
    for s in ("helpful_base", "koala", "oasst", "selfinstruct", "vicuna"):
        r = load(f"rq3loso_cnn_{s}")
        S["loso"]["cnn"][s] = r["test"]["accuracy"] if r else None
    # in-distribution per-source accuracy of the RQ1 CNN (same prompts are never in train)
    src = df[df.family == "GPT"].set_index("prompt_id").source
    per_src = {}
    for s in range(3):
        r = load(f"rq1_cnn_output_s{s}")
        if not r:
            continue
        tp = r["test_pred"]
        d = pd.DataFrame(tp)
        d["source"] = d.prompt_id.map(src)
        for k, g in d.groupby("source"):
            per_src.setdefault(k, []).append(float((g.y == g.pred).mean()))
    S["loso"]["cnn_in_dist"] = {k: float(np.mean(v)) for k, v in per_src.items()}
    S["siblings"] = {}
    for arch in ("cnn", "lstm"):
        rs = [load(f"rq1_{arch}_output_s{s}") for s in range(3)]
        rs = [r for r in rs if r]
        if not rs:
            continue
        fams = list(rs[0]["siblings"]["recall_by_family"])
        S["siblings"][arch] = {
            "models": rs[0]["siblings"]["models"],
            "sib_recall": {f: float(np.mean([r["siblings"]["recall_by_family"][f] for r in rs])) for f in fams},
            "prim_recall": {f: float(np.mean([r["siblings"]["primary_recall_by_family"][f] for r in rs])) for f in fams},
            "sib_acc": ms([float(np.mean(list(r["siblings"]["recall_by_family"].values()))) for r in rs]),
            "prim_acc_4fam": ms([float(np.mean([r["siblings"]["primary_recall_by_family"][f] for f in fams])) for r in rs]),
            "pred_counts": np.sum([[r["siblings"]["pred_counts"][f] for f in fams] for r in rs], 0).tolist(),
        }

    # ------------------------------------------------ RQ4
    S["ablation"] = {}
    for arch in ("cnn", "lstm"):
        for a in ("no_format", "skeleton", "content_only", "words_only", "no_open_close", "first64"):
            rs = [load(f"rq4abl_{arch}_{a}_s{s}") for s in (0, 1)]
            S["ablation"][f"{arch}_{a}"] = ms([r["test"]["accuracy"] if r else None for r in rs])
    S["features"] = {"clf": {k: v["accuracy"] for k, v in sk["feature_clf"].items()},
                     "perm": sk["perm_importance"], "medians": sk["feature_medians"], "means": sk["feature_means"],
                     "groups": sk["feature_groups"]}
    S["top_ngrams"] = sk["top_ngrams"]
    S["typography"] = {f: {"curly_apostrophe": float(df[df.family == f].output.str.contains("\u2019").mean()),
                           "em_dash": float(df[df.family == f].output.str.contains("\u2014").mean()),
                           "starts_certainly": float(df[df.family == f].output.str.match(r"\W*Certainly").mean()),
                           "starts_here": float(df[df.family == f].output.str.match(r"\W*Here").mean()),
                           "apologize": float(df[df.family == f].output.str.contains("I apologize").mean()),
                           "what_a_question": float(df[df.family == f].output.str.contains(r"^What an? .{0,60}question", regex=True).mean()),
                           "cjk": float(df[df.family == f].output.str.contains(r"[\u4e00-\u9fff]").mean())}
                       for f in FAMILIES}
    S["openers"] = sk["openers"]
    if (RESULTS / "cnn_filters.json").exists():
        S["cnn_filters"] = json.load(open(RESULTS / "cnn_filters.json"))
    json.dump(S, open(RESULTS / "summary.json", "w"), indent=1)
    write_markdown(S)

    # ================================================= figures
    # 1. length distribution
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    data = [feats[feats.family == f].n_words.clip(lower=1) for f in FAMILIES]
    bp = ax[0].boxplot(data, tick_labels=FAMILIES, showfliers=False, patch_artist=True, widths=0.6)
    for p, f in zip(bp["boxes"], FAMILIES):
        p.set_facecolor(FAM_COL[f]); p.set_alpha(0.75)
    ax[0].set_ylabel("words per response"); ax[0].set_title("Response length")
    ax[0].tick_params(axis="x", rotation=30)
    fl = feats.groupby("family")[["frac_list_lines", "n_headers", "n_bold"]].mean().loc[FAMILIES]
    x = np.arange(len(FAMILIES))
    ax[1].bar(x - 0.27, fl.frac_list_lines, 0.27, label="share of lines that are list items", color="#555")
    ax[1].bar(x, fl.n_headers / 10, 0.27, label="markdown headers / 10", color="#999")
    ax[1].bar(x + 0.27, fl.n_bold / 10, 0.27, label="bold spans / 10", color="#ccc", edgecolor="#777", linewidth=0.5)
    ax[1].set_xticks(x, FAMILIES, rotation=30); ax[1].set_title("Formatting habits (means)"); ax[1].set_ylim(0, 1.05)
    ax[1].legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(FIGS / "fig1_dataset.png"); plt.close(fig)

    # 2. confusion matrices
    have = [a for a in ("cnn", "lstm") if f"{a}_output" in S["main"]]
    fig, axes = plt.subplots(1, len(have), figsize=(3.6 * len(have), 3.2))
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, have):
        C = np.array(S["main"][f"{a}_output"]["confusion_sum"], float)
        C = C / C.sum(1, keepdims=True)
        im = ax.imshow(C, cmap="Blues", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=7, color="white" if C[i, j] > 0.5 else "black")
        ax.set_xticks(range(6), FAMILIES, rotation=40); ax.set_yticks(range(6), FAMILIES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        acc = S["main"][f"{a}_output"]["acc"]
        ax.set_title(f"{a.upper()} (output only): acc {acc[0]:.3f}")
        ax.spines[:].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGS / "fig2_confusion.png"); plt.close(fig)

    # 3. training curves (seed 0)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for a in have:
        r = load(f"rq1_{a}_output_s0")
        h = pd.DataFrame(r["history"])
        ax[0].plot(h.epoch, h.train_loss, "-o", ms=3, color=ARCH_COL[a.upper()], label=a.upper())
        ax[1].plot(h.epoch, h.val_macro_f1, "-o", ms=3, color=ARCH_COL[a.upper()], label=a.upper())
        ax[1].axvline(r["best_epoch"], color=ARCH_COL[a.upper()], ls=":", lw=1)
    ax[0].set_title("Training loss"); ax[1].set_title("Validation macro-F1 (dotted = kept epoch)")
    for x_ in ax:
        x_.set_xlabel("epoch"); x_.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGS / "fig3_curves.png"); plt.close(fig)

    # 4. RQ2 bars
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    modes = ["input", "output", "both"]
    lab = {"input": "Input only", "output": "Output only", "both": "Input + Output"}
    for k, (name, key) in enumerate([("CNN", "cnn"), ("LSTM", "lstm"), ("TF-IDF+LR", "tfidf")]):
        vals = [S["main"].get(f"{key}_{m}", {}).get("acc", (np.nan, 0, 0)) for m in modes]
        ax.bar(np.arange(3) + (k - 1) * 0.26, [v[0] if v[0] is not None else np.nan for v in vals], 0.26,
               yerr=[v[1] or 0 for v in vals], capsize=2, color=ARCH_COL[name], label=name)
    ax.axhline(1 / 6, color="k", ls="--", lw=0.8); ax.text(-0.45, 1 / 6 + 0.012, "chance", fontsize=7, ha="left")
    ax.set_xticks(range(3), [lab[m] for m in modes]); ax.set_ylabel("test accuracy"); ax.set_ylim(0, 0.8)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper left"); ax.set_title("RQ2: which part of the pair identifies the LLM?")
    fig.tight_layout(); fig.savefig(FIGS / "fig4_rq2.png"); plt.close(fig)

    # 5. RQ2 confound
    S["confound_sweep"] = sk.get("confound_sweep", {})
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.5), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    ax[1].sharey(ax[0])
    for a_, kind, ttl in [(ax[0], "random", "Fair collection (random family)"),
                          (ax[1], "topic_biased", "Topic-biased collection (p = 0.7)")]:
        for k, (name, key) in enumerate([("CNN", "cnn"), ("TF-IDF+LR", "tfidf")]):
            vals = [S["confound"].get(f"{key}_{kind}_{m}") for m in modes]
            a_.bar(np.arange(3) + (k - 0.5) * 0.35, [v if v is not None else np.nan for v in vals], 0.35, color=ARCH_COL[name], label=name)
        a_.axhline(S["confound"][f"majority_{kind}"], color="k", ls="--", lw=0.8)
        a_.set_xticks(range(3), [lab[m] for m in modes]); a_.set_title(ttl, fontsize=8.5)
    ax[0].set_ylabel("test accuracy"); ax[0].legend(frameon=False, fontsize=7, ncol=2, loc="upper left"); ax[0].set_ylim(0, 0.68)
    for a_ in ax[:2]:
        a_.tick_params(axis="x", labelsize=7)
    sw = S["confound_sweep"]
    if sw:
        ps = sorted(sw, key=float)
        xs = [float(p) for p in ps]
        for key, col, lab_ in [("input", "#E69F00", "input only"), ("output", "#8C8C8C", "output only"), ("majority", "k", "majority class")]:
            m = np.array([sw[p][key][0] for p in ps]); sd = np.array([sw[p][key][1] for p in ps])
            ax[2].plot(xs, m, "-o" if key != "majority" else "--", ms=3, color=col, label=lab_, lw=1.2 if key != "majority" else 0.8)
            if key != "majority":
                ax[2].fill_between(xs, m - sd, m + sd, color=col, alpha=0.15, lw=0)
        ax[2].set_xlabel("p = P(topic's preferred family), 0 = fair"); ax[2].set_ylim(0, 0.75)
        ax[2].set_title("TF-IDF+LR vs. strength of bias", fontsize=8.5); ax[2].legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout(); fig.savefig(FIGS / "fig5_confound.png"); plt.close(fig)

    # 6. RQ3
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [1.2, 1.2, 1.3]})
    for i, (exp, ttl) in enumerate([("write", "Test on WRITING\n(train: technical + QA)"), ("tech", "Test on CODE/MATH\n(train: QA)")]):
        for k, (name, key) in enumerate([("CNN", "cnn"), ("LSTM", "lstm"), ("TF-IDF+LR", "tfidf")]):
            v = [S["taskshift"].get(f"{key}_{exp}_{c}", (None, 0, 0)) for c in ("id", "ood")]
            ax[i].bar(np.arange(2) + (k - 1) * 0.26, [x[0] if x[0] is not None else np.nan for x in v], 0.26,
                      yerr=[x[1] or 0 for x in v], capsize=2, color=ARCH_COL[name], label=name)
        ax[i].set_xticks([0, 1], ["in-domain\nin training", "unseen\ndomain"]); ax[i].set_title(ttl, fontsize=8.5)
        ax[i].set_ylim(0, 0.75); ax[i].axhline(1 / 6, color="k", ls="--", lw=0.8)
    ax[0].set_ylabel("test accuracy"); ax[0].legend(frameon=False, fontsize=7, loc="upper right")
    if "cnn" in S["siblings"]:
        sb = S["siblings"]["cnn"]
        fams = list(sb["sib_recall"])
        x = np.arange(len(fams))
        ax[2].bar(x - 0.2, [sb["prim_recall"][f] for f in fams], 0.4, color="#0072B2", label="same version as training")
        ax[2].bar(x + 0.2, [sb["sib_recall"][f] for f in fams], 0.4, color="#9ecae1", label="unseen sibling version")
        ax[2].set_xticks(x, fams); ax[2].set_title("CNN recall: new model versions", fontsize=8.5)
        ax[2].legend(frameon=False, fontsize=7); ax[2].set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(FIGS / "fig6_rq3.png"); plt.close(fig)

    # 7. LOSO
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    srcs = ["helpful_base", "koala", "oasst", "selfinstruct", "vicuna"]
    x = np.arange(len(srcs))
    ax.bar(x - 0.27, [S["loso"]["cnn_in_dist"].get(s, np.nan) for s in srcs], 0.27, color="#0072B2", label="CNN, source seen in training")
    ax.bar(x, [S["loso"]["cnn"][s] if S["loso"]["cnn"][s] is not None else np.nan for s in srcs], 0.27, color="#9ecae1", label="CNN, source held out")
    ax.bar(x + 0.27, [S["loso"]["tfidf"][s] for s in srcs], 0.27, color="#8C8C8C", label="TF-IDF+LR, source held out")
    ax.set_xticks(x, srcs); ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1.0)
    ax.axhline(1 / 6, color="k", ls="--", lw=0.8)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center"); ax.set_title("RQ3: leave-one-prompt-source-out", pad=4)
    fig.tight_layout(); fig.savefig(FIGS / "fig7_loso.png"); plt.close(fig)

    # 8. feature heatmap (z-scored family means)
    sel = ["n_words", "n_lines", "n_headers", "n_bullets", "n_numbered", "n_bold", "n_code_blocks", "frac_list_lines",
           "avg_word_len", "avg_sent_len", "ttr_first200", "rate_I", "rate_you", "rate_exclaim", "rate_colon",
           "rate_dash", "opens_affirmation", "has_conclusion", "offers_more_help", "ends_with_question"]
    M = pd.DataFrame(S["features"]["means"]).T.loc[FAMILIES, sel]
    Z = (M - M.mean()) / M.std(ddof=0).replace(0, 1)
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    im = ax.imshow(Z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_yticks(range(6), FAMILIES); ax.set_xticks(range(len(sel)), sel, rotation=55, ha="right", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.02, label="z-score across families")
    ax.set_title("RQ4: stylometric profile (colour = family mean relative to the other families)")
    ax.spines[:].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGS / "fig8_features.png"); plt.close(fig)

    # 9. ablations
    abl = [("full", "original text"), ("no_open_close", "drop first & last sentence"), ("no_format", "strip markdown + line breaks"),
           ("words_only", "words only (lower, no punct.)"), ("content_only", "content words only"),
           ("skeleton", "style skeleton (content words -> W)"), ("first64", "first 64 tokens only")]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    y = np.arange(len(abl))
    for k, (name, key) in enumerate([("CNN", "cnn"), ("LSTM", "lstm")]):
        vals = []
        for a, _ in abl:
            if a == "full":
                vals.append(S["main"].get(f"{key}_output", {}).get("acc", (np.nan, 0, 0)))
            else:
                vals.append(S["ablation"].get(f"{key}_{a}", (None, 0, 0)))
        ax.barh(y + (k - 0.5) * 0.38, [v[0] if v[0] is not None else np.nan for v in vals], 0.38,
                xerr=[v[1] or 0 for v in vals], capsize=2, color=ARCH_COL[name], label=name)
    ax.set_yticks(y, [b for _, b in abl]); ax.invert_yaxis(); ax.axvline(1 / 6, color="k", ls="--", lw=0.8)
    ax.set_xlabel("test accuracy"); ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.set_xlim(0, 0.75)
    ax.set_title("RQ4: what happens when a signal is removed?")
    fig.tight_layout(); fig.savefig(FIGS / "fig9_ablation.png"); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
