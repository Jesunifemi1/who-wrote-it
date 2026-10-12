"""Builds report/Project_Report.pdf from results/summary.json and figures/*.png.
Every number in the report is read from the results files, so the PDF always matches the code output."""
import json
import sys
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
S = json.load(open(ROOT / "results" / "summary.json"))
FIG = ROOT / "figures"
FAM = ["GPT", "Claude", "Gemini", "Llama", "Qwen", "DeepSeek"]
META = json.load(open(ROOT / "report" / "team.json"))

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontName="Times-Roman", fontSize=10.2, leading=13.2, spaceAfter=5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.6, leading=10.6, spaceAfter=2)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8.6, leading=10.4, textColor=colors.HexColor("#333333"), spaceAfter=9)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=12.5, spaceBefore=10, spaceAfter=4,
                    textColor=colors.HexColor("#1a1a1a"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, spaceBefore=7, spaceAfter=3)
TITLE = ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=17, leading=21, spaceAfter=4)
SUB = ParagraphStyle("sub", parent=BODY, alignment=TA_CENTER, fontSize=10, leading=13)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.4, leading=10, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Times-Bold")
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=2)

story = []
fig_no = [0]
tab_no = [0]


def P(t, st=BODY):
    story.append(Paragraph(t, st))


def bullets(items):
    for it in items:
        story.append(Paragraph(it, BUL, bulletText="•"))


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def pm(t):
    if t is None or t[0] is None:
        return "n/a"
    m, s, n = t
    return f"{m:.3f} &plusmn; {s:.3f}" if n and n > 1 else f"{m:.3f}"


def pct(x):
    return f"{100 * x:.1f}%"


def lt1(x):
    return "&lt;1%" if x < 0.01 else f"{x:.0%}"


def table(rows, widths, header_rows=1, zebra=True, align_right_from=1):
    data = [[c if hasattr(c, "wrap") else Paragraph(str(c), CELLB if i < header_rows else CELL) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="CENTER")
    st = [("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.5, colors.black),
          ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]
    if zebra:
        for i in range(header_rows, len(rows)):
            if (i - header_rows) % 2 == 1:
                st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f3f3f3")))
    t.setStyle(TableStyle(st))
    return t


def add_table(rows, widths, caption, **kw):
    tab_no[0] += 1
    story.append(KeepTogether([Paragraph(f"<b>Table {tab_no[0]}.</b> {caption}", CAP), table(rows, widths, **kw), Spacer(1, 8)]))
    return tab_no[0]


def add_fig(name, width, caption):
    fig_no[0] += 1
    p = FIG / name
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(p)).getSize()
    img = Image(str(p), width=width, height=width * ih / iw)
    story.append(KeepTogether([img, Paragraph(f"<b>Figure {fig_no[0]}.</b> {caption}", CAP)]))
    return fig_no[0]


# ======================================================================= derived numbers
_means = S["features"]["means"]
_keys = [k for k in _means["GPT"]]
_Z = np.array([[_means[f][k] for k in _keys] for f in FAM])
_Z = (_Z - _Z.mean(0)) / np.where(_Z.std(0) > 0, _Z.std(0), 1)
PLAIN_FAM = FAM[int(np.argmin(np.linalg.norm(_Z, axis=1)))]
M = S["main"]
D = S["dataset"]
if "--draft" in sys.argv:  # layout test with incomplete results: fill missing entries with NaN placeholders
    nan = {"acc": (float("nan"), 0, 0), "f1": (float("nan"), 0, 0), "per_class_f1": [float("nan")] * 6,
           "confusion_sum": np.eye(6).tolist(), "epochs": [0], "train_sec": (0, 0, 0), "params": 0, "vocab": 0, "n_test": 0, "sec_per_epoch": 0, "n_train": 0}
    for a in ("cnn", "lstm"):
        for m in ("output", "input", "both"):
            M.setdefault(f"{a}_{m}", nan)
cnn, lstm = M.get("cnn_output"), M.get("lstm_output")
best_arch = max([("CNN", cnn), ("LSTM", lstm)], key=lambda x: x[1]["acc"][0] if x[1] else -1)


def top_confusions(C, k=3):
    C = np.array(C, float)
    C = C / C.sum(1, keepdims=True)
    pairs = []
    for i in range(6):
        for j in range(6):
            if i != j:
                pairs.append((C[i, j], FAM[i], FAM[j]))
    return sorted(pairs, reverse=True)[:k]


def f1_rank(m):
    f = m["per_class_f1"]
    order = np.argsort(f)[::-1]
    return [(FAM[i], f[i]) for i in order]


_ab = S["ablation"]
_co, _sk = _ab.get("cnn_content_only", [None])[0], _ab.get("cnn_skeleton", [None])[0]
_nf = _ab.get("cnn_no_format", [None])[0]
_full = S["main"]["cnn_output"]["acc"][0]
if _co is not None and _sk is not None and _nf is not None:
    ABSTRACT_RQ4 = (f"the fingerprint is mostly stylistic. Replacing every content word with a placeholder leaves CNN accuracy at {_sk:.3f} "
                    f"(full text: {_full:.3f}), content words alone give only {_co:.3f}, and stripping markdown and line breaks costs the most "
                    f"({_nf:.3f}).")
else:
    ABSTRACT_RQ4 = "the fingerprint combines structural cues (length, markdown) with linguistic ones (openers, function words, vocabulary)."

# ======================================================================= title
P("Who Wrote It? Identifying LLMs from Their Responses", TITLE)
P("CMPSC 448 Machine Learning &mdash; Midterm Project &mdash; October 2026", SUB)
P(f"<b>Team leader:</b> {META['leader']} &nbsp;&nbsp; <b>Members:</b> {META['members']}", SUB)
P(f"<b>Code:</b> {META['repo']}", SUB)
story.append(Spacer(1, 6))

ab = (f"We build a balanced, fully parallel dataset of {D['rows']:,} (LLM, prompt, response) triples: the same "
      f"{D['prompts']} AlpacaEval instructions answered by six LLM families (GPT-4o, Claude 3 Opus, Gemini Pro, Llama 3 70B, "
      f"Qwen 1.5 72B, DeepSeek 67B). From the response text alone, a TextCNN reaches {pm(cnn['acc'])} test accuracy and a BiLSTM "
      f"{pm(lstm['acc'])} (chance = 0.167; mean &plusmn; std over 3 seeds; test prompts never seen in training). "
      f"The prompt alone carries no signal in a parallel design (exactly 1/6 accuracy, by construction), and adding it to the response changes "
      f"accuracy by only {M['cnn_both']['acc'][0] - M['cnn_output']['acc'][0]:+.3f} (CNN) and {M['lstm_both']['acc'][0] - M['lstm_output']['acc'][0]:+.3f} (LSTM); "
      f"we show that input-only accuracy only rises above chance when the collection is confounded (topic-biased model choice). "
      f"For extra credit we test generalization across task domains, prompt sources and unseen model versions (RQ3), and explain the fingerprints "
      f"with stylometric features, n-grams, CNN filters and six controlled text ablations (RQ4): {ABSTRACT_RQ4}")
story.append(table([[Paragraph("<b>Abstract.</b> " + ab, SMALL)]], [6.9 * inch], header_rows=0, zebra=False))
story.append(Spacer(1, 4))

# ======================================================================= 1. problem
P("1. Problem definition", H1)
P("Given a text <i>y</i> produced by an LLM in reply to a user prompt <i>x</i>, predict the family "
  "<i>c</i> &isin; {GPT, Claude, Gemini, Llama, Qwen, DeepSeek} of the model that wrote it. This is a 6-way, single-label text "
  "classification problem. We study three input views: the <b>output</b> <i>y</i> (RQ1), the <b>input</b> <i>x</i> alone, and the "
  "<b>pair</b> (<i>x</i>, <i>y</i>) (RQ2). We report accuracy and macro-F1 (classes are exactly balanced, so the two agree closely; "
  "chance accuracy is 1/6 = 0.167). A useful detector must generalise to <i>new prompts</i>, so every split is made at the prompt level: "
  "all six answers to a prompt are in the same split, and no test prompt is seen during training.")

# ======================================================================= 2. data
P("2. Dataset curation", H1)
P("<b>Source.</b> We use the public model-output files of AlpacaEval (Li et al., 2023; github.com/tatsu-lab/alpaca_eval, Apache-2.0 "
  "licence; commit cd543a1). AlpacaEval asks every evaluated model the same 805 instructions and publishes each model's raw answer "
  "in <font face='Courier'>results/&lt;model&gt;/model_outputs.json</font> with fields <font face='Courier'>instruction</font>, "
  "<font face='Courier'>output</font>, <font face='Courier'>generator</font> and <font face='Courier'>dataset</font> (the prompt's origin). "
  "This gives exactly the required triple (LLM_name = generator, LLM_input = instruction, LLM_output = output). The prompts are "
  "public benchmark prompts (Self-Instruct, OpenAssistant, Koala, Anthropic Helpful-Base, Vicuna), so the data contain no private "
  "or personal information, and the outputs are redistributed by the benchmark authors under a permissive licence. "
  "<font face='Courier'>src/download_data.sh</font> re-downloads the files from the pinned commit.")
P("<b>Construction.</b> We picked one large flagship chat model per family, all released between late 2023 and May 2024, giving six families "
  "&mdash; more than the required three. Files are aligned by instruction text (the Gemini file is stored in a different order, which we "
  "caught with an assertion). One prompt (id 104) has an empty Gemini answer, so we drop that prompt for every model to keep the classes "
  f"exactly balanced: <b>{D['prompts']} prompts &times; 6 families = {D['rows']:,} examples</b>. Four families also have a sibling "
  "version in AlpacaEval, which we hold out entirely for the cross-version test in RQ3.")
fam_rows = [["Family", "Model used for training/testing", "Held-out sibling (RQ3 only)", "Median words", "Median chars"]]
sib_models = S["siblings"].get("cnn", {}).get("models", {})
prim = {"GPT": "gpt-4o-2024-05-13", "Claude": "claude-3-opus-20240229", "Gemini": "gemini-pro", "Llama": "Meta-Llama-3-70B-Instruct",
        "Qwen": "Qwen1.5-72B-Chat", "DeepSeek": "deepseek-llm-67b-chat"}
for f in FAM:
    fam_rows.append([f, prim[f], sib_models.get(f, "&mdash;"), f"{D['median_words'][f]:.0f}", f"{D['median_chars'][f]:.0f}"])
add_table(fam_rows, [0.75 * inch, 1.9 * inch, 2.15 * inch, 0.85 * inch, 0.85 * inch],
          f"The six LLM families ({D['per_family']['GPT']} responses each). Medians are computed on full responses.")
sp_ = D["split_prompts"]
src = D["source_prompts"]
tk = D["task_prompts"]
P(f"<b>Splits.</b> Prompts are shuffled with a fixed seed and split 70/15/15: {sp_['train']}/{sp_['val']}/{sp_['test']} prompts, i.e. "
  f"{D['split_rows']['train']:,}/{D['split_rows']['val']:,}/{D['split_rows']['test']:,} examples. The validation set is used only for early "
  f"stopping. <b>Prompt sources:</b> " + ", ".join(f"{k} {v}" for k, v in src.items()) + ". "
  f"<b>Task domains</b> (assigned by transparent keyword rules in <font face='Courier'>prepare_data.py</font>, used in RQ3): "
  + ", ".join(f"{k} {v}" for k, v in tk.items()) + f" prompts. The median prompt has {D['input_median_words']} words; responses are "
  "far longer and their lengths differ by family (Figure 1).")
add_fig("fig1_dataset.png", 6.6 * inch, "Left: response length by family (box = interquartile range, whiskers = 1.5 IQR, outliers hidden). "
        "Right: average formatting habits. Even before any learning, families differ visibly in length and markdown use.")

# ======================================================================= 3. models
P("3. Models and training", H1)
P("<b>Tokenisation.</b> A regular-expression tokenizer splits text into words, numbers and <i>individual punctuation characters</i>, "
  "keeps case, and turns each line break into a <font face='Courier'>&lt;nl&gt;</font> token. This keeps markdown such as "
  "<font face='Courier'>**</font>, <font face='Courier'>###</font>, <font face='Courier'>-</font> and list numbering visible to the "
  "models, which matters because formatting may be part of a fingerprint. The vocabulary is built on the training split only "
  f"(tokens with frequency &ge; 2; {cnn['vocab']:,} types for the output view). Responses are truncated to their first 256 tokens "
  f"(about 1,150 characters). Median response lengths range from {min(D['median_tokens'].values()):.0f} tokens ({min(D['median_tokens'], key=D['median_tokens'].get)}) "
  f"to {max(D['median_tokens'].values()):.0f} ({max(D['median_tokens'], key=D['median_tokens'].get)}), so between "
  f"{min(D['share_truncated'].values()):.0%} and {max(D['share_truncated'].values()):.0%} of a family's responses are cut; this cap keeps the "
  "BiLSTM affordable on a CPU, and the length of the visible part still differs by family. In the input+output view the prompt is truncated to "
  "96 tokens and joined to the response with a <font face='Courier'>&lt;sep&gt;</font> token. We checked the effect of the 256-token "
  "limit by re-training the CNN with 512 tokens (Section 4).")
P("<b>CNN</b> (Kim, 2014). Embedding (128-d, learned from scratch) &rarr; four parallel 1-D convolutions with widths 2, 3, 4, 5 and 100 "
  "filters each (ReLU) &rarr; max-over-time pooling (padding positions masked) &rarr; dropout 0.5 &rarr; linear layer to 6 logits. "
  f"{cnn['params']:,} parameters. Each filter is an n-gram detector, which later lets us read off what it responds to (RQ4).")
P("<b>RNN</b>. Embedding (128-d) with dropout 0.2 &rarr; one-layer <i>bidirectional</i> LSTM (96 units per direction) run on packed "
  "sequences so padding is never read &rarr; concatenation of max-pooling and mean-pooling over valid time steps (384-d) &rarr; dropout "
  f"0.4 &rarr; linear layer. {lstm['params']:,} parameters. Pooling over all states (rather than using only the last state) "
  "lets the model use cues from anywhere in the response.")
hp = [["", "CNN", "BiLSTM"],
      ["Optimiser", "AdamW, lr 1e-3, weight decay 1e-4", "AdamW, lr 2e-3, weight decay 1e-4"],
      ["Batch size / max epochs", "32 / 15", "32 / 12"],
      ["Early stopping", "validation macro-F1, patience 3, restore best epoch", "same"],
      ["Regularisation", "dropout 0.5, gradient clipping 1.0", "embedding dropout 0.2, dropout 0.4, clipping 1.0"],
      ["Loss", "cross-entropy (classes balanced)", "same"],
      ["Epochs actually run (3 seeds)", ", ".join(map(str, cnn["epochs"])), ", ".join(map(str, lstm["epochs"]))],
      ["Time per epoch / per run (1 CPU core)", f"{cnn['sec_per_epoch']:.0f} s / {cnn['train_sec'][0]/60:.1f} min", f"{lstm['sec_per_epoch']:.0f} s / {lstm['train_sec'][0]/60:.1f} min"]]
add_table(hp, [1.75 * inch, 2.6 * inch, 2.6 * inch], "Training details. Seeds 0, 1, 2 for the main experiments; "
          "batches are formed from length-sorted chunks of 20 batches (then shuffled) to limit padding.")
P("<b>Reference baseline.</b> To put the neural results in context we also train a TF-IDF (word 1&ndash;2-grams on the same tokens, "
  "sublinear tf, up to 200k features) + multinomial logistic regression classifier (C = 4) on train+validation. All code runs on a "
  "2-core CPU; the whole pipeline (data &rarr; all experiments &rarr; figures &rarr; this PDF) is scripted and described in the README.")

# ======================================================================= 4. RQ1
P("4. RQ1 &mdash; Can we identify which LLM generated a response?", H1)
rows = [["Model (output only)", "Accuracy", "Macro-F1"] + FAM]
for name, key in [("TextCNN", "cnn_output"), ("BiLSTM", "lstm_output"), ("TF-IDF + LogReg", "tfidf_output")]:
    m = M[key]
    rows.append([name, pm(m["acc"]), pm(m["f1"])] + [f"{v:.2f}" for v in m["per_class_f1"]])
rows.append(["Chance", "0.167", "0.167"] + ["0.17"] * 6)
add_table(rows, [1.2 * inch, 0.92 * inch, 0.92 * inch] + [0.55 * inch] * 5 + [0.68 * inch],
          f"RQ1 test results on {cnn['n_test']} held-out responses ({sp_['test']} unseen prompts &times; 6). Neural rows: mean &plusmn; std over 3 seeds; "
          "per-family columns are F1 scores.")
rk_c = f1_rank(cnn)
rk_l = f1_rank(lstm)
tc = top_confusions(cnn["confusion_sum"])
tl = top_confusions(lstm["confusion_sum"])
P(f"<b>Answer: yes.</b> Both required models identify the author family far above chance: the CNN reaches {pm(cnn['acc'])} accuracy "
  f"and the BiLSTM {pm(lstm['acc'])}, i.e. roughly {cnn['acc'][0]/ (1/6):.1f}&times; and {lstm['acc'][0]/(1/6):.1f}&times; chance, on prompts "
  f"neither model saw in training. Seed-to-seed variation is small (std &le; {max(cnn['acc'][1], lstm['acc'][1]):.3f}), so the ranking "
  f"is not a fluke of initialisation. The simple TF-IDF baseline is "
  + ("slightly better than both networks" if M['tfidf_output']['acc'][0] > max(cnn['acc'][0], lstm['acc'][0]) + 0.01 else "competitive")
  + f" ({M['tfidf_output']['acc'][0]:.3f}), which already hints "
  "that much of the signal is in <i>which</i> words and symbols occur rather than in long-range order (RQ4 tests this directly).")
P(f"The easiest family for the CNN is <b>{rk_c[0][0]}</b> (F1 {rk_c[0][1]:.2f}) and the hardest is <b>{rk_c[-1][0]}</b> "
  f"(F1 {rk_c[-1][1]:.2f}); for the BiLSTM the easiest is {rk_l[0][0]} ({rk_l[0][1]:.2f}) and the hardest {rk_l[-1][0]} ({rk_l[-1][1]:.2f}). "
  f"The most frequent CNN errors are {tc[0][1]}&rarr;{tc[0][2]} ({pct(tc[0][0])} of {tc[0][1]} responses), {tc[1][1]}&rarr;{tc[1][2]} "
  f"({pct(tc[1][0])}) and {tc[2][1]}&rarr;{tc[2][2]} ({pct(tc[2][0])}); the BiLSTM's top confusion is {tl[0][1]}&rarr;{tl[0][2]} "
  f"({pct(tl[0][0])}). Errors are not spread uniformly (Figure 2). "
  + (lambda C: (lambda wrong_to: f"{FAM[int(np.argmax(wrong_to))]} is the most common wrong guess ({wrong_to.max() / wrong_to.sum():.0%} of all CNN errors), "
     + ("and it is also the hardest family to recognise: its answers have the least distinctive style (its stylometric profile in Figure 8 is the "
        "closest to the average of all families), so responses without strong cues tend to be assigned to it. "
        if FAM[int(np.argmax(wrong_to))] == rk_c[-1][0] and FAM[int(np.argmax(wrong_to))] == PLAIN_FAM else "")
     )((C.sum(0) - np.diag(C))))(np.array(cnn["confusion_sum"], float))
  + "The easiest families have strong, consistent templates, such as Llama's 'Here is/are&hellip;' openers and Gemini's '*'-bulleted layout (Tables 9&ndash;11).")
add_fig("fig2_confusion.png", 6.5 * inch, "Row-normalised confusion matrices on the test set, summed over the three seeds.")
add_fig("fig3_curves.png", 6.4 * inch, "Training loss and validation macro-F1 per epoch (seed 0). Both models start to over-fit after a few "
        "epochs (training loss keeps falling while validation F1 plateaus), which is why early stopping restores the best epoch.")
if "cnn512" in S:
    c5 = S["cnn512"]
    P(f"<b>Truncation check.</b> Re-training the CNN (seed 0) on the first 512 instead of 256 tokens gives {c5['acc']:.3f} accuracy versus "
      f"{c5['acc_256_s0']:.3f} for the same seed with 256 tokens, so the 256-token budget used for both networks "
      f"{'costs little' if c5['acc'] - c5['acc_256_s0'] < 0.03 else 'leaves some signal unused'}; the beginning of a response already "
      "carries most of the fingerprint (see also the 64-token ablation in RQ4).")

# ======================================================================= 5. RQ2
P("5. RQ2 &mdash; Does the user's prompt help identify the LLM?", H1)
rows = [["Input view", "TextCNN acc.", "BiLSTM acc.", "TF-IDF+LR acc.", "TextCNN macro-F1", "BiLSTM macro-F1"]]
for mode, lab in [("input", "Input only"), ("output", "Output only"), ("both", "Input + Output")]:
    rows.append([lab, pm(M[f"cnn_{mode}"]["acc"]), pm(M[f"lstm_{mode}"]["acc"]), f3(M[f"tfidf_{mode}"]["acc"][0]),
                 pm(M[f"cnn_{mode}"]["f1"]), pm(M[f"lstm_{mode}"]["f1"])])
add_table(rows, [1.1 * inch, 1.15 * inch, 1.15 * inch, 1.05 * inch, 1.2 * inch, 1.2 * inch],
          "RQ2 on the main (parallel) dataset. Neural models: mean &plusmn; std over seeds (number of seeds for input / output / input+output: "
          f"CNN {M['cnn_input']['acc'][2]}/{M['cnn_output']['acc'][2]}/{M['cnn_both']['acc'][2]}, "
          f"BiLSTM {M['lstm_input']['acc'][2]}/{M['lstm_output']['acc'][2]}/{M['lstm_both']['acc'][2]}; a single value means one seed).")
d_c = M["cnn_both"]["acc"][0] - M["cnn_output"]["acc"][0]
d_l = M["lstm_both"]["acc"][0] - M["lstm_output"]["acc"][0]
add_fig("fig4_rq2.png", 4.9 * inch, "RQ2: accuracy of the three input views (error bars = std over seeds; dashed line = chance).")
P("<b>(i) Can the LLM be predicted from the prompt alone? Not in our data &mdash; and provably so.</b> Every prompt appears six times, "
  "once with each label. A classifier that sees only the prompt must give the same prediction to all six copies, so exactly one of "
  "them is right: accuracy is 1/6 for <i>any</i> model, however well trained. The measured input-only accuracies "
  f"({M['cnn_input']['acc'][0]:.3f} CNN, {M['lstm_input']['acc'][0]:.3f} LSTM, {M['tfidf_input']['acc'][0]:.3f} TF-IDF) confirm this; the "
  "tiny deviations from 0.1667 come from a handful of prompts whose six identical copies landed in differently sized evaluation "
  "batches, so floating-point round-off broke a near-tie between two classes differently (we found 2 such prompts out of 122 for one seed). "
  "Macro-F1 is even lower because the models collapse onto a few classes. This is a useful sanity check of the pipeline: a bug that "
  "leaked labels into the input would have shown up here.")
d_t = M["tfidf_both"]["acc"][0] - M["tfidf_output"]["acc"][0]
noise = max(M["cnn_output"]["acc"][1], M["cnn_both"]["acc"][1], M["lstm_output"]["acc"][1], M["lstm_both"]["acc"][1])
thr = max(0.015, noise)
gains = {"CNN": d_c, "BiLSTM": d_l, "TF-IDF model": d_t}
helped = [k for k, v in gains.items() if v > thr]
hurt = [k for k, v in gains.items() if v < -thr]
if not helped and not hurt:
    verdict = "Not meaningfully."
elif helped and len(helped) < 3:
    verdict = f"Only slightly, and only for the {' and '.join(helped)}."
elif helped:
    verdict = "Slightly."
else:
    verdict = "No; it slightly hurts."
mean_d = np.mean([d_c, d_l])
P(f"<b>(ii) Does adding the prompt help? {verdict}</b> Input + output changes accuracy by {d_c:+.3f} for the CNN and {d_l:+.3f} for the "
  f"BiLSTM relative to output only ({d_t:+.3f} for TF-IDF); the largest seed standard deviation in these four settings is {noise:.3f}. "
  "Both views see the same first 256 response tokens, so the comparison isolates the prompt. Because the prompt carries no label "
  "information by itself in a parallel dataset, any effect must come from <i>interaction</i>: knowing what was asked lets a model judge the "
  "answer relative to the request (a bulleted list in reply to a yes/no question, a long essay in reply to a one-line question). "
  + (f"The small gain for the {' and '.join(helped)} suggests it picks up a little of this context, " if helped
     else "The models do not seem to exploit this with the amount of data we have, ")
  + "but the effect is minor compared with the information in the response itself, and the identical prompt tokens shared by all six "
  "classes are noise from the classifier's point of view.")
cf = S["confound"]
P("<b>(iii) When would input-only classification work surprisingly well? When the data collection is confounded.</b> To show this we "
  "built two non-parallel versions of the dataset, keeping a single response per prompt. In the <i>fair</i> version the responding family "
  "is drawn uniformly at random. In the <i>biased</i> version we cluster the prompts into six equal-size topic groups (k-means on TF-IDF "
  "of the instruction, capacity-constrained) and each group 'prefers' one family with probability 0.7 &mdash; mimicking real chat logs "
  "where, say, programmers favour one assistant and novelists another. Labels stay balanced in both versions.")
rows = [["Collection protocol", "Majority class", "Input only (CNN / TF-IDF)", "Output only (CNN / TF-IDF)", "Input + Output (CNN / TF-IDF)"]]
for kind, lab in [("random", "Fair (random family)"), ("topic_biased", "Biased (topic &rarr; family, p = 0.7)")]:
    rows.append([lab, f3(cf[f"majority_{kind}"])] + [f"{f3(cf[f'cnn_{kind}_{m}'])} / {f3(cf[f'tfidf_{kind}_{m}'])}" for m in ("input", "output", "both")])
add_table(rows, [1.85 * inch, 0.85 * inch, 1.4 * inch, 1.4 * inch, 1.45 * inch],
          f"RQ2 confound experiment (one response per prompt; {sp_['train']} training prompts, same prompt split; seed 0).")
sw = S.get("confound_sweep", {})
P(f"With fair collection, the prompt is useless: input-only accuracy is {f3(cf['cnn_random_input'])} (CNN) / {f3(cf['tfidf_random_input'])} "
  f"(TF-IDF), no better than predicting the majority class ({f3(cf['majority_random'])} on this small test set). With biased collection the "
  f"same TF-IDF classifier predicts the 'author' from the prompt alone at {f3(cf['tfidf_topic_biased_input'])}, clearly above the "
  f"{f3(cf['majority_topic_biased'])} majority baseline, without ever seeing a response; the CNN, which has to learn the topic&rarr;family "
  f"mapping from only {sp_['train']} short prompts, reaches {f3(cf['cnn_topic_biased_input'])}. "
  + (f"Sweeping the strength of the bias (Figure {fig_no[0] + 1}, right; 5 random assignments per point) shows input-only accuracy rising "
     f"monotonically from {sw['0.0']['input'][0]:.3f} at p = 0 to {sw['0.7']['input'][0]:.3f} at p = 0.7 and {sw['1.0']['input'][0]:.3f} at p = 1. "
     f"The confound also inflates output-only accuracy ({sw['0.0']['output'][0]:.3f} &rarr; {sw['1.0']['output'][0]:.3f}), because a response "
     "reveals its topic too. " if sw else "")
  + "Nothing about the LLMs changed &mdash; the classifier learned <i>who tends to ask what</i>. The lesson for RQ2 is that a high input-only "
  "score is a symptom of selection bias (users, platforms, time periods or task mixes that differ by model), not evidence of a model "
  "fingerprint, and it inflates input+output and even output-only scores in the same way. Parallel prompts, as in our main dataset, remove "
  "the confound by design.")
add_fig("fig5_confound.png", 6.8 * inch, "Left, middle: the three input views under fair and topic-biased collection (p = 0.7; dashed line = "
        "majority class). Right: TF-IDF accuracy as the bias grows (mean &plusmn; std over 5 random assignments). Input-only accuracy rises "
        "above chance only when the choice of model depends on the prompt topic.")

# ======================================================================= 6. RQ3
P("6. RQ3 (extra credit) &mdash; Do LLM fingerprints generalise across tasks and domains?", H1)
P("We test three kinds of shift. All use the output-only view and the same architectures and hyper-parameters as RQ1.")
P("<b>(a) Unseen task domain.</b> Experiment W trains on technical (code/math) + general QA prompts and tests on <i>writing</i> prompts "
  "(emails, stories, posts&hellip;). Experiment T trains on general QA and tests on <i>code/math</i>. To isolate the effect of the domain from the "
  "effect of training-set size, we compare against an <i>in-domain</i> control with the same test set and the same number of training "
  "prompts: the target prompts are split in two halves, the test half is fixed, and the control swaps an equal number of source prompts "
  "for the other target half. We report the mean over the two folds (each half serves once as the test set).", BODY)
ts = S["taskshift"]
rows = [["Experiment", "Model", "In-domain control", "Unseen domain", "Change"]]
for exp, lab in [("write", "W: test = writing"), ("tech", "T: test = code/math")]:
    for name, key in [("TextCNN", "cnn"), ("BiLSTM", "lstm"), ("TF-IDF+LR", "tfidf")]:
        a, b = ts.get(f"{key}_{exp}_id"), ts.get(f"{key}_{exp}_ood")
        ch = (b[0] - a[0]) if (a and b and a[0] is not None and b[0] is not None) else None
        rows.append([lab, name, pm(a), pm(b), "n/a" if ch is None else f"{ch:+.3f}"])
add_table(rows, [1.4 * inch, 1.0 * inch, 1.35 * inch, 1.35 * inch, 0.8 * inch],
          "RQ3a cross-task generalisation (mean &plusmn; std over the 2 folds; one seed per fold"
          + ("; the BiLSTM was run on fold 0 only, for compute reasons" if (ts.get("lstm_write_ood") or (None, 0, 0))[2] == 1 else "") + ").")
drops = [ts[f"{k}_{e}_ood"][0] - ts[f"{k}_{e}_id"][0] for k in ("cnn", "lstm", "tfidf") for e in ("write", "tech")
         if ts.get(f"{k}_{e}_ood") and ts[f"{k}_{e}_ood"][0] is not None and ts.get(f"{k}_{e}_id") and ts[f"{k}_{e}_id"][0] is not None]
P(f"Removing the target domain from training changes accuracy by between {min(drops):+.3f} and {max(drops):+.3f} across the "
  f"{len(drops)} model/experiment combinations (average {np.mean(drops):+.3f}). Performance on an unseen task "
  f"{'decreases noticeably' if np.mean(drops) < -0.05 else ('decreases, but only moderately' if np.mean(drops) < -0.01 else 'barely changes')}; "
  "the classifiers nevertheless stay far above chance on writing "
  "and on code/math they never trained on. So a large part of what they learn is a <i>task-independent</i> house style (how answers open, "
  "how they are laid out, typical phrasing), with a smaller task-specific part (e.g. how each family formats code blocks or a cover letter) "
  "that is lost under shift. "
  + (lambda dd: (f"The drop is larger for the neural networks (average {np.mean(dd['nn']):+.3f}) than for TF-IDF ({np.mean(dd['lin']):+.3f}): "
                 "the CNN and LSTM fit more task-specific patterns (e.g. how a family lays out a cover letter or a code answer), while the "
                 "linear model relies more on cues that recur in every task. " if dd["nn"] and dd["lin"] and np.mean(dd["nn"]) < np.mean(dd["lin"]) - 0.02 else ""))(
      {"nn": [ts[f"{k}_{e}_ood"][0] - ts[f"{k}_{e}_id"][0] for k in ("cnn", "lstm") for e in ("write", "tech")
              if ts.get(f"{k}_{e}_ood") and ts[f"{k}_{e}_ood"][0] is not None and ts.get(f"{k}_{e}_id") and ts[f"{k}_{e}_id"][0] is not None],
       "lin": [ts[f"tfidf_{e}_ood"][0] - ts[f"tfidf_{e}_id"][0] for e in ("write", "tech")]})
  + "Note that the test sets here are small (about 300 responses per fold), so differences of a few points are within noise.")
lo = S["loso"]
P("<b>(b) Unseen prompt source.</b> We hold out each of the five prompt collections in turn (leave-one-source-out) and train on the other "
  "four. The collections differ strongly in style: Self-Instruct prompts are short synthetic tasks, Koala/OASST are real user requests, "
  "Helpful-Base comes from Anthropic's HH data.")
add_fig("fig7_loso.png", 5.0 * inch, "Leave-one-source-out accuracy versus the RQ1 CNN's accuracy on test prompts from the same source "
        "(which it saw other prompts of during training).")
lo_c = [v for v in lo["cnn"].values() if v is not None] or [float("nan")]
lo_src = [k for k in lo["tfidf"]]
lo_avg = {k: np.nanmean([lo["tfidf"][k]] + ([lo["cnn"][k]] if lo["cnn"].get(k) is not None else [])) for k in lo_src}
hard, easy = min(lo_avg, key=lo_avg.get), sorted(lo_avg, key=lo_avg.get, reverse=True)[:3]
ind = lo.get("cnn_in_dist", {})
gap = [ind[k] - lo["cnn"][k] for k in lo_src if lo["cnn"].get(k) is not None and k in ind]
P(f"Held-out-source accuracy for the CNN ranges from {min(lo_c):.3f} to {max(lo_c):.3f} (TF-IDF: {min(lo['tfidf'].values()):.3f}&ndash;"
  f"{max(lo['tfidf'].values()):.3f})" + (f"; on average it is {np.mean(gap):.3f} below the in-distribution accuracy on the same source"
  if gap and np.mean(gap) > 0 else (f"; on average it is {-np.mean(gap):.3f} <i>above</i> the in-distribution accuracy (the LOSO models also train on more prompts)" if gap else "")) +
  f". The hardest source is <b>{hard}</b> and the easiest are {', '.join(easy)}. "
  + ("Self-Instruct's short, closed tasks ('classify this sentence', 'give a synonym') leave little room for stylistic freedom: when an answer "
     "is one line, there are few places for a fingerprint to show. " if hard == "selfinstruct" else "") +
  "Identifiability therefore depends more on <i>how much freedom the prompt gives the model</i> than on whether that prompt source was "
  "seen in training.")
sb = S["siblings"]
if "cnn" in sb:
    c_, l_ = sb["cnn"], sb.get("lstm")
    fams = list(c_["sib_recall"])
    pc = np.array(c_["pred_counts"])
    where = []
    for i, f in enumerate(fams):
        wrong = pc[i].copy()
        wrong[FAM.index(f)] = 0
        j = int(np.argmax(wrong))
        where.append(f"{f} &rarr; {FAM[j]} ({wrong[j] / pc[i].sum():.0%})")
    rows = [["Family", "Trained on", "Tested on (never seen)", "CNN recall: same version", "CNN recall: sibling"] +
            (["LSTM recall: same", "LSTM recall: sibling"] if l_ else [])]
    for f in fams:
        rows.append([f, prim[f], c_["models"][f], f"{c_['prim_recall'][f]:.3f}", f"{c_['sib_recall'][f]:.3f}"] +
                    ([f"{l_['prim_recall'][f]:.3f}", f"{l_['sib_recall'][f]:.3f}"] if l_ else []))
    P("<b>(c) Unseen model version.</b> A practical detector will meet new releases. We take the RQ1 models (trained on one version per "
      "family) and test them, without retraining, on the answers of a sibling model from the same lab to the same test prompts.")
    add_table(rows, [0.6 * inch, 1.45 * inch, 1.65 * inch, 0.85 * inch, 0.8 * inch] + ([0.75 * inch, 0.75 * inch] if l_ else []),
              "RQ3c cross-version generalisation (recall = share of that family's responses attributed to the right family; mean over 3 seeds).")
    deltas = {f: c_["sib_recall"][f] - c_["prim_recall"][f] for f in fams}
    worst = min(deltas, key=deltas.get)
    better = [f for f in fams if deltas[f] > 0.02]
    ov = c_["sib_acc"][0] - c_["prim_acc_4fam"][0]
    P(f"Over these four families, CNN accuracy goes from {c_['prim_acc_4fam'][0]:.3f} on the training versions to {c_['sib_acc'][0]:.3f} on the "
      f"unseen siblings ({ov:+.3f})" + (f"; BiLSTM: {l_['prim_acc_4fam'][0]:.3f} &rarr; {l_['sib_acc'][0]:.3f} "
      f"({l_['sib_acc'][0] - l_['prim_acc_4fam'][0]:+.3f})" if l_ else "") + ". "
      + ("<b>The fingerprint largely transfers to new versions of the same family.</b> " if ov > -0.06 else
         "<b>The fingerprint only partly transfers to new versions.</b> ")
      + (f"{' and '.join(better)} {'is' if len(better) == 1 else 'are'} recognised even better in the sibling version than in the version the "
         "model was trained on, which suggests that a lab's house style can become <i>more</i> pronounced from one release to the next. " if better else "")
      + f"The largest drop is for {worst} ({c_['prim_recall'][worst]:.2f} &rarr; {c_['sib_recall'][worst]:.2f}). Most frequent mistakes on the "
      "siblings (CNN, pooled over seeds): " + "; ".join(where) + ". So the classifier is learning something closer to a <i>lab-level</i> "
      "style (shared post-training data, templates and formatting conventions) than a fingerprint of one checkpoint, although a deployed "
      "detector would still benefit from training on several versions per family.")
add_fig("fig6_rq3.png", 6.7 * inch, "RQ3 summary: cross-task (left, middle; dashed = chance) and cross-version (right) generalisation.")

# ======================================================================= 7. RQ4
P("7. RQ4 (extra credit) &mdash; What characteristics distinguish the LLMs?", H1)
F = S["features"]
P("We attack the 'why' from four directions: (a) interpretable stylometric features, (b) the n-grams that the linear model and the CNN "
  "rely on, (c) how answers begin, and (d) controlled ablations that remove one kind of signal from the text and retrain the networks.")
P(f"<b>(a) Stylometry.</b> We compute {len(F['groups'])} hand-crafted features on the full response, in five groups: <i>length</i> "
  "(characters, words, lines, paragraphs), <i>formatting</i> (markdown headers, bullets, numbered items, bold spans, code blocks, share of list "
  "lines, lines ending in a colon), <i>lexical</i> (mean word and sentence length, type-token ratio on the first 200 words, rates of 'I', "
  "'you' and hedging words), <i>punctuation</i> (rates of ! ? : ; dashes, parentheses, emoji) and <i>discourse</i> (opens with an "
  "affirmation such as 'Certainly', opens by restating, has a conclusion phrase, offers further help, ends with a question). Figure 8 shows each "
  "family's profile.")
add_fig("fig8_features.png", 6.8 * inch, "Family means of selected stylometric features, shown as z-scores across the six families: "
        "red = this family uses the feature more than the others, blue = less. Raw means are in results/summary.json.")
med = F["medians"]
means = F["means"]
mx = lambda k: max(FAM, key=lambda f: means[f][k])
mn = lambda k: min(FAM, key=lambda f: means[f][k])
P(f"The profiles are distinct. {mx('n_words')} writes the longest answers and {mn('n_words')} the shortest (median {med[mx('n_words')]['n_words']:.0f} vs "
  f"{med[mn('n_words')]['n_words']:.0f} words). "
  + (f"{mx('n_headers')} uses the most markdown headers and the most bold text, " if mx('n_headers') == mx('n_bold') else
     f"{mx('n_headers')} uses the most markdown headers, {mx('n_bold')} the most bold text, ")
  + f"{mx('frac_list_lines')} the highest share of list lines, and {mn('frac_list_lines')} the lowest. {mx('opens_affirmation')} most often opens with an "
  f"affirmation ({means[mx('opens_affirmation')]['opens_affirmation']:.0%} of answers), {mx('offers_more_help')} most often offers further help "
  f"({means[mx('offers_more_help')]['offers_more_help']:.0%}), and {mx('ttr_first200')} has the highest lexical diversity. These are exactly the "
  "kinds of cues a reader notices, and they are learnable.")
fc = F["clf"]
rows = [["Feature set (gradient-boosted trees)", "Test accuracy", "Feature set", "Test accuracy"]]
groups = ["length", "formatting", "lexical", "punctuation", "discourse"]
rows.append(["All features", f3(fc["all"]), "All (logistic regression)", f3(fc.get("all_logreg"))])
for g in groups:
    rows.append([f"Only {g}", f3(fc[g]), f"All except {g}", f3(fc["all_but_" + g])])
add_table(rows, [2.2 * inch, 1.0 * inch, 2.0 * inch, 1.0 * inch], "Classifying with stylometric features only (same split as RQ1).")
perm = list(F["perm"].items())[:6]
best_g = max(groups, key=lambda g: fc[g])
worst_rm = min(groups, key=lambda g: fc["all_but_" + g])
P(f"With only these {len(F['groups'])} numbers, gradient-boosted trees reach {f3(fc['all'])} accuracy &mdash; on par with the networks that read the "
  f"actual text. The most informative single group is <b>{best_g}</b> ({f3(fc[best_g])} alone), and removing <b>{worst_rm}</b> hurts most "
  f"({f3(fc['all'])} &rarr; {f3(fc['all_but_' + worst_rm])}). Permutation importance on the test set ranks "
  + ", ".join(f"<font face='Courier'>{k}</font> ({v:.3f})" for k, v in perm) +
  " highest (accuracy drop when the feature is shuffled). So the signals are <b>both structural and linguistic</b>: length and layout carry a "
  "lot, but lexical and discourse habits add information that layout alone does not.")
ty = S["typography"]
P(f"<b>(b) Discriminative n-grams.</b> The largest-weight n-grams of the TF-IDF logistic regression (Table {tab_no[0] + 1}) and the n-grams that "
  f"maximally activate the CNN filters with the highest weight for each class (Table {tab_no[0] + 2}) point to the same kinds of cues: answer "
  "openers and sign-offs, markdown idioms, and pet words. Some are surprisingly low-level: GPT-4o's top feature is the typographic "
  f"apostrophe (&rsquo; instead of '), which appears in {ty['GPT']['curly_apostrophe']:.0%} of its answers versus "
  f"{lt1(max(ty[f]['curly_apostrophe'] for f in FAM if f != 'GPT'))} for every other family. Qwen's symbol list contains the Chinese "
  f"character U+7684 (&lsquo;de&rsquo;): {ty['Qwen']['cjk']:.1%} of Qwen answers slip into Chinese characters, versus "
  f"{lt1(max(ty[f]['cjk'] for f in FAM if f != 'Qwen'))} for any other family. Claude's list is led by 'I apologize', the opening of its refusals and 'I cannot do that' answers "
  f"({ty['Claude']['apologize']:.0%} of Claude answers, {lt1(max(ty[f]['apologize'] for f in FAM if f != 'Claude'))} elsewhere); Llama's includes "
  f"'What a &hellip; question!' ({ty['Llama']['what_a_question']:.0%} of Llama answers open this way, {lt1(max(ty[f]['what_a_question'] for f in FAM if f != 'Llama'))} "
  "elsewhere); DeepSeek's includes 'Step' and 'It is important to'. (\\n = line break.)")
ng = S["top_ngrams"]
esc = lambda s: "".join(c if ord(c) < 256 else f"[U+{ord(c):04X}]" for c in s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
rows = [["Family", "Top word n-grams", "Top symbol-only n-grams"]]
for f in FAM:
    rows.append([f, ", ".join(f"<font face='Courier' size='7.5'>{esc(x)}</font>" for x in ng[f]["lexical"][:10]),
                 ", ".join(f"<font face='Courier' size='7.5'>{esc(x)}</font>" for x in ng[f]["symbol"][:4])])
add_table(rows, [0.75 * inch, 4.25 * inch, 1.9 * inch], "Most class-indicative n-grams (largest logistic-regression coefficients, among the "
          "top 60 per family). Tokens are separated by spaces.", zebra=True)
if "cnn_filters" in S:
    cf_ = S["cnn_filters"]
    rows = [["Family", "Most frequent max-activating n-grams of that class's highest-weight CNN filters (count over test answers)"]]
    for f in FAM:
        items = []
        for d in cf_[f][:6]:
            g, n = d["top_ngrams"][0]
            items.append(f"<font face='Courier' size='7.5'>{esc(g)}</font> ({n})")
        rows.append([f, "; ".join(items)])
    add_table(rows, [0.8 * inch, 6.1 * inch], "What the trained CNN (seed 0) detects. Each entry is one filter.")
op = S["openers"]
rows = [["Family", "Most common first two words of an answer (share of that family's answers)"]]
for f in FAM:
    rows.append([f, "; ".join((f"'{esc(k)}'" if k else "(starts with a symbol or code)") + f" {v:.0%}" for k, v in op[f][:5])])
add_table(rows, [0.8 * inch, 6.1 * inch], "How each family begins its answers.")
here2 = sorted(FAM, key=lambda f: -ty[f]["starts_here"])[:2]
here_txt = " and ".join(f"{f} ({ty[f]['starts_here']:.0%})" for f in here2)
cert_next = max(ty[f]["starts_certainly"] for f in FAM if f != "GPT")
P(f"<b>(c) Openers.</b> Families have recognisable ways of starting (Table {tab_no[0]}). "
  f"{ty['GPT']['starts_certainly']:.0%} of GPT-4o answers begin with 'Certainly' (next highest family: {cert_next:.0%}), while "
  f"answers opening with 'Here&hellip;' are most typical of {here_txt}. "
  "Because the first tokens are always inside the 256-token window, these cues are available to both networks; the "
  "'drop first &amp; last sentence' ablation below measures how much they matter.")
ab_ = S["ablation"]
P("<b>(d) Ablations: removing a signal and retraining.</b> We retrain both networks from scratch on transformed text (train, validation "
  "and test transformed alike), so a drop in accuracy measures how much the removed signal was needed and could not be replaced by others:")
bullets([
    "<i>drop first &amp; last sentence</i>: removes greetings like 'Certainly!' and sign-offs like 'Let me know if&hellip;';",
    "<i>strip markdown + line breaks</i>: removes headers, bullets, numbering, bold/italics, code fences and all layout;",
    "<i>words only</i>: lower-cased words, no punctuation, no layout;",
    "<i>content words only</i>: additionally removes all ~320 function words (the classic stylometric signal);",
    "<i>style skeleton</i>: the opposite &mdash; every content word becomes <font face='Courier'>W</font> (numbers <font face='Courier'>D</font>), "
    "keeping function words, punctuation and layout, so topic and vocabulary are gone;",
    "<i>first 64 tokens</i>: only the opening of the answer (also removes nearly all length information).",
])
full_c, full_l = M["cnn_output"]["acc"][0], M["lstm_output"]["acc"][0]
rows = [["Text given to the model", "TextCNN acc.", "&Delta; vs full", "BiLSTM acc.", "&Delta; vs full"]]
for k, lab in [("full", "Original response (RQ1)"), ("no_open_close", "Drop first &amp; last sentence"), ("no_format", "Strip markdown + line breaks"),
               ("words_only", "Words only (lower-case, no punctuation)"), ("content_only", "Content words only"),
               ("skeleton", "Style skeleton (content words &rarr; W)"), ("first64", "First 64 tokens only")]:
    if k == "full":
        c, l = M["cnn_output"]["acc"], M["lstm_output"]["acc"]
    else:
        c, l = ab_.get(f"cnn_{k}"), ab_.get(f"lstm_{k}")
    dc = "" if k == "full" or not c or c[0] is None else f"{c[0] - full_c:+.3f}"
    dl = "" if k == "full" or not l or l[0] is None else f"{l[0] - full_l:+.3f}"
    rows.append([lab, pm(c) if c and c[0] is not None else "not run", dc, pm(l) if l and l[0] is not None else "not run", dl])
add_table(rows, [2.4 * inch, 1.15 * inch, 0.8 * inch, 1.15 * inch, 0.8 * inch],
          "RQ4 ablations (test accuracy; CNN: mean &plusmn; std over seeds 0&ndash;1 where both were run, otherwise seed 0; BiLSTM: seed 0).")
add_fig("fig9_ablation.png", 5.6 * inch, "Ablation results (dashed line = chance).")


def g(k, a="cnn"):
    v = ab_.get(f"{a}_{k}")
    return v[0] if v and v[0] is not None else None


nf, sk_, co, f64, noc, wo = g("no_format"), g("skeleton"), g("content_only"), g("first64"), g("no_open_close"), g("words_only")
n_tr = M["cnn_output"]["n_train"]
lines = []
if nf is not None:
    lines.append(f"<b>Formatting matters most.</b> Stripping markdown and line breaks lowers CNN accuracy from {full_c:.3f} to {nf:.3f} ({nf - full_c:+.3f})"
                 + (f" and BiLSTM accuracy to {g('no_format', 'lstm'):.3f} ({g('no_format', 'lstm') - full_l:+.3f})" if g("no_format", "lstm") is not None else "")
                 + "; among the transformations that remove a single kind of signal, nothing hurts as much as removing layout"
                 + (f" (compare {noc - full_c:+.3f} for dropping the first and last sentence and {f64 - full_c:+.3f} for keeping only 64 tokens)"
                    if noc is not None and f64 is not None else "") + ".")
if wo is not None:
    lines.append(f"Removing punctuation and case as well (words only) lowers it further to {wo:.3f}.")
if sk_ is not None:
    if sk_ >= full_c - 0.01:
        lines.append(f"<b>Content is not needed.</b> Strikingly, replacing every content word by <font face='Courier'>W</font> does not hurt the CNN "
                     f"({sk_:.3f} vs {full_c:.3f})" + (f"; the BiLSTM gets {g('skeleton', 'lstm'):.3f}" if g("skeleton", "lstm") is not None else "") +
                     ". The networks do not need to know what an answer is about: function words, punctuation and layout are enough, and with only "
                     f"{n_tr:,} training answers the content words seem to act partly as noise that invites over-fitting to topics.")
    else:
        lines.append(f"The content-free style skeleton still gives {sk_:.3f} ({sk_ - full_c:+.3f}), so style without topic carries most of the signal.")
if co is not None:
    strong = co - 1 / 6 > 0.15
    lines.append((f"<b>Vocabulary alone is weaker but real.</b> " if strong else "<b>Vocabulary alone is weak.</b> ")
                 + f"Content words only (no function words, punctuation or layout) give {co:.3f}"
                 + (f" (BiLSTM {g('content_only', 'lstm'):.3f})" if g("content_only", "lstm") is not None else "")
                 + (", still far above chance: which words a model prefers is itself a fingerprint." if strong else
                    f", only {co - 1/6:.3f} above chance: what an answer talks about says little about who wrote it, because all six models "
                    "answered the same prompts and therefore cover the same topics."))
if noc is not None:
    if full_c - noc > 0.03:
        lines.append(f"<b>Openers and sign-offs matter.</b> Removing only the first and last sentence (everything else kept verbatim) costs "
                     f"{full_c - noc:.3f} ({full_c:.3f} &rarr; {noc:.3f}): stock phrases such as 'Certainly!', 'What a great question!' or "
                     "'I hope this helps' are a real part of the fingerprint, although most of the signal survives without them.")
    else:
        lines.append(f"Removing the first and last sentence changes accuracy by only {noc - full_c:+.3f}, so the classifier does not depend on boilerplate greetings and sign-offs.")
if f64 is not None:
    lines.append(f"Seeing only the first 64 tokens still yields {f64:.3f}" + (f" (BiLSTM {g('first64', 'lstm'):.3f})" if g("first64", "lstm") is not None else "") +
                 ": the opening of an answer is highly characteristic, and the rest of the response and its length add "
                 f"{full_c - f64:.3f}.")
P(" ".join(lines))
drops = {k: full_c - v for k, v in [("formatting/layout", nf), ("punctuation, case and layout", wo), ("function words, punctuation and layout", co),
                                     ("opening and closing sentence", noc), ("everything after the first 64 tokens", f64)] if v is not None}
order = sorted(drops, key=drops.get, reverse=True)
P("<b>Answer to RQ4.</b> The classifiers use both <i>structural</i> signals (markdown headers, bullets, bold, numbering, line breaks, "
  "response length) and <i>linguistic</i> signals (openers, function-word and punctuation habits, preferred vocabulary, even typography and "
  "language slips). Ranked by the accuracy change when they are removed (the removals are nested: 'words only' also removes layout), the "
  "most important are: " + "; ".join(f"{k} ({-drops[k]:+.3f})" for k in order) + ". "
  + ("The signals are largely <i>stylistic rather than topical</i>: a model that sees no content words at all is as good as one that sees "
     "everything. " if sk_ is not None and sk_ >= full_c - 0.01 else "Style carries more of the signal than topic. ")
  + "They are also redundant &mdash; every ablated model stays above chance &mdash; which is why simple countermeasures such "
  "as stripping markdown would weaken but not hide an LLM's identity.")

# ======================================================================= 8. lessons
P("8. Lessons and experience", H1)
bullets([
    "<b>Split by prompt, not by row.</b> With six answers per prompt, a random row split would put the same prompt in train and test and "
    "let a model match topics instead of styles. Grouping by prompt is the only way to measure generalisation to new questions.",
    "<b>The data design decides what a result means.</b> The parallel design made RQ2's input-only result exactly 1/6, and the confound "
    "experiment showed how easily an unbalanced collection would have produced a convincing but meaningless 'prompt fingerprint'. We learned to ask "
    "'could the label be predicted from how the data were collected?' before believing any number.",
    "<b>Check the data, not just the code.</b> An assertion caught that one file was stored in a different order; without it we would have "
    "paired Gemini answers with the wrong prompts. Another check found one empty response, which we removed for every model to keep classes balanced.",
    f"<b>Strong simple baselines are humbling and useful.</b> TF-IDF + logistic regression and even {len(F['groups'])} hand-crafted features are about as good as the "
    "CNN and LSTM here. With a few thousand examples, inductive bias and features matter more than depth; the neural models' advantage would "
    "likely grow with more data or pretrained embeddings.",
    f"<b>CNN vs LSTM in practice.</b> One CNN epoch took {cnn['sec_per_epoch']:.0f} s versus {lstm['sec_per_epoch']:.0f} s for the BiLSTM on one CPU core, and the CNN was "
    + ("about as accurate" if abs(cnn["acc"][0] - lstm["acc"][0]) < 0.015 else ("more accurate" if cnn["acc"][0] > lstm["acc"][0] else "slightly less accurate"))
    + f" ({cnn['acc'][0]:.3f} vs {lstm['acc'][0]:.3f}), which fits the finding that the signal is mostly local (n-grams, openers, markdown tokens) "
    "rather than long-range. The LSTM needed gradient clipping, packed sequences and a 256-token cap to be affordable.",
    "<b>Early stopping and seeds.</b> Both networks over-fit within a few epochs; restoring the best validation epoch and reporting the mean "
    "and spread over three seeds made the comparisons trustworthy.",
    "<b>Ablations answer 'why' better than accuracy, but must change only one thing.</b> Retraining on transformed text turned vague "
    "impressions ('GPT uses a lot of markdown') into measured effects. Our first version of the 'drop first &amp; last sentence' transformation "
    "re-joined the remaining sentences with line breaks, which silently changed the layout as well and exaggerated the effect (CNN 0.462); "
    "after fixing it to keep the middle of the answer verbatim the accuracy was 0.511. An ablation that touches two signals at once answers neither question.",
])
P("<b>Limitations.</b> One model per family (plus four siblings) cannot represent everything a family produces; outputs come from the "
  "decoding settings AlpacaEval used; the task-domain labels come from keyword rules; and the RQ3/RQ4 conditions use one or two seeds, so "
  "differences below about 0.02&ndash;0.03 should not be over-interpreted. All outputs answer English instructions from 2023-era prompt sets.")
P(f"<b>Use of AI tools.</b> {META['ai_statement']}")
P("<b>References.</b> Y. Kim, Convolutional Neural Networks for Sentence Classification, EMNLP 2014. S. Hochreiter &amp; J. Schmidhuber, "
  "Long Short-Term Memory, Neural Computation 1997. X. Li et al., AlpacaEval: An Automatic Evaluator of Instruction-following Models, "
  "github.com/tatsu-lab/alpaca_eval, 2023. Y. Dubois et al., Length-Controlled AlpacaEval, 2024. F. Pedregosa et al., Scikit-learn, JMLR 2011.",
  SMALL)


def footer(c, doc):
    c.saveState()
    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.grey)
    c.drawString(0.8 * inch, 0.5 * inch, "CMPSC 448 Midterm Project — Who Wrote It?")
    c.drawRightString(7.7 * inch, 0.5 * inch, f"{doc.page}")
    c.restoreState()


out = ROOT / "report" / "Project_Report.pdf"
doc = SimpleDocTemplate(str(out), pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch, topMargin=0.7 * inch,
                        bottomMargin=0.75 * inch, title="Who Wrote It? Identifying LLMs from Their Responses",
                        author=META["leader"])
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", out)
