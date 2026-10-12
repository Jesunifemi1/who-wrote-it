"""Build the (LLM_name, LLM_input, LLM_output) dataset from AlpacaEval public model outputs.

Source: tatsu-lab/alpaca_eval (Apache-2.0), results/<model>/model_outputs.json,
commit cd543a149df89434d8a54582c0151c0b945c3d20. Every model answered the SAME 805
instructions, so the dataset is perfectly parallel and balanced.

Outputs
  data/processed/dataset.csv   primary set: one flagship model per family (6 x 805 rows)
  data/processed/siblings.csv  held-out sibling versions (4 families x 805) for RQ3
  data/processed/splits.json   prompt-level train/val/test split (no prompt crosses splits)
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

PRIMARY = {
    "GPT": "gpt-4o-2024-05-13",
    "Claude": "claude-3-opus-20240229",
    "Gemini": "gemini-pro",
    "Llama": "Meta-Llama-3-70B-Instruct",
    "Qwen": "Qwen1.5-72B-Chat",
    "DeepSeek": "deepseek-llm-67b-chat",
}
SIBLINGS = {
    "GPT": "gpt-4-turbo-2024-04-09",
    "Claude": "claude-3-5-sonnet-20240620",
    "Llama": "Meta-Llama-3.1-70B-Instruct-Turbo",
    "Qwen": "Qwen2-72B-Instruct",
}
FAMILIES = list(PRIMARY)

# ---------------------------------------------------------------- task categories (for RQ3)
CODE_RE = re.compile(
    r"\b(python|javascript|typescript|java|c\+\+|c#|golang|rust|sql|html|css|bash|shell script|"
    r"regex|regular expression|code|coding|program|programming|function|algorithm|compile|"
    r"debug|api|json|latex|git|linux command|python script|leetcode|array|for loop|data structure)\b",
    re.I,
)
MATH_RE = re.compile(
    r"\b(calculate|compute|solve|equation|math|mathematical|arithmetic|integral|derivative|"
    r"probability|percent|percentage|prime number|algebra|geometry|theorem|sum of|square root|"
    r"average of|multiply|divide|fraction|solve for)\b|\d\s*[\+\-\*/\^=]\s*\d",
    re.I,
)
WRITE_VERB_RE = re.compile(
    r"^\W*(please\s+)?(can you\s+)?(help me\s+)?(write|compose|draft|rewrite|paraphrase|summari[sz]e|"
    r"proofread|edit|polish|rephrase|translate|brainstorm|craft|create a (poem|story|slogan|tagline))\b",
    re.I,
)
WRITE_RE = re.compile(
    r"\b(write|compose|draft|create|generate|come up with|make up|rewrite|paraphrase|edit|"
    r"brainstorm)\b[^.?!]{0,80}\b(poem|story|stories|essay|letter|e-?mail|song|lyrics|haiku|"
    r"speech|blog|article|script|tweet|post|caption|slogan|review|description|bio|limerick|"
    r"joke|dialogue|conversation|rap|sonnet|ad|advertisement|headline|title|names?|plot|"
    r"paragraph|summary|outline|toast|introduction|message|cover letter|product description|"
    r"sentence|text)\b",
    re.I,
)


def task_category(instr: str) -> str:
    head = instr[:400]
    if CODE_RE.search(head):
        return "coding"
    if MATH_RE.search(head):
        return "math"
    if WRITE_VERB_RE.search(head) or WRITE_RE.search(head):
        return "writing"
    return "qa"  # factual / advice / open questions


REF_ORDER = {r["instruction"]: i for i, r in enumerate(json.load(open(RAW / f"{PRIMARY['GPT']}.json")))}


def load_model(name: str, family: str) -> pd.DataFrame:
    rows = json.load(open(RAW / f"{name}.json"))
    df = pd.DataFrame(rows)[["dataset", "instruction", "output"]]
    # files are not all stored in the same order (gemini-pro differs) -> align by instruction text
    df["prompt_id"] = df["instruction"].map(REF_ORDER)
    assert df.prompt_id.notna().all() and df.prompt_id.is_unique, name
    df = df.sort_values("prompt_id").reset_index(drop=True)
    df["model"] = name
    df["family"] = family
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prim = pd.concat([load_model(m, f) for f, m in PRIMARY.items()], ignore_index=True)
    sib = pd.concat([load_model(m, f) for f, m in SIBLINGS.items()], ignore_index=True)

    # sanity: all models answered the same instructions in the same order
    ref = prim[prim.family == "GPT"].instruction.tolist()
    for m, g in pd.concat([prim, sib]).groupby("model"):
        assert g.instruction.tolist() == ref, m

    cats = {i: task_category(t) for i, t in enumerate(ref)}
    for d in (prim, sib):
        d["task"] = d.prompt_id.map(cats)
        d["output"] = d["output"].fillna("").astype(str)
        d.rename(columns={"dataset": "source", "instruction": "input"}, inplace=True)

    # drop prompts where ANY primary model returned an empty answer (keeps the set balanced)
    empty = set(prim.loc[prim.output.str.strip() == "", "prompt_id"])
    prim = prim[~prim.prompt_id.isin(empty)].reset_index(drop=True)
    sib = sib[~sib.prompt_id.isin(empty)].reset_index(drop=True)

    # prompt-level split 70/15/15 (fixed seed) -> no prompt appears in two splits
    pids = np.array(sorted(prim.prompt_id.unique()))
    rng = np.random.default_rng(448)
    rng.shuffle(pids)
    n = len(pids)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    splits = {
        "train": sorted(map(int, pids[:n_tr])),
        "val": sorted(map(int, pids[n_tr : n_tr + n_va])),
        "test": sorted(map(int, pids[n_tr + n_va :])),
    }
    cols = ["prompt_id", "family", "model", "source", "task", "input", "output"]
    prim[cols].to_csv(OUT / "dataset.csv", index=False)
    sib[cols].to_csv(OUT / "siblings.csv", index=False)
    json.dump(splits, open(OUT / "splits.json", "w"))

    print("dropped prompts with empty outputs:", sorted(empty))
    print("rows:", len(prim), "| per family:", prim.family.value_counts().to_dict())
    print("split sizes (prompts):", {k: len(v) for k, v in splits.items()})
    print("task categories (prompts):", pd.Series(cats).loc[pids].value_counts().to_dict())
    print("source datasets (prompts):", prim[prim.family == "GPT"].source.value_counts().to_dict())


if __name__ == "__main__":
    main()
