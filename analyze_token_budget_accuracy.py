#!/usr/bin/env python3
"""Compute Avg16 accuracy at token budgets from 32k eval JSON outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from eval_math_vllm_local import extract_boxed_answer, grade_answer

try:
    from transformers import AutoTokenizer
except ImportError as exc:
    raise SystemExit("transformers required") from exc

BUDGETS = [1024, 2048, 4096, 8192, 16384, 32768]
BUDGET_LABELS = ["1k", "2k", "4k", "8k", "16k", "32k"]
DEFAULT_MODEL = "/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b"

RUNS: List[Tuple[str, str, str, Optional[str]]] = [
    # flip wrong boost nogt (CAST)
    ("flip_wrong_boost_nogt", "math500", "outputs/eval_32k_math500/no_cot_32k/eval_20260525_195827_job1790208.json", None),
    ("flip_wrong_boost_nogt", "hmmt25", "outputs/eval_32k_hmmt25/no_cot_32k/eval_20260525_195827_job1790207.json", None),
    ("flip_wrong_boost_nogt", "aime25", "outputs/eval_32k_aime25/no_cot_32k/eval_20260525_195823_job1790209.json", None),
    ("flip_wrong_boost_nogt", "aime24", "outputs/eval_32k_aime24/no_cot_32k/eval_20260525_195826_job1790210.json", "aime24"),
    # rlsd (paper)
    ("rlsd_paper", "math500", "outputs/eval_32k_math500/no_cot_32k/eval_20260522_185107_job1764434.json", None),
    ("rlsd_paper", "hmmt25", "outputs/eval_32k_hmmt25/no_cot_32k/eval_20260522_185107_job1764433.json", None),
    ("rlsd_paper", "aime25", "outputs/eval_32k_aime25/no_cot_32k/eval_20260522_185037_job1764431.json", None),
    ("rlsd_paper", "aime24", "outputs/eval_32k_aime24/no_cot_32k/eval_20260522_185037_job1764430.json", "aime24"),
    # rlrt
    ("rlrt", "math500", "outputs/eval_32k_math500/no_cot_32k/eval_20260524_010653_job1770887.json", None),
    ("rlrt", "hmmt25", "outputs/eval_32k_hmmt25/no_cot_32k/eval_20260524_010653_job1770886.json", None),
    ("rlrt", "aime25", "outputs/eval_32k_aime25/no_cot_32k/eval_20260524_010653_job1770888.json", None),
    ("rlrt", "aime24", "outputs/eval_32k_aime24/no_cot_32k/eval_20260524_010653_job1770889.json", "aime24"),
    # grpo
    ("grpo", "math500", "outputs/eval_32k_math500/no_cot_32k/eval_20260514_181958_job1722373.json", None),
    ("grpo", "hmmt25", "outputs/eval_32k_hmmt25/no_cot_32k/eval_20260514_165601_job1722264.json", None),
    ("grpo", "aime25", "outputs/eval_32k_aime25/no_cot_32k/eval_20260514_182500_job1722388.json", "aime25"),
    ("grpo", "aime24", "outputs/eval_32k_aime24/no_cot_32k/eval_20260519_171022_job1752559.json", "aime24"),
    # opsd + grpo
    ("opsd+grpo", "math500", "outputs/eval_32k_math500/no_cot_32k/eval_20260515_015739_job1722956.json", None),
    ("opsd+grpo", "hmmt25", "outputs/eval_32k_hmmt25/no_cot_32k/eval_20260515_005512_job1722955.json", None),
    ("opsd+grpo", "aime25", "outputs/eval_32k_aime25/no_cot_32k/eval_20260515_005445_job1722954.json", "aime25"),
    ("opsd+grpo", "aime24", "outputs/eval_32k_aime24/no_cot_32k/eval_20260515_005445_job1722953.json", "aime24"),
    # 4b base (no LoRA)
    ("4b_base", "math500", "outputs/eval_32k_aime25/no_cot_32k/eval_20260514_124325_job1720517.json", "math500"),
    ("4b_base", "hmmt25", "outputs/eval_32k_hmmt25/no_cot_32k/eval_20260514_133351_job1721657.json", None),
    ("4b_base", "aime25", "outputs/eval_32k_aime25/no_cot_32k/eval_20260514_124325_job1720517.json", "aime25"),
    ("4b_base", "aime24", "outputs/eval_32k_aime24/no_cot_32k/eval_20260513_074900_job1711629.json", "aime24"),
]

LORA_MAP = {
    "flip_wrong_boost_nogt": "rlsd_4b_strict_split_flip_wrong_boost_nodecay_no_teacher_ref/checkpoint-300",
    "rlsd_paper": "rlsd_4b_paper/checkpoint-300",
    "rlrt": "rlrt_4b/checkpoint-300",
    "grpo": "grpo_4b_strict/checkpoint-300",
    "opsd+grpo": "opsd_4b_pure/checkpoint-300",
    "4b_base": "(no LoRA, base qwen3-4b)",
}


def analyze_json(
    path: Path,
    tokenizer,
    dataset_filter: Optional[str] = None,
) -> Dict:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("results", [])
    if dataset_filter:
        rows = [r for r in rows if r.get("dataset_tag") == dataset_filter]
    n_probs = len(rows)
    if n_probs == 0:
        return {"n": 0, "gen_n": 0, "budgets": {}, "path": str(path)}

    gen_n = rows[0].get("gen_n", len(rows[0].get("generations", [])))
    total_samples = n_probs * gen_n
    budget_correct = {b: 0 for b in BUDGETS}
    budget_pass1 = {b: 0 for b in BUDGETS}
    budget_pass16 = {b: 0 for b in BUDGETS}

    for r in rows:
        gt = r["ground_truth"]
        problem_any_correct = {b: False for b in BUDGETS}
        for gi, g in enumerate(r.get("generations", [])):
            full = g.get("full_generation", "")
            stored_ok = bool(g.get("correct", False))
            ids = tokenizer.encode(full, add_special_tokens=False)
            n_tok = len(ids)

            for b in BUDGETS:
                if b == 32768 and n_tok <= b:
                    ok = stored_ok
                else:
                    if n_tok <= b:
                        text = full
                    else:
                        text = tokenizer.decode(ids[:b], skip_special_tokens=True)
                    pred = extract_boxed_answer(text)
                    ok = grade_answer(pred, gt)
                if ok:
                    budget_correct[b] += 1
                    problem_any_correct[b] = True
                    if gi == 0:
                        budget_pass1[b] += 1
        for b in BUDGETS:
            if problem_any_correct[b]:
                budget_pass16[b] += 1

    out = {"n": n_probs, "gen_n": gen_n, "path": str(path), "budgets": {}}
    for b, lbl in zip(BUDGETS, BUDGET_LABELS):
        out["budgets"][lbl] = {
            "avg16_pct": 100.0 * budget_correct[b] / total_samples,
            "pass1_pct": 100.0 * budget_pass1[b] / n_probs,
            "pass16_pct": 100.0 * budget_pass16[b] / n_probs,
            "correct": budget_correct[b],
            "total": total_samples,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--csv", default="outputs/token_budget_accuracy_summary.csv")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    results = []
    for method, ds, rel, filt in RUNS:
        p = Path(rel)
        if not p.exists():
            print(f"MISSING: {p}", file=sys.stderr)
            continue
        print(f"Analyzing {method} / {ds} ...", flush=True)
        r = analyze_json(p, tok, filt)
        r["method"] = method
        r["dataset"] = ds
        results.append(r)

    methods_order = ["flip_wrong_boost_nogt", "rlsd_paper", "rlrt", "grpo", "opsd+grpo", "4b_base"]
    datasets_main = ["math500", "hmmt25", "aime25", "aime24"]

    print("\n" + "=" * 96)
    print("Token-budget Avg16 accuracy (truncate completion, re-extract last \\boxed{}, math_verify grade)")
    print("=" * 96)

    for ds in datasets_main:
        subset = [r for r in results if r["dataset"] == ds]
        if not subset:
            continue
        print(f"\n### {ds.upper()} (n={subset[0]['n']})")
        header = f"{'Method':<28}" + "".join(f"{lbl:>16}" for lbl in BUDGET_LABELS)
        print(header)
        print("-" * len(header))
        for m in methods_order:
            row = next((r for r in subset if r["method"] == m), None)
            if not row:
                continue
            line = f"{m:<28}"
            for lbl in BUDGET_LABELS:
                line += f"{row['budgets'][lbl]['avg16_pct']:6.2f}/{row['budgets'][lbl]['pass16_pct']:6.2f}"
            print(line)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "dataset", "budget", "avg16_pct", "pass16_pct", "pass1_pct", "n_problems", "gen_n", "json_path"])
        for r in results:
            for lbl in BUDGET_LABELS:
                b = r["budgets"][lbl]
                w.writerow([
                    r["method"], r["dataset"], lbl,
                    f"{b['avg16_pct']:.4f}", f"{b['pass16_pct']:.4f}", f"{b['pass1_pct']:.4f}",
                    r["n"], r["gen_n"], r["path"],
                ])
    print(f"\nSaved CSV -> {csv_path}")


if __name__ == "__main__":
    main()
