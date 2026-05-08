import argparse
import os
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset


SYSTEM_PROMPT = (
    "You are a helpful math reasoning assistant. "
    "Think step by step, and put the final answer in \\boxed{}."
)


def _build_prompt(problem: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


def _to_str(x: Any) -> str:
    if x is None:
        return ""
    return x if isinstance(x, str) else str(x)


def convert_opsd_to_verl_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        problem = _to_str(r.get("problem", "")).strip()
        solution = _to_str(r.get("solution", "")).strip()
        if not problem or not solution:
            continue

        subject = _to_str(r.get("subject", "")).strip().lower() or "math"
        problem_type = _to_str(r.get("type", "")).strip()
        level = _to_str(r.get("level", "")).strip()

        rows.append(
            {
                "data_source": "opsd_math",
                "prompt": _build_prompt(problem),
                "ability": subject,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution,
                },
                "extra_info": {
                    "subject": subject,
                    "type": problem_type,
                    "level": level,
                },
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_parquet",
        type=str,
        default="data/aggregated_l3plus/train.parquet",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/verl_opsd",
    )
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_parquet(args.input_parquet)
    if args.max_samples is not None and args.max_samples > 0:
        df = df.head(args.max_samples)

    rows = convert_opsd_to_verl_rows(df)
    ds = Dataset.from_list(rows)

    if args.val_ratio > 0:
        split = ds.train_test_split(test_size=args.val_ratio, seed=args.seed, shuffle=True)
        train_ds = split["train"]
        val_ds = split["test"]
    else:
        train_ds = ds
        val_ds = ds.select(range(min(256, len(ds))))

    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "val.parquet")
    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)

    print(f"[verl_data] train={len(train_ds)} -> {train_path}")
    print(f"[verl_data] val={len(val_ds)} -> {val_path}")


if __name__ == "__main__":
    main()
