#!/usr/bin/env python3
"""
Rewrite DAPO-style math parquet prompts to: (stripped problem stem) + standard ``\\boxed{}`` instruction.

Reads ``dapo-math-17k.parquet``-like files (columns include ``prompt``, typically ``reward_model``).
Preserves all other columns; only ``prompt`` is updated in-place on a shallow-copied row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, load_dataset

from data_utils import DEFAULT_MATH_INSTRUCTION_SUFFIX, normalize_prompt_to_standard_instruction


def _preview_prompt(prompt, max_chars: int = 400) -> str:
    if isinstance(prompt, list):
        try:
            s = json.dumps(prompt, ensure_ascii=False)
        except TypeError:
            s = str(prompt)
    else:
        s = str(prompt)
    s = s.replace("\n", "\\n")
    if len(s) > max_chars:
        return s[:max_chars] + "…"
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess DAPO math parquet: normalize user prompt + boxed suffix.")
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/dapo/dapo-math-17k.parquet",
        help="Source parquet (DAPO schema: prompt + reward_model, etc.).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/dapo/dapo-math-17k-standard-boxed.parquet",
        help="Output parquet path (parent dirs are created).",
    )
    parser.add_argument(
        "--math_instruction_suffix",
        type=str,
        default=DEFAULT_MATH_INSTRUCTION_SUFFIX,
        help="Suffix appended after stripped stem (default matches data_utils.DEFAULT_MATH_INSTRUCTION_SUFFIX).",
    )
    parser.add_argument("--num_proc", type=int, default=1, help="datasets.map num_proc (set >1 for large CPU pools).")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print first-row before/after preview and exit without writing.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input parquet not found: {input_path.resolve()}")

    ds: Dataset = load_dataset("parquet", data_files={"train": str(input_path)}, split="train")
    if "prompt" not in ds.column_names:
        raise ValueError(f"Expected column 'prompt', got: {ds.column_names}")

    def _rewrite(row: dict) -> dict:
        return {
            "prompt": normalize_prompt_to_standard_instruction(
                row.get("prompt"),
                suffix=args.math_instruction_suffix,
            ),
        }

    if args.dry_run:
        one = ds[0]
        before = one.get("prompt")
        after = normalize_prompt_to_standard_instruction(before, suffix=args.math_instruction_suffix)
        print("[dry_run] first row prompt BEFORE:", _preview_prompt(before))
        print("[dry_run] first row prompt AFTER: ", _preview_prompt(after))
        print("[dry_run] num_rows:", len(ds))
        return

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ds_out = ds.map(
        _rewrite,
        desc="normalize_prompt_to_standard_instruction",
        num_proc=max(1, int(args.num_proc)),
    )
    ds_out.to_parquet(str(out))
    print(f"[ok] wrote {out.resolve()} ({len(ds_out)} rows)")


if __name__ == "__main__":
    main()
