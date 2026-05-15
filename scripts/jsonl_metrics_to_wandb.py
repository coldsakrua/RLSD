#!/usr/bin/env python3
"""Replay train_metrics.jsonl to Weights & Biases (online or offline run folder)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


def _is_loggable_scalar(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float)):
        return True
    return False


def _wandb_key(key: str, *, train_prefix: str) -> str:
    """Match HuggingFace Trainer + W&B: training scalars appear under train/* in the UI."""
    if not train_prefix:
        return key
    if key.startswith(f"{train_prefix}/") or key.startswith("eval/"):
        return key
    return f"{train_prefix}/{key}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, help="Path to train_metrics.jsonl")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--run_name", required=True, help="Run display name")
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity; defaults from wandb login (online / sync only)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Write W&B run under wandb_root (WANDB_MODE=offline); sync later with wandb sync",
    )
    parser.add_argument(
        "--wandb_root",
        default=None,
        help="Parent dir for ./wandb/ (default: cwd). Use a stable path so you can wandb sync it later.",
    )
    parser.add_argument(
        "--train-prefix",
        default="train",
        metavar="PREFIX",
        help="Prefix keys so metrics show in W&B 'train' panel (default: train). Use '' to disable.",
    )
    parser.add_argument(
        "--no-train-prefix",
        action="store_true",
        help="Log keys as in JSONL (charts-only layout); same as --train-prefix=''.",
    )
    args = parser.parse_args()
    train_prefix = "" if args.no_train_prefix else (args.train_prefix or "").strip()

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
    elif "WANDB_MODE" in os.environ and os.environ["WANDB_MODE"].lower() == "offline":
        print(
            "WANDB_MODE=offline is set but --offline was not passed; "
            "pass --offline to write local offline runs, or unset WANDB_MODE for online upload.",
            file=sys.stderr,
        )

    jsonl_path = os.path.abspath(args.jsonl)
    wandb_root = os.path.abspath(args.wandb_root) if args.wandb_root else os.getcwd()
    os.makedirs(wandb_root, exist_ok=True)
    wandb_data = os.path.join(wandb_root, ".wandb_data")
    os.makedirs(wandb_data, exist_ok=True)
    os.environ.setdefault("WANDB_DATA_DIR", wandb_data)

    import wandb

    kwargs = dict(
        project=args.project,
        name=args.run_name,
        entity=args.entity,
        dir=wandb_root,
        config={"source_jsonl": jsonl_path},
    )
    wandb.init(**kwargs)
    try:
        with open(args.jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                step = rec.get("step")
                if step is None:
                    continue
                payload = {
                    _wandb_key(k, train_prefix=train_prefix): v
                    for k, v in rec.items()
                    if k not in ("timestamp_utc", "step") and _is_loggable_scalar(v)
                }
                if payload:
                    wandb.log(payload, step=int(step))
    finally:
        wandb.finish()

    mode = "offline" if args.offline else "online"
    print(f"Finished ({mode}) metrics replay from:", jsonl_path)
    print("W&B files under:", os.path.join(wandb_root, "wandb"))


if __name__ == "__main__":
    main()
