"""Dump the latest training rollout to JSON on checkpoint save and/or every N optimizer steps."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from transformers import TrainerCallback


class SaveRolloutSnapshotCallback(TrainerCallback):
    def __init__(self, trainer):
        self._trainer = trainer

    def _write_snapshot(self, args, state, *, trigger: str) -> None:
        if not getattr(args, "save_rollout_snapshots", True):
            return
        if hasattr(self._trainer, "accelerator") and not self._trainer.accelerator.is_main_process:
            return
        snap = getattr(self._trainer, "_last_rollout_snapshot", None)
        if not snap:
            return
        path = os.path.join(args.output_dir, f"rollout_snapshot_step_{state.global_step:06d}.json")
        payload = {
            "checkpoint_global_step": int(state.global_step),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot_trigger": trigger,
            **snap,
        }
        os.makedirs(args.output_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[rollout_snapshot] wrote {path} (trigger={trigger})", flush=True)

    def on_save(self, args, state, control, **kwargs):
        self._write_snapshot(args, state, trigger="checkpoint_save")

    def on_step_end(self, args, state, control, **kwargs):
        interval = int(getattr(args, "rollout_snapshot_interval_steps", 0) or 0)
        if interval <= 0:
            return
        step = int(getattr(state, "global_step", 0) or 0)
        if step <= 0:
            return
        if step % interval != 0:
            return
        self._write_snapshot(args, state, trigger=f"every_{interval}_steps")
