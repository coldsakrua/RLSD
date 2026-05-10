"""Dump the latest training rollout to JSON whenever a checkpoint is saved."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from transformers import TrainerCallback


class SaveRolloutSnapshotCallback(TrainerCallback):
    def __init__(self, trainer):
        self._trainer = trainer

    def on_save(self, args, state, control, **kwargs):
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
            **snap,
        }
        os.makedirs(args.output_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[rollout_snapshot] wrote {path}", flush=True)
