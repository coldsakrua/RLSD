from __future__ import annotations

import importlib
import json
import os
from typing import Any

import torch

_PATCHED = False
_LAST_ROLLOUT_BATCH: Any = None


def _response_lengths_from_batch(batch: Any) -> list[int] | None:
    batch_data = getattr(batch, "batch", None)
    if batch_data is None:
        return None

    if "response_length" in batch_data:
        lengths = batch_data["response_length"]
        if isinstance(lengths, torch.Tensor):
            return [int(x) for x in lengths.reshape(-1).tolist()]
        return [int(x) for x in lengths]

    if "response_mask" in batch_data:
        mask = batch_data["response_mask"]
        if isinstance(mask, torch.Tensor):
            return [int(x) for x in mask.sum(dim=-1).tolist()]
        return [int(x) for x in mask]

    responses = batch_data.get("responses")
    attention_mask = batch_data.get("attention_mask")
    if responses is not None and attention_mask is not None:
        width = int(responses.shape[-1])
        if attention_mask.shape[-1] >= width:
            return [int(x) for x in attention_mask[:, -width:].sum(dim=-1).tolist()]
    return None


def _rollout_dump_enabled() -> bool:
    return os.environ.get("ENABLE_ROLLOUT_DUMP", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def patch_rollout_dump() -> None:
    global _PATCHED
    if _PATCHED or not _rollout_dump_enabled():
        return

    try:
        ray_trainer_mod = importlib.import_module("verl.trainer.ppo.ray_trainer")
    except Exception:
        return

    trainer_cls = getattr(ray_trainer_mod, "RayPPOTrainer", None)
    if trainer_cls is None:
        return

    orig_compute = ray_trainer_mod.compute_advantage
    if not getattr(orig_compute, "_rlsd_rollout_dump_stash", False):

        def compute_advantage_with_stash(*args: Any, **kwargs: Any) -> Any:
            global _LAST_ROLLOUT_BATCH
            data = kwargs.get("data")
            if data is None and args:
                data = args[0]
            _LAST_ROLLOUT_BATCH = data
            return orig_compute(*args, **kwargs)

        compute_advantage_with_stash._rlsd_rollout_dump_stash = True
        ray_trainer_mod.compute_advantage = compute_advantage_with_stash

    orig_dump = trainer_cls._dump_generations
    if getattr(orig_dump, "_rlsd_rollout_dump_patched", False):
        _PATCHED = True
        return

    def patched_dump_generations(
        self,
        inputs,
        outputs,
        scores,
        reward_extra_infos_dict,
        dump_path,
    ):
        max_samples = int(os.environ.get("ROLLOUT_DUMP_MAX_SAMPLES", "8") or "8")
        if max_samples > 0:
            inputs = inputs[:max_samples]
            outputs = outputs[:max_samples]
            scores = scores[:max_samples]
            reward_extra_infos_dict = {
                k: v[:max_samples]
                for k, v in reward_extra_infos_dict.items()
                if len(v) >= max_samples
            }

        response_lengths: list[int] | None = None
        if _LAST_ROLLOUT_BATCH is not None:
            all_lengths = _response_lengths_from_batch(_LAST_ROLLOUT_BATCH)
            if all_lengths is not None:
                response_lengths = all_lengths[: len(inputs)]

        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")
        n = len(inputs)
        base_data: dict[str, Any] = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }
        if response_lengths is not None and len(response_lengths) == n:
            base_data["response_length"] = response_lengths

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        msg = f"[step {self.global_steps}] dumped {n} rollout samples to {filename}"
        if response_lengths:
            avg_len = sum(response_lengths) / len(response_lengths)
            msg += f" (avg response_length={avg_len:.1f}, max={max(response_lengths)})"
        print(msg, flush=True)

    patched_dump_generations._rlsd_rollout_dump_patched = True
    trainer_cls._dump_generations = patched_dump_generations
    _PATCHED = True
