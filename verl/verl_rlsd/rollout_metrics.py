from __future__ import annotations

import importlib
import logging
import math
from typing import Any

import torch

logger = logging.getLogger(__name__)

_PATCHED = False
_METRIC_KEY = "rollout/avg_response_length"


def _batch_tensor(batch: Any, *keys: str) -> torch.Tensor | None:
    batch_data = getattr(batch, "batch", None)
    if batch_data is None:
        return None
    for key in keys:
        if key not in batch_data:
            continue
        value = batch_data[key]
        if isinstance(value, torch.Tensor):
            return value
        try:
            return torch.as_tensor(value)
        except Exception:
            continue
    return None


def _response_lengths(batch: Any) -> torch.Tensor | None:
    precomputed = _batch_tensor(batch, "response_length")
    if precomputed is not None:
        return precomputed.float().reshape(-1)

    response_mask = _batch_tensor(batch, "response_mask")
    if response_mask is not None:
        return response_mask.sum(dim=-1).float()

    attention_mask = _batch_tensor(batch, "attention_mask")
    responses = _batch_tensor(batch, "responses")
    if attention_mask is not None and responses is not None:
        width = int(responses.shape[-1])
        if attention_mask.shape[-1] >= width:
            return attention_mask[:, -width:].sum(dim=-1).float()
    return None


def _avg_rollout_response_length(batch: Any) -> float | None:
    lengths = _response_lengths(batch)
    if lengths is None or lengths.numel() == 0:
        return None
    avg = float(lengths.mean().item())
    if math.isnan(avg):
        return None
    return avg


def _avg_from_metrics(metrics: dict[str, Any]) -> float | None:
    for key in (_METRIC_KEY, "response_length/mean", "response_length_non_aborted/mean"):
        value = metrics.get(key)
        if value is None:
            continue
        try:
            avg = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(avg):
            return avg
    return None


def _emit_rollout_avg_length(step: int, avg_len: float) -> None:
    message = f"[step {step}] rollout avg response length: {avg_len:.2f}"
    print(message, flush=True)
    logger.info(message)


def _patch_compute_data_metrics() -> None:
    try:
        metric_utils = importlib.import_module("verl.trainer.ppo.metric_utils")
    except Exception:
        return
    if getattr(metric_utils.compute_data_metrics, "_rlsd_rollout_metrics_patched", False):
        return
    original = metric_utils.compute_data_metrics

    def patched_compute_data_metrics(batch: Any, use_critic: bool = True) -> dict[str, Any]:
        metrics = original(batch, use_critic=use_critic)
        avg_len = _avg_from_metrics(metrics)
        if avg_len is None:
            avg_len = _avg_rollout_response_length(batch)
        if avg_len is not None:
            metrics[_METRIC_KEY] = avg_len
        return metrics

    patched_compute_data_metrics._rlsd_rollout_metrics_patched = True
    metric_utils.compute_data_metrics = patched_compute_data_metrics

    try:
        ray_trainer = importlib.import_module("verl.trainer.ppo.ray_trainer")
        if hasattr(ray_trainer, "compute_data_metrics"):
            ray_trainer.compute_data_metrics = patched_compute_data_metrics
    except Exception:
        pass


def _patch_tracking_log() -> None:
    try:
        tracking_mod = importlib.import_module("verl.utils.tracking")
    except Exception:
        return
    tracking_cls = getattr(tracking_mod, "Tracking", None)
    if tracking_cls is None or getattr(tracking_cls.log, "_rlsd_rollout_metrics_patched", False):
        return
    original_log = tracking_cls.log

    def patched_log(self, data: dict[str, Any], step: int, *args: Any, **kwargs: Any) -> None:
        avg_len = _avg_from_metrics(data)
        if avg_len is not None:
            _emit_rollout_avg_length(step, avg_len)
        return original_log(self, data, step, *args, **kwargs)

    patched_log._rlsd_rollout_metrics_patched = True
    tracking_cls.log = patched_log


def patch_rollout_length_logging() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _patch_compute_data_metrics()
    _patch_tracking_log()
    _PATCHED = True
