from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch


def _env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name, "")
    if not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def configured_max_seq_len() -> int | None:
    explicit = _env_int("RLSD_MAX_SEQ_LEN", 0)
    if explicit > 0:
        return explicit
    prompt = _env_int("RLSD_MAX_PROMPT_LENGTH", 0)
    response = _env_int("RLSD_MAX_RESPONSE_LENGTH", 0)
    if prompt > 0 and response > 0:
        return prompt + response
    return None


def configured_max_response_len() -> int | None:
    explicit = _env_int("RLSD_MAX_RESPONSE_LENGTH", 0)
    return explicit if explicit > 0 else None


def _slice_tensor_seq_dim(tensor: torch.Tensor, max_len: int) -> torch.Tensor:
    if tensor.dim() < 2 or tensor.shape[1] <= max_len:
        return tensor
    return tensor[:, -max_len:, ...]


def truncate_rollout_batch(batch: Any) -> int:
    """Cap rollout tensors to configured prompt+response budget.

    Prompts longer than the configured limit should already be filtered at dataset
    load time. Here we hard-cap generated responses and the full sequence width.
    """
    max_seq_len = configured_max_seq_len()
    max_response_len = configured_max_response_len()
    if max_seq_len is None or batch is None:
        return 0

    tensors = getattr(batch, "batch", None)
    if tensors is None:
        return 0

    input_ids = tensors.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.dim() < 2:
        return 0

    removed = 0
    seq_width = int(input_ids.shape[1])

    responses = tensors.get("responses")
    if isinstance(responses, torch.Tensor) and responses.dim() >= 2 and max_response_len:
        resp_width = int(responses.shape[1])
        if resp_width > max_response_len:
            tensors["responses"] = responses[:, :max_response_len]
            removed += resp_width - max_response_len

    if seq_width > max_seq_len:
        start = seq_width - max_seq_len
        for key in ("input_ids", "attention_mask", "position_ids"):
            value = tensors.get(key)
            if isinstance(value, torch.Tensor) and value.dim() >= 2 and value.shape[1] == seq_width:
                tensors[key] = value[:, start:]
        removed += seq_width - max_seq_len
        seq_width = max_seq_len

    responses = tensors.get("responses")
    if isinstance(responses, torch.Tensor) and responses.dim() >= 2:
        resp_width = int(responses.shape[1])
        if max_response_len and resp_width > max_response_len:
            tensors["responses"] = responses[:, :max_response_len]
            resp_width = max_response_len
        if isinstance(tensors.get("input_ids"), torch.Tensor):
            tensors["responses"] = tensors["input_ids"][:, -resp_width:]

    for key in ("rollout_log_probs", "old_log_probs", "old_log_prob", "ref_log_prob", "ref_log_probs"):
        value = tensors.get(key)
        if isinstance(value, torch.Tensor):
            tensors[key] = _slice_tensor_seq_dim(value, max_seq_len)

    for key in ("teacher_log_probs", "teacher_logprobs", "teacher_log_probs_all"):
        value = tensors.get(key)
        if isinstance(value, torch.Tensor):
            tensors[key] = _slice_tensor_seq_dim(value, max_seq_len)

    teacher_ids = tensors.get("teacher_ids")
    if isinstance(teacher_ids, torch.Tensor) and teacher_ids.dim() >= 3:
        tensors["teacher_ids"] = _slice_tensor_seq_dim(teacher_ids, max_seq_len)

    return removed


def truncate_data_proto(data: Any) -> int:
    return truncate_rollout_batch(data)


def _discard_truncated_enabled() -> bool:
    return os.environ.get("RLSD_DISCARD_TRUNCATED_COMPLETIONS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


_EOS_TOKEN_ID: int | None = None


def _eos_token_id() -> int | None:
    global _EOS_TOKEN_ID
    if _EOS_TOKEN_ID is not None:
        return _EOS_TOKEN_ID
    model_path = os.environ.get("MODEL_PATH") or os.environ.get("TOKENIZER_PATH")
    if not model_path:
        return None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        _EOS_TOKEN_ID = int(eos_id) if eos_id is not None else None
    except Exception:
        _EOS_TOKEN_ID = None
    return _EOS_TOKEN_ID


def _truncated_completion_mask(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    eos_token_id: int | None,
) -> torch.Tensor:
    """True when a completion should be discarded (length-truncated or over budget)."""
    batch_size = int(responses.shape[0])
    truncated = torch.zeros(batch_size, dtype=torch.bool, device=responses.device)
    max_response_len = configured_max_response_len()
    max_seq_len = configured_max_seq_len()

    for idx in range(batch_size):
        mask = response_mask[idx].bool()
        valid_len = int(mask.sum().item())
        if valid_len <= 0:
            truncated[idx] = True
            continue

        if max_response_len and valid_len >= max_response_len:
            truncated[idx] = True
            continue

        if eos_token_id is None:
            continue

        row = responses[idx][mask]
        if row.numel() == 0 or not torch.any(row == int(eos_token_id)):
            truncated[idx] = True

    return truncated


def discard_truncated_samples(data: Any) -> int:
    """Drop rollout rows that hit the response/sequence budget instead of truncating."""
    if data is None:
        return 0

    tensors = getattr(data, "batch", None)
    if tensors is None:
        return 0

    responses = tensors.get("responses")
    if not isinstance(responses, torch.Tensor) or responses.dim() < 2:
        return 0

    attention_mask = tensors.get("attention_mask")
    if not isinstance(attention_mask, torch.Tensor):
        return 0

    response_width = int(responses.shape[1])
    response_mask = attention_mask[:, -response_width:]
    max_seq_len = configured_max_seq_len()
    if max_seq_len and int(attention_mask.shape[1]) > max_seq_len:
        return int(attention_mask.shape[0])

    eos_token_id = _eos_token_id()
    truncated = _truncated_completion_mask(
        responses,
        response_mask,
        eos_token_id=eos_token_id,
    )
    keep = (~truncated).nonzero(as_tuple=False).reshape(-1).detach().cpu().tolist()
    if len(keep) == int(truncated.shape[0]):
        return 0
    if not keep:
        raise RuntimeError(
            "All rollout samples were truncated; reduce max_response_length or disable "
            "RLSD_DISCARD_TRUNCATED_COMPLETIONS."
        )

    kept = data.select_idxs(keep)
    data.batch = kept.batch
    data.non_tensor_batch = kept.non_tensor_batch
    return int(truncated.shape[0]) - len(keep)


def prepare_rollout_batch(data: Any) -> tuple[str, int]:
    if _discard_truncated_enabled():
        return "discard", discard_truncated_samples(data)
    return "truncate", truncate_data_proto(data)
