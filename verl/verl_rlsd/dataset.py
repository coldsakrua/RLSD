from __future__ import annotations

import copy
import os
from typing import Any, Mapping

from verl.utils.dataset.rl_dataset import RLHFDataset

from data_utils import _strip_math_prompt_boilerplate

DEFAULT_MATH_PROMPT_PREFIX = ""


def _strip_enabled() -> bool:
    return os.environ.get("STRIP_DAPO_PROMPT_BOILERPLATE", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _strip_text_content(content: Any) -> Any:
    if isinstance(content, str):
        return _strip_math_prompt_boilerplate(content)
    if isinstance(content, list):
        out: list[Any] = []
        for part in content:
            if isinstance(part, Mapping) and part.get("type") == "text":
                item = dict(part)
                item["text"] = _strip_math_prompt_boilerplate(str(item.get("text", "")))
                out.append(item)
            else:
                out.append(part)
        return out
    return content


def _math_prompt_prefix() -> str:
    return os.environ.get("MATH_PROMPT_PREFIX", DEFAULT_MATH_PROMPT_PREFIX).strip()


def _prepend_math_instruction(content: Any) -> Any:
    prefix = _math_prompt_prefix()
    if not prefix:
        return content
    if isinstance(content, str):
        text = content.strip()
        if prefix.lower() in text.lower():
            return text
        return f"{prefix}\n\n{text}"
    if isinstance(content, list):
        out: list[Any] = []
        prefixed = False
        for part in content:
            if (
                not prefixed
                and isinstance(part, Mapping)
                and part.get("type") == "text"
            ):
                item = dict(part)
                item["text"] = _prepend_math_instruction(str(item.get("text", "")))
                out.append(item)
                prefixed = True
            else:
                out.append(part)
        return out
    return content


def normalize_user_prompt_messages(messages: list[Any]) -> list[Any]:
    """Strip DAPO boilerplate and prepend a short math instruction prefix."""
    out: list[Any] = []
    for msg in messages:
        if not isinstance(msg, Mapping):
            out.append(msg)
            continue
        role = str(msg.get("role", "")).lower()
        if role != "user":
            out.append(msg)
            continue
        normalized = copy.deepcopy(msg)
        content = msg.get("content", "")
        if _strip_enabled():
            content = _strip_text_content(content)
        content = _prepend_math_instruction(content)
        normalized["content"] = content
        out.append(normalized)
    return out


def strip_dapo_prompt_boilerplate(messages: list[Any]) -> list[Any]:
    """Remove baked-in DAPO math instruction wrappers from user turns."""
    return normalize_user_prompt_messages(messages)


class RLSDRLHFDataset(RLHFDataset):
    """RLHFDataset that strips DAPO-style instruction boilerplate from user prompts."""

    def _build_messages(self, example: dict):
        messages = super()._build_messages(example)
        return strip_dapo_prompt_boilerplate(messages)
