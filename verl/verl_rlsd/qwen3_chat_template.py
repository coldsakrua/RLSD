from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

# Qwen3 non-thinking chat template injects an empty block after the assistant header.
_THINK_TAG = "think"
_THINK_OPEN = f"<{_THINK_TAG}>"
_THINK_CLOSE = f"</{_THINK_TAG}>"

_EMPTY_THINKING_SUFFIXES = (
    f"<|im_start|>assistant\n{_THINK_OPEN}\n\n{_THINK_CLOSE}\n\n",
    f"<|im_start|>assistant\n{_THINK_OPEN}{_THINK_CLOSE}\n\n",
    f"assistant\n{_THINK_OPEN}\n\n{_THINK_CLOSE}\n\n",
    f"assistant\n{_THINK_OPEN}{_THINK_CLOSE}\n\n",
)

_JINJA_EMPTY_THINKING_BLOCK = re.compile(
    r"\{%- if enable_thinking is defined and enable_thinking is false %\}.*?\{%- endif %\}\s*",
    re.DOTALL,
)


def strip_empty_thinking_enabled() -> bool:
    return os.environ.get("STRIP_EMPTY_THINKING_GENERATION_PROMPT", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


_EMPTY_THINKING_BLOCK_RE = re.compile(
    r"(<\|im_start\|>assistant\n|assistant\n)"
    r"\s*"
    + re.escape(_THINK_OPEN)
    + r"\s*"
    + re.escape(_THINK_CLOSE)
    + r"\s*",
    re.MULTILINE,
)


def strip_empty_thinking_generation_prompt(text: str) -> str:
    """Remove Qwen3's empty thinking placeholder from a generation prompt string."""
    if not isinstance(text, str) or not text:
        return text
    out = text
    for suffix in _EMPTY_THINKING_SUFFIXES:
        if suffix.startswith("<|im_start|>"):
            out = out.replace(suffix, "<|im_start|>assistant\n")
        else:
            out = out.replace(suffix, "assistant\n")
    out = _EMPTY_THINKING_BLOCK_RE.sub(
        lambda m: m.group(1),
        out,
    )
    return out


def qwen3_chat_template_without_empty_thinking_block(base_template: str) -> str:
    # Qwen3 uses the empty <think></think> block as the official no-thinking
    # generation marker when enable_thinking=False. Keep it in the template.
    return base_template


def load_qwen3_chat_template_without_empty_thinking(model_path: str) -> str:
    cfg_path = Path(model_path) / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    template = str(cfg.get("chat_template") or "")
    if not template:
        raise ValueError(f"chat_template missing in {cfg_path}")
    return qwen3_chat_template_without_empty_thinking_block(template)


def install_qwen3_no_think_chat_template(tokenizer: Any) -> None:
    """Restore the official Qwen3 template; enable_thinking=False selects no-think."""
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return
    model_path = getattr(tokenizer, "name_or_path", None)
    if not model_path:
        return
    try:
        tokenizer.chat_template = load_qwen3_chat_template_without_empty_thinking(str(model_path))
    except Exception:
        pass
