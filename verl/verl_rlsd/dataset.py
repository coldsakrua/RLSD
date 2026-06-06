from __future__ import annotations

import copy
import os
from typing import Any, Mapping

from verl.utils.dataset.rl_dataset import RLHFDataset

from data_utils import DEFAULT_MATH_INSTRUCTION_SUFFIX, _strip_math_prompt_boilerplate

DEFAULT_MATH_PROMPT_PREFIX = ""
DEFAULT_MATH_PROMPT_SUFFIX = DEFAULT_MATH_INSTRUCTION_SUFFIX


def _strip_enabled() -> bool:
    return os.environ.get("STRIP_DAPO_PROMPT_BOILERPLATE", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _math_prompt_prefix() -> str:
    return os.environ.get("MATH_PROMPT_PREFIX", DEFAULT_MATH_PROMPT_PREFIX).strip()


def _math_prompt_suffix() -> str:
    return os.environ.get("MATH_PROMPT_SUFFIX", DEFAULT_MATH_PROMPT_SUFFIX).strip()


def _has_boxed_instruction(text: str) -> bool:
    low = text.lower()
    return "final answer" in low and "boxed" in low


def _format_math_instruction_text(
    text: str,
    *,
    add_prefix: bool = True,
    add_suffix: bool = True,
) -> str:
    text = text.strip()
    prefix = _math_prompt_prefix() if add_prefix else ""
    if prefix and prefix.lower() not in text.lower():
        text = f"{prefix}\n\n{text}"
    suffix = _math_prompt_suffix() if add_suffix else ""
    if suffix and not _has_boxed_instruction(text):
        text = f"{text}\n\n{suffix}"
    return text


def _normalize_math_content(content: Any) -> Any:
    strip_enabled = _strip_enabled()
    if isinstance(content, str):
        if strip_enabled:
            content = _strip_math_prompt_boilerplate(content)
        return _format_math_instruction_text(content)
    if isinstance(content, list):
        out: list[Any] = []
        text_indices: list[int] = []
        for part in content:
            if isinstance(part, Mapping) and part.get("type") == "text":
                item = dict(part)
                text = str(item.get("text", ""))
                if strip_enabled:
                    text = _strip_math_prompt_boilerplate(text)
                item["text"] = text
                text_indices.append(len(out))
                out.append(item)
            else:
                out.append(part)
        if text_indices:
            first_idx = text_indices[0]
            last_idx = text_indices[-1]
            if first_idx == last_idx:
                item = dict(out[first_idx])
                item["text"] = _format_math_instruction_text(str(item.get("text", "")))
                out[first_idx] = item
            else:
                first_item = dict(out[first_idx])
                first_item["text"] = _format_math_instruction_text(
                    str(first_item.get("text", "")),
                    add_suffix=False,
                )
                out[first_idx] = first_item
                last_item = dict(out[last_idx])
                last_item["text"] = _format_math_instruction_text(
                    str(last_item.get("text", "")),
                    add_prefix=False,
                )
                out[last_idx] = last_item
        return out
    return content


def normalize_user_prompt_messages(messages: list[Any]) -> list[Any]:
    """Strip DAPO boilerplate and append the boxed-answer math instruction."""
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
        normalized["content"] = _normalize_math_content(msg.get("content", ""))
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

    def _read_files_and_tokenize(self):
        import datasets

        dataframes = []
        for parquet_file in self.data_files:
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer

            def doc2len(doc) -> int:
                row = dict(doc)
                messages = self._build_messages(row)
                return len(
                    tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        **self.apply_chat_template_kwargs,
                    )
                )

            self.dataframe = self.dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )
            print(f"filter dataset len: {len(self.dataframe)}")
