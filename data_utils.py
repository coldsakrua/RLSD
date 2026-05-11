import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

from datasets import Dataset, load_dataset

# Standard user instruction (single LaTeX backslash before "boxed").
DEFAULT_MATH_INSTRUCTION_SUFFIX = (
    "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
)

# DAPO / OpenR1-style lead-in; ``math`` and comma before ``step`` are optional.
_SOLVE_FOLLOWING_PROBLEM_STEP = (
    r"Solve\s+the\s+following(?:\s+math)?\s+problem\s*,?\s+step\s+by\s+step\s*[.:]?\s*"
)
_LEADING_STEP_BY_STEP = re.compile(
    rf"^\s*(?:{_SOLVE_FOLLOWING_PROBLEM_STEP})",
    re.IGNORECASE | re.DOTALL,
)
# DAPO-style “Answer: $Answer … answer to the problem.” preamble (often after the Solve… sentence).
_LEADING_LAST_LINE_ANSWER_BLOCK = re.compile(
    r"^\s*The last line of your response should be.+?answer\s+to\s+the\s+problem\.\s*",
    re.IGNORECASE | re.DOTALL,
)
_TRAILING_REASON_BOXED = re.compile(
    r"\s*Please\s+reason\s+step\s+by\s+step.*$",
    re.IGNORECASE | re.DOTALL,
)


_PROMPT_KEYS = ("prompt", "problem", "question", "query", "input", "instruction")
_SOLUTION_KEYS = ("solution", "answer", "ground_truth", "target", "reference")
_AGGREGATED_L3PLUS_KEYS = ("problem", "solution", "level", "type", "subject")


def _looks_like_dapo(columns: Iterable[str]) -> bool:
    """DAPO-style parquet: chat `prompt` + `reward_model.ground_truth` (no top-level solution)."""
    col_set = set(columns)
    return "prompt" in col_set and "reward_model" in col_set


def _ground_truth_from_reward_model(reward_model) -> str:
    if reward_model is None:
        return ""
    if isinstance(reward_model, dict):
        gt = reward_model.get("ground_truth")
        return str(gt).strip() if gt is not None else ""
    # Arrow/pyarrow struct may surface as simple namespace in some versions
    gt = getattr(reward_model, "ground_truth", None)
    if gt is not None:
        return str(gt).strip()
    return ""


def _coerce_prompt_for_rlsd(prompt) -> Union[str, list]:
    """Keep chat-style message lists for tokenizer.apply_chat_template; stringify only scalars."""
    if isinstance(prompt, str):
        return prompt.strip()
    if isinstance(prompt, list):
        return prompt
    return str(prompt).strip()


def coerce_prompt_to_qwen3_user_messages(prompt: Any) -> list:
    """
    Match Qwen3 HF examples: rollout prompts are a ``conversation`` list so TRL GRPO calls
    ``apply_chat_template`` (with ``add_generation_prompt=True``) instead of raw ``encode`` on plain text.

    - ``str`` / non-list scalars -> ``[{"role": "user", "content": ...}]``
    - existing chat ``list[dict]`` -> returned unchanged (shallow copy of list only if we mutate elsewhere)
    """
    if isinstance(prompt, list):
        return prompt
    if isinstance(prompt, dict):
        return [prompt]
    text = prompt.strip() if isinstance(prompt, str) else str(prompt).strip()
    return [{"role": "user", "content": text}]


def _content_to_text(content: Any) -> str:
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                elif "text" in part:
                    parts.append(str(part.get("text", "")))
                elif "content" in part:
                    parts.append(str(part.get("content", "")))
            elif part is not None:
                parts.append(str(part))
        return "\n".join(x.strip() for x in parts if str(x).strip()).strip()
    if content is None:
        return ""
    return str(content).strip()


def extract_last_user_text(prompt: Any) -> str:
    """
    Best-effort extraction of the last user turn text without chat template tokens.
    """
    if isinstance(prompt, list):
        last_user = None
        for msg in prompt:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).lower()
            if role == "user":
                last_user = msg
            elif role == "" and "content" in msg and last_user is None:
                # Some datasets store single-turn dicts without explicit role.
                last_user = msg
        if last_user is not None:
            return _content_to_text(last_user.get("content", ""))
        return _content_to_text(prompt)
    if isinstance(prompt, dict):
        if "content" in prompt:
            return _content_to_text(prompt.get("content", ""))
        return str(prompt).strip()
    if isinstance(prompt, str):
        return prompt.strip()
    if prompt is None:
        return ""
    return str(prompt).strip()


def apply_qwen3_rollout_chat_template(
    tokenizer,
    prompt: Any,
    *,
    enable_thinking: bool = False,
    add_generation_prompt: bool = True,
    tokenize: bool = False,
) -> str:
    """
    Same call shape as Qwen3 docs: ``apply_chat_template(messages, tokenize=..., add_generation_prompt=...,
    enable_thinking=...)``. ``enable_thinking=False`` is the non-thinking rollout path.
    """
    messages = coerce_prompt_to_qwen3_user_messages(prompt)
    kwargs: dict = {"tokenize": tokenize, "add_generation_prompt": add_generation_prompt}
    kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _coerce_solution_scalar(solution) -> str:
    if isinstance(solution, str):
        return solution.strip()
    if solution is None:
        return ""
    return str(solution).strip()


def _pick_key(candidates: Iterable[str], columns: Iterable[str]) -> Optional[str]:
    col_set = set(columns)
    for key in candidates:
        if key in col_set:
            return key
    return None


def _resolve_data_file(dataset_path: str, split: str) -> Path:
    path = Path(dataset_path)
    if path.is_file():
        return path

    candidates = [
        path / f"{split}.jsonl",
        path / f"{split}.json",
        path / f"{split}.parquet",
        path / "data.jsonl",
        path / "data.json",
        path / "data.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot find dataset file under {dataset_path}. "
        f"Tried: {', '.join(str(p) for p in candidates)}"
    )


def _load_single_file(path: Path, split: str) -> Dataset:
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".json"):
        return load_dataset("json", data_files={split: str(path)}, split=split)
    if suffix == ".parquet":
        return load_dataset("parquet", data_files={split: str(path)}, split=split)
    raise ValueError(f"Unsupported dataset format: {path}")


def _looks_like_aggregated_l3plus(columns: Iterable[str]) -> bool:
    col_set = set(columns)
    return all(k in col_set for k in _AGGREGATED_L3PLUS_KEYS)


def _non_empty_text(x) -> bool:
    if x is None:
        return False
    return bool(str(x).strip())


def _strip_math_prompt_boilerplate(text: str) -> str:
    """Remove common DAPO-style wrappers so the stem is mostly the math statement."""
    t = text.strip()
    m = re.search(
        r"<\|im_start\|>\s*user\s*(.*?)(?:<\|im_end\|>|<\|im_start\|>\s*assistant|$)",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        t = m.group(1).strip()
    t = _LEADING_STEP_BY_STEP.sub("", t)
    t = _LEADING_LAST_LINE_ANSWER_BLOCK.sub("", t)
    t = _TRAILING_REASON_BOXED.sub("", t).strip()
    return t


def _already_has_standard_suffix(text: str) -> bool:
    low = text.lower()
    return "please reason step by step" in low and "boxed" in low


def normalize_prompt_to_standard_instruction(
    prompt: Any,
    *,
    suffix: str = DEFAULT_MATH_INSTRUCTION_SUFFIX,
) -> Any:
    """
    Collapse dataset-specific instructions to: (core problem text) + fixed suffix with ``\\boxed{}``.

    - Chat-style ``list[dict]``: shallow-copies messages and rewrites the last user turn.
    - Plain ``str``: strips known boilerplate then appends ``suffix`` (once).
    """
    suf = suffix or DEFAULT_MATH_INSTRUCTION_SUFFIX

    if isinstance(prompt, list):
        out: List[Any] = []
        last_user_idx = None
        for msg in prompt:
            out.append(dict(msg) if isinstance(msg, dict) else msg)
            if isinstance(out[-1], dict) and str(out[-1].get("role", "")).lower() == "user":
                last_user_idx = len(out) - 1
        # Some parquet rows store a single ``{"content": "..."}`` turn without ``role``.
        if last_user_idx is None and len(out) == 1 and isinstance(out[0], dict) and "content" in out[0]:
            last_user_idx = 0
        if last_user_idx is None:
            return prompt
        user_msg = dict(out[last_user_idx])
        content = user_msg.get("content", "")
        if isinstance(content, list):
            new_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    p = _strip_math_prompt_boilerplate(str(part.get("text", "")))
                    if not _already_has_standard_suffix(p):
                        p = f"{p}{suf}"
                    new_parts.append({**part, "text": p})
                else:
                    new_parts.append(part)
            user_msg["content"] = new_parts
        else:
            p = _strip_math_prompt_boilerplate(str(content))
            if not _already_has_standard_suffix(p):
                p = f"{p}{suf}"
            user_msg["content"] = p
        out[last_user_idx] = user_msg
        return out

    if isinstance(prompt, str):
        p = _strip_math_prompt_boilerplate(prompt)
        if not _already_has_standard_suffix(p):
            p = f"{p}{suf}"
        return p
    return prompt


def load_rlsd_dataset(dataset_path: str, split: str = "train") -> Dataset:
    data_file = _resolve_data_file(dataset_path, split)
    ds = _load_single_file(data_file, split=split)

    if _looks_like_aggregated_l3plus(ds.column_names):
        def _normalize_l3plus(row):
            row["prompt"] = str(row["problem"]).strip()
            row["solution"] = str(row["solution"]).strip()
            row["problem_level"] = str(row.get("level", "")).strip()
            row["problem_type"] = str(row.get("type", "")).strip()
            row["problem_subject"] = str(row.get("subject", "")).strip()
            return row

        ds = ds.map(_normalize_l3plus, desc="Normalizing aggregated_l3plus schema")
        ds = ds.filter(
            lambda row: _non_empty_text(row["prompt"]) and _non_empty_text(row["solution"]),
            desc="Filtering empty prompt/solution rows",
        )
        return ds

    if _looks_like_dapo(ds.column_names):

        def _normalize_dapo(row):
            row["solution"] = _ground_truth_from_reward_model(row.get("reward_model"))
            # Always strip DAPO/OpenR1-style wrappers here. This makes prompt cleaning
            # robust even if upper-level script flags are omitted.
            row["prompt"] = normalize_prompt_to_standard_instruction(
                _coerce_prompt_for_rlsd(row.get("prompt")),
            )
            return row

        ds = ds.map(_normalize_dapo, desc="Normalizing DAPO schema (prompt + reward_model.ground_truth)")
        return ds

    prompt_key = _pick_key(_PROMPT_KEYS, ds.column_names)
    solution_key = _pick_key(_SOLUTION_KEYS, ds.column_names)
    if prompt_key is None or solution_key is None:
        raise ValueError(
            f"Failed to infer prompt/solution columns from {ds.column_names}. "
            f"Expected prompt in {_PROMPT_KEYS}, solution in {_SOLUTION_KEYS}."
        )

    def _normalize(row):
        row["prompt"] = _coerce_prompt_for_rlsd(row[prompt_key])
        row["solution"] = _coerce_solution_scalar(row[solution_key])
        return row

    ds = ds.map(_normalize, desc="Normalizing prompt and solution columns")
    return ds
