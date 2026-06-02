from __future__ import annotations

import importlib.util
import re
import os
from typing import Any, Mapping


def _load_extract_solution():
    try:
        from verl_rlsd.prompt_utils import extract_solution

        return extract_solution
    except ImportError:
        pass
    # veRL loads this file as a standalone module, so relative imports fail.
    prompt_utils_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_utils.py")
    spec = importlib.util.spec_from_file_location("verl_rlsd_prompt_utils", prompt_utils_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_solution


extract_solution = _load_extract_solution()

try:
    from reward_fn import (
        configure_math_reward_extraction,
        verifiable_math_reward,
        verifiable_math_reward_with_format_penalties,
    )
except Exception:  # pragma: no cover - server import path handles the normal case.
    configure_math_reward_extraction = None
    verifiable_math_reward = None
    verifiable_math_reward_with_format_penalties = None


_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_REWARD_EXTRACTION_CONFIGURED = False


def _last_boxed(text: str) -> str:
    matches = _BOXED_RE.findall(text or "")
    if matches:
        return matches[-1].strip()
    return (text or "").strip()


def _fallback_score(completion: str, ground_truth: str) -> float:
    pred = _last_boxed(completion)
    gt = _last_boxed(ground_truth)
    return 1.0 if pred and gt and pred == gt else 0.0


def _maybe_configure_reward_extraction() -> None:
    global _REWARD_EXTRACTION_CONFIGURED
    if _REWARD_EXTRACTION_CONFIGURED or configure_math_reward_extraction is None:
        return
    frac = float(os.environ.get("REWARD_BOXED_LAST_TOKEN_FRACTION", "0.05") or 0.0)
    if frac <= 0:
        _REWARD_EXTRACTION_CONFIGURED = True
        return
    model_path = os.environ.get("TOKENIZER_PATH") or os.environ.get("MODEL_PATH")
    if not model_path:
        return
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        configure_math_reward_extraction(
            tokenizer=tokenizer,
            boxed_last_token_fraction=frac,
        )
        _REWARD_EXTRACTION_CONFIGURED = True
    except Exception:
        # Leave reward_fn on its default extraction path if tokenizer loading is
        # unavailable in a worker process.
        return


def _pick_completion(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> str:
    for key in ("solution_str", "completion", "response", "output", "text"):
        if key in kwargs and kwargs[key] is not None:
            return str(kwargs[key])
    if len(args) >= 2 and args[1] is not None:
        return str(args[1])
    if len(args) >= 1 and isinstance(args[0], Mapping):
        row = args[0]
        for key in ("solution_str", "completion", "response", "output", "text"):
            if key in row and row[key] is not None:
                return str(row[key])
    return ""


def _pick_ground_truth(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> str:
    for key in ("ground_truth", "solution", "answer", "target", "reference"):
        if key in kwargs and kwargs[key] is not None:
            return str(kwargs[key])
    extra = kwargs.get("extra_info")
    if isinstance(extra, Mapping):
        gt = extract_solution(extra)
        if gt:
            return gt
    if len(args) >= 3 and args[2] is not None:
        return str(args[2])
    if len(args) >= 1 and isinstance(args[0], Mapping):
        gt = extract_solution(args[0])
        if gt:
            return gt
    return ""


def compute_score(*args: Any, **kwargs: Any) -> float:
    _maybe_configure_reward_extraction()
    completion = _pick_completion(args, kwargs)
    ground_truth = _pick_ground_truth(args, kwargs)
    if not ground_truth:
        return 0.0

    use_format_penalties = str(
        kwargs.get("reward_format_penalties", os.environ.get("REWARD_FORMAT_PENALTIES", "false"))
    ).lower() in {"1", "true", "yes", "y"}
    if use_format_penalties and verifiable_math_reward_with_format_penalties is not None:
        return float(
            verifiable_math_reward_with_format_penalties(
                [completion],
                [ground_truth],
                ended_with_eos=kwargs.get("ended_with_eos"),
                no_eos_penalty=float(kwargs.get("no_eos_penalty", os.environ.get("REWARD_NO_EOS_PENALTY", 0.15))),
                multi_boxed_penalty=float(kwargs.get("multi_boxed_penalty", os.environ.get("REWARD_MULTI_BOXED_PENALTY", 0.15))),
                min_consecutive_boxed=int(kwargs.get("min_consecutive_boxed", os.environ.get("REWARD_MIN_CONSECUTIVE_BOXED", 2))),
                repeat_triplet_penalty=float(kwargs.get("repeat_triplet_penalty", os.environ.get("REWARD_REPEAT_TRIPLET_PENALTY", 0.0))),
                repeat_triplet_levenshtein_threshold=int(
                    kwargs.get("repeat_triplet_levenshtein_threshold", os.environ.get("REWARD_REPEAT_TRIPLET_LEV_THRESHOLD", 0))
                ),
            )[0]
        )

    if verifiable_math_reward is not None:
        return float(verifiable_math_reward([completion], [ground_truth])[0])
    return _fallback_score(completion, ground_truth)


def compute_score_zero(*args: Any, **kwargs: Any) -> float:
    return 0.0
