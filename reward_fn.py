import re
from typing import Iterable, List, Optional, Sequence

try:
    from math_verify import parse, verify

    _HAS_MATH_VERIFY = True
except Exception:
    _HAS_MATH_VERIFY = False


_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text).strip().lower()


def _extract_final_answer(text: str) -> str:
    if not text:
        return ""

    tag_matches = _ANSWER_TAG_RE.findall(text)
    if tag_matches:
        return tag_matches[-1].strip()

    boxed_matches = _BOXED_RE.findall(text)
    if boxed_matches:
        return boxed_matches[-1].strip()

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1]


def _symbolic_match(pred: str, gt: str) -> bool:
    if not _HAS_MATH_VERIFY:
        return False
    try:
        return float(verify(parse(pred), parse(gt))) > 0
    except Exception:
        return False


def _math_correctness_reward(completion: str, gt: str) -> float:
    completion = completion or ""
    gt = "" if gt is None else str(gt)

    pred_answer = _extract_final_answer(completion)
    gt_answer = _extract_final_answer(gt)

    if _symbolic_match(pred_answer, gt_answer) or _symbolic_match(completion, gt):
        return 1.0
    if _normalize_text(pred_answer) == _normalize_text(gt_answer) and pred_answer:
        return 1.0
    return 0.0


def _max_consecutive_boxed_run(text: str) -> int:
    """Longest run of \\boxed{...} tokens that are only separated by whitespace."""
    if not text:
        return 0
    last_end: Optional[int] = None
    max_run = 0
    cur = 0
    for m in _BOXED_RE.finditer(text):
        if last_end is None:
            cur = 1
        else:
            between = text[last_end : m.start()]
            cur = cur + 1 if between.strip() == "" else 1
        max_run = max(max_run, cur)
        last_end = m.end()
    return max_run


def verifiable_math_reward(completions: Iterable[str], solution: Iterable[str], **kwargs) -> List[float]:
    rewards: List[float] = []
    for completion, gt in zip(completions, solution):
        rewards.append(_math_correctness_reward(completion or "", "" if gt is None else str(gt)))
    return rewards


def verifiable_math_reward_with_format_penalties(
    completions: Sequence[str],
    solution: Sequence[str],
    *,
    ended_with_eos: Optional[Sequence[bool]] = None,
    no_eos_penalty: float = 0.15,
    multi_boxed_penalty: float = 0.15,
    min_consecutive_boxed: int = 3,
) -> List[float]:
    """
    Correctness reward minus optional penalties:
    - If ``ended_with_eos[i]`` is False, subtract ``no_eos_penalty`` (natural stop vs length/truncation).
    - If the longest whitespace-separated run of ``\\boxed{...}`` is >= ``min_consecutive_boxed``,
      subtract ``multi_boxed_penalty`` (discourage repeated boxed spam).
    """
    rewards: List[float] = []
    comp_list = list(completions)
    sol_list = list(solution)
    n = max(len(comp_list), len(sol_list))
    eos_list = list(ended_with_eos) if ended_with_eos is not None else None
    for i in range(n):
        c = comp_list[i] if i < len(comp_list) else ""
        gt = sol_list[i] if i < len(sol_list) else ""
        base = _math_correctness_reward(c, gt)
        penalty = 0.0
        if eos_list is not None and i < len(eos_list) and not eos_list[i]:
            penalty += float(no_eos_penalty)
        if _max_consecutive_boxed_run(c) >= int(min_consecutive_boxed):
            penalty += float(multi_boxed_penalty)
        rewards.append(max(0.0, min(1.0, base - penalty)))
    return rewards
