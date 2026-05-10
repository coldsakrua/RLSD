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


def _longest_consecutive_boxed_run(text: str) -> tuple[int, int]:
    """
    Return (start_idx, run_len) over boxed matches where neighbors are separated only by whitespace.
    start_idx is the index in the boxed-match list (not character index).
    """
    matches = list(_BOXED_RE.finditer(text or ""))
    if not matches:
        return -1, 0

    best_start, best_len = 0, 1
    cur_start, cur_len = 0, 1
    for i in range(1, len(matches)):
        between = text[matches[i - 1].end() : matches[i].start()]
        if between.strip() == "":
            cur_len += 1
        else:
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
            cur_start, cur_len = i, 1
    if cur_len > best_len:
        best_start, best_len = cur_start, cur_len
    return best_start, best_len


def _extract_final_answer(text: str) -> str:
    if not text:
        return ""

    tag_matches = _ANSWER_TAG_RE.findall(text)
    if tag_matches:
        return tag_matches[-1].strip()

    boxed_match_objs = list(_BOXED_RE.finditer(text))
    if boxed_match_objs:
        run_start, run_len = _longest_consecutive_boxed_run(text)
        # If there is a consecutive boxed run (e.g. boxed spam), score by the FIRST boxed answer.
        if run_start >= 0 and run_len >= 2:
            return boxed_match_objs[run_start].group(1).strip()
        # Otherwise keep original "final boxed answer" behavior.
        return boxed_match_objs[-1].group(1).strip()

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
    return _longest_consecutive_boxed_run(text)[1]


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
    min_consecutive_boxed: int = 2,
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
        run_len = _max_consecutive_boxed_run(c)
        if run_len >= int(min_consecutive_boxed):
            # Only the first boxed answer is credited; each additional consecutive box is penalized.
            extra_boxes = max(0, run_len - 1)
            penalty += float(multi_boxed_penalty) * float(extra_boxes)
        rewards.append(max(0.0, min(1.0, base - penalty)))
    return rewards
