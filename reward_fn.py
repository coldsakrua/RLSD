import re
from typing import Iterable, List

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


def verifiable_math_reward(completions: Iterable[str], solution: Iterable[str], **kwargs) -> List[float]:
    rewards: List[float] = []
    for completion, gt in zip(completions, solution):
        completion = completion or ""
        gt = "" if gt is None else str(gt)

        pred_answer = _extract_final_answer(completion)
        gt_answer = _extract_final_answer(gt)

        reward = 0.0
        if _symbolic_match(pred_answer, gt_answer) or _symbolic_match(completion, gt):
            reward = 1.0
        else:
            if _normalize_text(pred_answer) == _normalize_text(gt_answer) and pred_answer:
                reward = 1.0

        rewards.append(reward)
    return rewards
