import json
from typing import Any

from reward_fn import verifiable_math_reward


def _to_ground_truth(ground_truth: Any) -> str:
    if isinstance(ground_truth, dict):
        if "ground_truth" in ground_truth:
            return str(ground_truth["ground_truth"])
        if "answer" in ground_truth:
            return str(ground_truth["answer"])
        return str(ground_truth)

    if isinstance(ground_truth, str):
        text = ground_truth.strip()
        if not text:
            return ""
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    if "ground_truth" in parsed:
                        return str(parsed["ground_truth"])
                    if "answer" in parsed:
                        return str(parsed["answer"])
            except Exception:
                pass
        return text

    return "" if ground_truth is None else str(ground_truth)


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    verl custom reward function interface (rule-based reward).
    Returns scalar reward in [0, 1].
    """
    pred = "" if solution_str is None else str(solution_str)
    gt = _to_ground_truth(ground_truth)
    score = float(verifiable_math_reward([pred], [gt])[0])
    return {
        "score": score,
        "is_correct": score,
    }
