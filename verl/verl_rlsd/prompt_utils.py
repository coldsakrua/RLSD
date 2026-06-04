from __future__ import annotations

from typing import Any, Mapping


DEFAULT_TRANSITION_PROMPT = (
    "\n\nAfter reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step -- do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and do not be afraid to backtrack "
    "or reconsider if something does not work out:\n"
)

DEFAULT_OFFICIAL_TEACHER_PROMPT = (
    "Problem: {prompt}\n\n"
    "Here is a reference solution to this problem:\n"
    "=== Reference Solution Begin ===\n"
    "{solution}\n"
    "=== Reference Solution End ===\n"
    "{transition}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)

DEFAULT_REF_TEACHER_PROMPT = (
    "{prompt}\n\n[Reference solution]\n{solution}\n\n[Student response]\n"
)
DEFAULT_NO_REF_TEACHER_PROMPT = "{prompt}\n\n[Student response]\n"
DEFAULT_ROLLOUT_TEACHER_PROMPT = (
    "{prompt}\n\n[Correct rollout]\n{correct_rollout}\n\n[Student response]\n"
)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, Mapping):
                if part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                elif "text" in part:
                    parts.append(str(part.get("text", "")))
                elif "content" in part:
                    parts.append(str(part.get("content", "")))
            elif part is not None:
                parts.append(str(part))
        return "\n".join(x.strip() for x in parts if str(x).strip()).strip()
    return str(content).strip()


def extract_prompt_text(prompt: Any) -> str:
    if isinstance(prompt, list):
        last_user = None
        for msg in prompt:
            if not isinstance(msg, Mapping):
                continue
            role = str(msg.get("role", "")).lower()
            if role == "user":
                last_user = msg
            elif role == "" and "content" in msg and last_user is None:
                last_user = msg
        if last_user is not None:
            return _content_to_text(last_user.get("content", ""))
        return _content_to_text(prompt)
    if isinstance(prompt, Mapping):
        if "content" in prompt:
            return _content_to_text(prompt.get("content", ""))
        return str(prompt).strip()
    return "" if prompt is None else str(prompt).strip()


def extract_solution(row: Mapping[str, Any]) -> str:
    for key in ("solution", "ground_truth", "answer", "target", "reference"):
        if key in row and row[key] is not None:
            text = str(row[key]).strip()
            if text:
                return text

    reward_model = row.get("reward_model")
    if isinstance(reward_model, Mapping):
        for key in ("ground_truth", "answer", "solution", "target"):
            if key in reward_model and reward_model[key] is not None:
                text = str(reward_model[key]).strip()
                if text:
                    return text
    if reward_model is not None:
        ground_truth = getattr(reward_model, "ground_truth", None)
        if ground_truth is not None:
            return str(ground_truth).strip()
    return ""


def build_teacher_prompt(
    *,
    prompt: Any,
    solution: Any = "",
    mode: str = "reference_solution",
    correct_rollout: str = "",
    teacher_prompt_template: str | None = None,
    teacher_prompt_template_no_reference: str | None = None,
    teacher_prompt_template_with_rollout: str | None = None,
    teacher_transition_prompt: str | None = None,
) -> str:
    prompt_text = extract_prompt_text(prompt)
    solution_text = "" if solution is None else str(solution)
    mode = (mode or "reference_solution").strip().lower()

    if mode in {
        "official_opsd",
        "official",
        "reference_solution",
        "student_reference_solution",
        "student_with_reference_solution",
        "with_gt",
        "with_ground_truth",
    }:
        template = teacher_prompt_template or DEFAULT_REF_TEACHER_PROMPT
        transition = teacher_transition_prompt or DEFAULT_TRANSITION_PROMPT
        try:
            return template.format(
                prompt=prompt_text,
                solution=solution_text,
                transition=transition,
            )
        except KeyError:
            return template.format(prompt=prompt_text, solution=solution_text)

    if mode in {"no_reference", "none", "student_prompt"}:
        template = teacher_prompt_template_no_reference or DEFAULT_NO_REF_TEACHER_PROMPT
        return template.format(prompt=prompt_text)

    if mode in {"identical_student", "same_as_student", "student_identical"}:
        # Teacher prompt ids are taken directly from the student rollout prompt.
        return prompt_text

    if mode in {"successful_rollout", "rollout"}:
        template = teacher_prompt_template_with_rollout or DEFAULT_ROLLOUT_TEACHER_PROMPT
        return template.format(prompt=prompt_text, correct_rollout=correct_rollout)

    raise ValueError(f"Unsupported teacher prompt mode: {mode}")
