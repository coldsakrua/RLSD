import re
from typing import Any, Iterable, List, Optional, Sequence, Tuple

# Filled by ``configure_math_reward_extraction`` from training script (tokenizer + tail fraction).
_MATH_REWARD_EXTRACT_CFG: dict[str, Any] = {"tokenizer": None, "boxed_last_token_fraction": 0.0}


def configure_math_reward_extraction(
    *,
    tokenizer=None,
    boxed_last_token_fraction: float = 0.0,
) -> None:
    """
    Configure answer extraction for reward scoring.

    When ``boxed_last_token_fraction > 0``, only ``\\boxed{...}`` / ``<answer>`` spans whose **start character index**
    lies in the last that fraction of **completion tokens** (via ``offset_mapping`` when ``tokenizer`` is set).
    If ``tokenizer`` is ``None``, non-whitespace runs (``re.finditer(r"\\S+", text)``) are used as a coarse token proxy.
    Ground-truth strings should use ``for_ground_truth=True`` in ``_extract_final_answer`` (no tail gate).
    """
    _MATH_REWARD_EXTRACT_CFG["tokenizer"] = tokenizer
    _MATH_REWARD_EXTRACT_CFG["boxed_last_token_fraction"] = max(0.0, float(boxed_last_token_fraction))

try:
    from math_verify import parse, verify

    _HAS_MATH_VERIFY = True
except Exception:
    _HAS_MATH_VERIFY = False


# Legacy shallow boxed (no nested braces). Prefer `_find_boxed_balanced` for scoring.
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)

_BOXED_BEGIN = "\\boxed{"

_THINK_STRIP_PATTERNS = (
    r"<think>.*?</think>",
    r"<reasoning>.*?</reasoning>",
)

# Openers whose closing tag may be missing when generation hits max_tokens (strip tail after opener).
_THINK_OPEN_CLOSE = (("<think>", "</think>"),)


def _strip_thinking_blocks(text: str) -> str:
    """Remove thinking / reasoning blocks before parsing answers (handles truncated unclosed blocks)."""
    t = text or ""
    for pat in _THINK_STRIP_PATTERNS:
        t = re.sub(pat, "", t, flags=re.DOTALL | re.IGNORECASE)
    tl = t.lower()
    for open_tag, close_tag in _THINK_OPEN_CLOSE:
        o = open_tag.lower()
        c = close_tag.lower()
        start = tl.find(o)
        if start < 0:
            continue
        rest_from = start + len(open_tag)
        cidx = tl.find(c, rest_from)
        if cidx < 0:
            # No closing tag (truncated mid-thinking): drop everything from opener onward.
            t = t[:start].strip()
        else:
            end = cidx + len(close_tag)
            t = (t[:start] + t[end:]).strip()
        tl = t.lower()
    return t


def _find_boxed_balanced(text: str) -> List[Tuple[int, int, str]]:
    """
    All ``\\boxed{...}`` spans with **balanced** braces (supports ``\\boxed{\\frac{4}{3}}``).

    Returns ``(start, end, inner)`` where ``end`` is exclusive and matches the substring
    ``text[start:end] == r'\\boxed{' + inner + '}'`` with proper nesting.
    """
    out: List[Tuple[int, int, str]] = []
    i = 0
    n = len(text)
    blen = len(_BOXED_BEGIN)
    while i < n:
        j = text.find(_BOXED_BEGIN, i)
        if j < 0:
            break
        body_start = j + blen
        depth = 1
        k = body_start
        while k < n and depth > 0:
            ch = text[k]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            k += 1
        if depth != 0:
            i = j + 1
            continue
        inner = text[body_start : k - 1]
        out.append((j, k, inner))
        i = k
    return out


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text).strip().lower()


def _levenshtein(a: str, b: str) -> int:
    """Classic edit distance (small strings only; completions are bounded in training)."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins, delete, sub = cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def _triple_repeat_from_strings(parts: list[str], *, lev_threshold: int) -> bool:
    """True iff some three consecutive strings are pairwise close under Levenshtein (threshold)."""
    if len(parts) < 3:
        return False
    normed = [_normalize_text(p) for p in parts]
    for i in range(len(normed) - 2):
        a, b, c = normed[i], normed[i + 1], normed[i + 2]
        if lev_threshold <= 0:
            if a == b == c:
                return True
            continue
        if (
            _levenshtein(a, b) <= lev_threshold
            and _levenshtein(b, c) <= lev_threshold
            and _levenshtein(a, c) <= lev_threshold * 2 + 1
        ):
            return True
    return False


def triple_repeat_answer_penalty(
    text: str,
    *,
    lev_threshold: int = 0,
) -> float:
    """
    Penalize repetitive answering:
    - consecutive three identical ``\\boxed{...}`` inner strings (after whitespace strip), or
    - consecutive three identical inter-``\\boxed`` text segments (whitespace-normalized),
      using Levenshtein distance when ``lev_threshold > 0`` to treat near-duplicates as equal.
    Returns ``1.0`` if a pattern is found (caller multiplies by weight), else ``0.0``.
    """
    if not (text or "").strip():
        return 0.0
    boxed = _find_boxed_balanced(text)
    boxed_inners = [inn.strip() for _, _, inn in boxed]
    if _triple_repeat_from_strings(boxed_inners, lev_threshold=lev_threshold):
        return 1.0
    between: List[str] = []
    if boxed:
        between.append(text[: boxed[0][0]].strip())
        for i in range(len(boxed) - 1):
            between.append(text[boxed[i][1] : boxed[i + 1][0]].strip())
        between.append(text[boxed[-1][1] :].strip())
    between = [b for b in between if b]
    if _triple_repeat_from_strings(between, lev_threshold=lev_threshold):
        return 1.0
    return 0.0


def _char_start_of_last_token_fraction(text: str, tokenizer, frac: float) -> int:
    """
    Character index where the last ``frac`` fraction of tokenizer tokens begins (0-based).
    Falls back to ``int((1-frac)*len(text))`` if tokenization fails.
    """
    if frac <= 0 or not (text or "").strip():
        return 0
    if tokenizer is None:
        # Without a real tokenizer, char-length (1-frac)*len(text) is a poor proxy for "last N% tokens"
        # (e.g. a long leading run of single chars pushes tail_start past a trailing \\boxed). Use non-whitespace
        # runs as a coarse token stand-in so smoke tests and rare no-tokenizer paths behave sensibly.
        spans = list(re.finditer(r"\S+", text))
        nt = len(spans)
        if nt == 0:
            return int((1.0 - frac) * max(1, len(text)))
        j0 = int((1.0 - frac) * nt)
        j0 = max(0, min(j0, nt - 1))
        return int(spans[j0].start())
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc.get("input_ids") or []
        offs = enc.get("offset_mapping") or []
        nt = len(ids)
        if nt == 0 or len(offs) < nt:
            return int((1.0 - frac) * len(text))
        j0 = int((1.0 - frac) * nt)
        j0 = max(0, min(j0, nt - 1))
        span_s, _ = offs[j0]
        return int(span_s)
    except Exception:
        return int((1.0 - frac) * len(text))


def _longest_consecutive_boxed_run_for_matches(text: str, matches: List[Tuple[int, int, str]]) -> tuple[int, int]:
    """
    Return (start_idx, run_len) over the given boxed match list where neighbors are separated only by whitespace.
    ``start_idx`` indexes into ``matches``.
    """
    if not matches:
        return -1, 0

    best_start, best_len = 0, 1
    cur_start, cur_len = 0, 1
    for i in range(1, len(matches)):
        between = text[matches[i - 1][1] : matches[i][0]]
        if between.strip() == "":
            cur_len += 1
        else:
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
            cur_start, cur_len = i, 1
    if cur_len > best_len:
        best_start, best_len = cur_start, cur_len
    return best_start, best_len


def _longest_consecutive_boxed_run(text: str) -> tuple[int, int]:
    """Same as `_longest_consecutive_boxed_run_for_matches` over all balanced ``\\boxed`` in ``text``."""
    return _longest_consecutive_boxed_run_for_matches(text or "", _find_boxed_balanced(text or ""))


def _extract_final_answer(text: str, *, for_ground_truth: bool = False) -> str:
    """
    Prefer the **last** tag/boxed match: last ``<answer>...</answer>``, else last balanced ``\\boxed{...}``,
    else the last non-empty line (after stripping thinking blocks).

    For model completions (``for_ground_truth=False``), if ``configure_math_reward_extraction`` set a positive
    ``boxed_last_token_fraction``, only matches whose **start** lies in the last that fraction of **tokens** count.
    """
    text = _strip_thinking_blocks(text or "")

    frac = 0.0 if for_ground_truth else float(_MATH_REWARD_EXTRACT_CFG.get("boxed_last_token_fraction") or 0.0)
    tok = None if for_ground_truth else _MATH_REWARD_EXTRACT_CFG.get("tokenizer")
    tail_start = _char_start_of_last_token_fraction(text, tok, frac) if frac > 0 else 0

    if frac > 0 and not for_ground_truth:
        last_ans = None
        for m in _ANSWER_TAG_RE.finditer(text):
            if m.start() >= tail_start:
                last_ans = m.group(1).strip()
        if last_ans is not None:
            return last_ans
    else:
        tag_matches = _ANSWER_TAG_RE.findall(text)
        if tag_matches:
            return tag_matches[-1].strip()

    boxed = _find_boxed_balanced(text)
    if frac > 0 and not for_ground_truth:
        boxed = [b for b in boxed if b[0] >= tail_start]
    if boxed:
        run_start, run_len = _longest_consecutive_boxed_run_for_matches(text, boxed)
        # If there is a consecutive boxed run (e.g. boxed spam), score by the FIRST boxed answer.
        if run_start >= 0 and run_len >= 2:
            return boxed[run_start][2].strip()
        return boxed[-1][2].strip()

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    if frac > 0 and not for_ground_truth:
        # Last line only counts if it begins in the tail token region (same char threshold proxy).
        last_line = lines[-1]
        idx = text.rfind(last_line)
        if idx >= 0 and idx >= tail_start:
            return last_line
        return ""
    return lines[-1]


def extract_math_reward_answer(text: str, *, for_ground_truth: bool = False) -> str:
    """Same extraction as math reward (``_extract_final_answer``); for logs/snapshots and debugging."""
    return _extract_final_answer(text or "", for_ground_truth=for_ground_truth)


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

    pred_answer = _extract_final_answer(completion, for_ground_truth=False)
    gt_answer = _extract_final_answer(gt, for_ground_truth=True)

    # Only compare extracted answers (not the full completion) to avoid spurious ``math_verify`` hits on long CoT.
    if _symbolic_match(pred_answer, gt_answer):
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
    repeat_triplet_penalty: float = 0.15,
    repeat_triplet_levenshtein_threshold: int = 0,
) -> List[float]:
    """
    Correctness reward minus optional penalties:
    - If ``ended_with_eos[i]`` is False, subtract ``no_eos_penalty`` (natural stop vs length/truncation).
    - If the longest whitespace-separated run of ``\\boxed{...}`` is >= ``min_consecutive_boxed``,
      subtract ``multi_boxed_penalty`` (discourage repeated boxed spam).
    - If three consecutive boxed inners or three consecutive inter-box segments count as equal
      (exact when ``repeat_triplet_levenshtein_threshold`` is 0, else Levenshtein-based),
      subtract ``repeat_triplet_penalty`` once for that completion.
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
        if float(repeat_triplet_penalty) > 0.0 and triple_repeat_answer_penalty(
            c, lev_threshold=int(repeat_triplet_levenshtein_threshold)
        ):
            penalty += float(repeat_triplet_penalty)
        rewards.append(max(0.0, min(1.0, base - penalty)))
    return rewards
