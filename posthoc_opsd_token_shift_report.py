#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-hoc streaming analysis for OPSD token-shift JSON logs.

The raw OPSD logs can be larger than memory because each trajectory stores full
text plus per-token teacher/student log-probabilities.  This script streams the
top-level ``trajectories`` array and keeps only compact sufficient statistics.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, DefaultDict, Dict, Iterable, Iterator, List, Optional, Tuple


BUCKETS = ("correct_up", "correct_down", "wrong_up", "wrong_down")
TAIL_FRAC = 0.05


def iter_trajectories(path: Path, *, chunk_size: int = 1 << 20) -> Iterator[Dict[str, Any]]:
    """Yield trajectory objects from the top-level trajectories array."""
    decoder = json.JSONDecoder()
    key = '"trajectories"'
    buf = ""
    found = False

    with path.open("r", encoding="utf-8", errors="replace") as f:
        while not found:
            chunk = f.read(chunk_size)
            if not chunk:
                return
            buf += chunk
            key_pos = buf.find(key)
            if key_pos < 0:
                buf = buf[-len(key) - 32 :]
                continue
            arr_pos = buf.find("[", key_pos + len(key))
            if arr_pos < 0:
                more = f.read(chunk_size)
                if not more:
                    return
                buf += more
                arr_pos = buf.find("[", key_pos + len(key))
                if arr_pos < 0:
                    return
            buf = buf[arr_pos + 1 :]
            found = True

        while True:
            while True:
                stripped = buf.lstrip()
                buf = stripped
                if buf.startswith(","):
                    buf = buf[1:]
                    continue
                break

            while not buf:
                chunk = f.read(chunk_size)
                if not chunk:
                    return
                buf += chunk
                buf = buf.lstrip()

            if buf.startswith("]"):
                return

            while True:
                try:
                    obj, end = decoder.raw_decode(buf)
                    yield obj
                    buf = buf[end:]
                    break
                except json.JSONDecodeError:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        raise
                    buf += chunk


def short_text(s: str, *, max_len: int = 500) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3].rstrip() + "..."


def final_excerpt(s: str, *, max_len: int = 420) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    if len(s) <= max_len:
        return s
    return "..." + s[-max_len + 3 :].lstrip()


def answer_guess(s: str) -> str:
    text = str(s)
    boxed = re.findall(r"\\boxed\s*\{([^{}]{0,80})\}", text)
    if boxed:
        return boxed[-1].strip()
    ans = re.findall(r"Answer:\s*([^\n\r]{0,120})", text, flags=re.IGNORECASE)
    if ans:
        return ans[-1].strip()
    return ""


def classify_token(token: str) -> str:
    text = str(token)
    stripped = text.strip()
    lower = stripped.lower()
    if "<|" in text or "|>" in text:
        return "special"
    if not stripped:
        return "whitespace"
    if any(ch == "\ufffd" for ch in text) or (not text.isascii()):
        return "non_ascii"
    if re.fullmatch(r"[\d,./+-]+", stripped):
        return "numeric"
    if any(x in stripped for x in ("\\boxed", "Answer", "answer")):
        return "answer_marker"
    if re.fullmatch(r"[\$\\{}()[\],.;:!?\-_=+*/^<>|]+", stripped):
        return "math_or_punct"
    if lower in {
        "so",
        "therefore",
        "thus",
        "hence",
        "since",
        "final",
        "step",
        "now",
        "then",
        "because",
    }:
        return "reasoning_marker"
    return "content"


@dataclass
class BucketStats:
    n: int = 0
    n_tail: int = 0
    sum_abs: float = 0.0
    sum_abs_sq: float = 0.0
    sum_signed: float = 0.0
    sum_prob_abs: float = 0.0
    values_abs: List[float] = field(default_factory=list)
    tail_abs: float = 0.0
    token_counts: Counter = field(default_factory=Counter)
    token_abs: DefaultDict[str, float] = field(default_factory=lambda: defaultdict(float))
    token_abs_sq: DefaultDict[str, float] = field(default_factory=lambda: defaultdict(float))
    token_signed: DefaultDict[str, float] = field(default_factory=lambda: defaultdict(float))
    token_trajs: DefaultDict[str, set] = field(default_factory=lambda: defaultdict(set))
    traj_ids: set = field(default_factory=set)

    def add(
        self,
        *,
        traj_key: Tuple[int, int],
        token_key: str,
        signed_delta: float,
        delta_prob: float,
        in_tail: bool,
    ) -> None:
        mag = abs(float(signed_delta))
        self.n += 1
        self.sum_abs += mag
        self.sum_abs_sq += mag * mag
        self.sum_signed += float(signed_delta)
        self.sum_prob_abs += abs(float(delta_prob))
        self.values_abs.append(mag)
        self.token_counts[token_key] += 1
        self.token_abs[token_key] += mag
        self.token_abs_sq[token_key] += mag * mag
        self.token_signed[token_key] += float(signed_delta)
        self.token_trajs[token_key].add(traj_key)
        self.traj_ids.add(traj_key)
        if in_tail:
            self.n_tail += 1
            self.tail_abs += mag

    def token_stats(self, token_key: str) -> Dict[str, Any]:
        count = int(self.token_counts[token_key])
        mass = float(self.token_abs[token_key])
        mean_abs = mass / count if count else 0.0
        mean_sq = float(self.token_abs_sq[token_key]) / count if count else 0.0
        var_abs = max(0.0, mean_sq - mean_abs * mean_abs)
        std_abs = math.sqrt(var_abs)
        token_id_str, token_text = token_key.split(":", 1)
        return {
            "token": token_text,
            "token_id": int(token_id_str),
            "class": classify_token(token_text),
            "count": count,
            "trajectory_count": len(self.token_trajs[token_key]),
            "abs_delta_logp_mass": mass,
            "mass_share": mass / self.sum_abs if self.sum_abs else 0.0,
            "mean_signed_delta_logp": float(self.token_signed[token_key]) / count if count else 0.0,
            "mean_abs_delta_logp": mean_abs,
            "var_abs_delta_logp": var_abs,
            "std_abs_delta_logp": std_abs,
            "cv_abs_delta_logp": std_abs / mean_abs if mean_abs > 0 else 0.0,
            "occurrence_per_trajectory": count / max(1, len(self.token_trajs[token_key])),
        }

    def topk_stability(self, ks: Iterable[int] = (10, 25, 50)) -> Dict[str, Any]:
        ranked_keys = sorted(self.token_abs, key=lambda k: self.token_abs[k], reverse=True)
        out: Dict[str, Any] = {}
        for k in ks:
            keys = ranked_keys[:k]
            if not keys:
                out[str(k)] = {
                    "mass_coverage": 0.0,
                    "weighted_mean_cv": 0.0,
                    "weighted_mean_var": 0.0,
                    "median_cv": 0.0,
                    "median_var": 0.0,
                    "trajectory_coverage": 0,
                    "class_mass_share": {},
                }
                continue
            rows = [self.token_stats(key) for key in keys]
            mass = sum(float(r["abs_delta_logp_mass"]) for r in rows)
            trajs = set()
            class_mass: DefaultDict[str, float] = defaultdict(float)
            for key, row in zip(keys, rows):
                trajs.update(self.token_trajs[key])
                class_mass[str(row["class"])] += float(row["abs_delta_logp_mass"])
            cvs = sorted(float(r["cv_abs_delta_logp"]) for r in rows)
            vars_ = sorted(float(r["var_abs_delta_logp"]) for r in rows)
            out[str(k)] = {
                "mass_coverage": mass / self.sum_abs if self.sum_abs else 0.0,
                "weighted_mean_cv": sum(
                    float(r["cv_abs_delta_logp"]) * float(r["abs_delta_logp_mass"]) for r in rows
                )
                / mass
                if mass
                else 0.0,
                "weighted_mean_var": sum(
                    float(r["var_abs_delta_logp"]) * float(r["abs_delta_logp_mass"]) for r in rows
                )
                / mass
                if mass
                else 0.0,
                "median_cv": median(cvs),
                "median_var": median(vars_),
                "trajectory_coverage": len(trajs),
                "class_mass_share": {
                    cls: value / mass if mass else 0.0 for cls, value in sorted(class_mass.items())
                },
            }
        return out

    def as_dict(self) -> Dict[str, Any]:
        vals = sorted(self.values_abs)
        n = self.n
        mean_abs = self.sum_abs / n if n else 0.0
        var_abs = self.sum_abs_sq / n - mean_abs * mean_abs if n else 0.0
        std_abs = math.sqrt(max(0.0, var_abs))

        def pct(q: float) -> float:
            if not vals:
                return 0.0
            idx = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
            return vals[idx]

        total_count = sum(self.token_counts.values())
        top10 = self.token_counts.most_common(10)
        top10_share = sum(c for _, c in top10) / total_count if total_count else 0.0
        entropy = 0.0
        if total_count:
            for count in self.token_counts.values():
                p = count / total_count
                entropy -= p * math.log2(p)
        singleton_occ = sum(c for c in self.token_counts.values() if c == 1)
        rare_le2_occ = sum(c for c in self.token_counts.values() if c <= 2)
        return {
            "tokens": n,
            "trajectories": len(self.traj_ids),
            "mean_signed_delta_logp": self.sum_signed / n if n else 0.0,
            "mean_abs_delta_logp": mean_abs,
            "std_abs_delta_logp": std_abs,
            "cv_abs_delta_logp": std_abs / mean_abs if mean_abs > 0 else 0.0,
            "median_abs_delta_logp": median(vals) if vals else 0.0,
            "p90_abs_delta_logp": pct(0.90),
            "p95_abs_delta_logp": pct(0.95),
            "p99_abs_delta_logp": pct(0.99),
            "mean_abs_delta_prob": self.sum_prob_abs / n if n else 0.0,
            "tail_token_frac": self.n_tail / n if n else 0.0,
            "tail_mass_frac": self.tail_abs / self.sum_abs if self.sum_abs else 0.0,
            "unique_tokens": len(self.token_counts),
            "unique_per_1k_occ": 1000.0 * len(self.token_counts) / n if n else 0.0,
            "token_entropy_bits": entropy,
            "effective_vocab": 2.0**entropy if entropy < 60 else float("inf"),
            "top10_token_share": top10_share,
            "singleton_occ_frac": singleton_occ / total_count if total_count else 0.0,
            "rare_le2_occ_frac": rare_le2_occ / total_count if total_count else 0.0,
            "top_tokens_by_count": [
                {"token": tok.split(":", 1)[1], "token_id": int(tok.split(":", 1)[0]), "count": count}
                for tok, count in top10
            ],
            "top_tokens_by_abs_mass": [
                self.token_stats(tok)
                for tok, mass in sorted(self.token_abs.items(), key=lambda kv: kv[1], reverse=True)[:10]
            ],
            "top_tokens_by_stability_score": [
                self.token_stats(tok)
                for tok in sorted(
                    self.token_abs,
                    key=lambda key: self.token_abs[key]
                    / (1.0 + self.token_stats(key)["cv_abs_delta_logp"]),
                    reverse=True,
                )[:10]
            ],
            "topk_stability_by_abs_mass": self.topk_stability(),
        }


@dataclass
class TrajIntensity:
    values: List[float] = field(default_factory=list)
    nonzero_count: int = 0

    def add(self, value: float) -> None:
        self.values.append(float(value))
        if value > 0:
            self.nonzero_count += 1

    def as_dict(self) -> Dict[str, Any]:
        n = len(self.values)
        if not n:
            return {
                "n_trajectories": 0,
                "nonzero_frac": 0.0,
                "mean_mass_per_token": 0.0,
                "std_mass_per_token": 0.0,
                "cv_mass_per_token": 0.0,
                "snr_mean_over_std": 0.0,
                "median_mass_per_token": 0.0,
                "p90_mass_per_token": 0.0,
                "p95_mass_per_token": 0.0,
            }
        vals = sorted(self.values)
        mean = sum(vals) / n
        var = sum((x - mean) ** 2 for x in vals) / n
        std = math.sqrt(max(0.0, var))

        def pct(q: float) -> float:
            idx = min(n - 1, max(0, int(round(q * (n - 1)))))
            return vals[idx]

        return {
            "n_trajectories": n,
            "nonzero_frac": self.nonzero_count / n,
            "mean_mass_per_token": mean,
            "std_mass_per_token": std,
            "cv_mass_per_token": std / mean if mean > 0 else 0.0,
            "snr_mean_over_std": mean / std if std > 0 else 0.0,
            "median_mass_per_token": median(vals),
            "p90_mass_per_token": pct(0.90),
            "p95_mass_per_token": pct(0.95),
        }


def maybe_push_candidate(
    heap: List[Tuple[float, int, Dict[str, Any]]],
    *,
    score: float,
    seq: int,
    limit: int,
    item: Dict[str, Any],
) -> None:
    if len(heap) < limit:
        heapq.heappush(heap, (score, seq, item))
    elif score > heap[0][0]:
        heapq.heapreplace(heap, (score, seq, item))


def build_span(
    *,
    traj: Dict[str, Any],
    tokens: List[Dict[str, Any]],
    start: int,
    end: int,
    bucket: str,
    score: float,
    token_count: int,
) -> Dict[str, Any]:
    lo = max(0, start - 20)
    hi = min(len(tokens), end + 20)
    center = "".join(str(t.get("token", "")) for t in tokens[start:end])
    context = "".join(str(t.get("token", "")) for t in tokens[lo:hi])
    return {
        "bucket": bucket,
        "score_abs_delta_logp": score,
        "span_token_count": token_count,
        "sample_pos": traj.get("sample_pos"),
        "dataset_index": traj.get("dataset_index"),
        "completion_idx": traj.get("completion_idx"),
        "correct": bool(traj.get("correct")),
        "reward": traj.get("reward"),
        "solution": traj.get("solution", ""),
        "answer_guess": answer_guess(str(traj.get("completion_text", ""))),
        "token_pos_start": int(start),
        "token_pos_end": int(end - 1),
        "span_text": short_text(center, max_len=360),
        "span_context": short_text(context, max_len=700),
        "completion_tail": final_excerpt(str(traj.get("completion_text", "")), max_len=520),
    }


def summarize_group_patterns(group_counts: Dict[int, List[bool]]) -> Dict[str, Any]:
    out = {
        "groups": len(group_counts),
        "all_correct": 0,
        "all_wrong": 0,
        "mixed": 0,
        "by_correct_count": Counter(),
    }
    for oks in group_counts.values():
        c = sum(1 for x in oks if x)
        out["by_correct_count"][str(c)] += 1
        if c == len(oks):
            out["all_correct"] += 1
        elif c == 0:
            out["all_wrong"] += 1
        else:
            out["mixed"] += 1
    out["by_correct_count"] = dict(sorted(out["by_correct_count"].items(), key=lambda kv: int(kv[0])))
    return out


def analyze(path: Path, *, max_trajectories: Optional[int], candidate_limit: int) -> Dict[str, Any]:
    bucket_stats = {b: BucketStats() for b in BUCKETS}
    intensities = {b: TrajIntensity() for b in BUCKETS}
    direction_counts_by_correct = {
        True: Counter(),
        False: Counter(),
    }
    group_counts: Dict[int, List[bool]] = defaultdict(list)
    zero_var_bucket_mass: Dict[str, DefaultDict[str, float]] = {
        "all_correct": defaultdict(float),
        "all_wrong": defaultdict(float),
    }
    candidates: Dict[str, List[Tuple[float, int, Dict[str, Any]]]] = {b: [] for b in BUCKETS}
    tail_candidates: Dict[str, List[Tuple[float, int, Dict[str, Any]]]] = {b: [] for b in BUCKETS}
    candidate_seq = 0

    n_traj = 0
    n_correct = 0
    n_tokens = 0
    len_by_correct = {True: [], False: []}

    for traj in iter_trajectories(path):
        n_traj += 1
        if max_trajectories is not None and n_traj > max_trajectories:
            break

        correct = bool(traj.get("correct"))
        if correct:
            n_correct += 1
        sample_pos = int(traj.get("sample_pos", -1))
        completion_idx = int(traj.get("completion_idx", -1))
        group_counts[sample_pos].append(correct)

        tokens = traj.get("tokens") or []
        total = len(tokens)
        if total <= 0:
            continue
        n_tokens += total
        len_by_correct[correct].append(total)
        traj_key = (sample_pos, completion_idx)
        tail_start = max(0, int(math.floor((1.0 - TAIL_FRAC) * total)))
        per_traj_mass = {b: 0.0 for b in BUCKETS}

        current_bucket = None
        run_start = 0
        run_score = 0.0
        run_count = 0

        def flush_run(end_pos: int) -> None:
            nonlocal candidate_seq, current_bucket, run_start, run_score, run_count
            if current_bucket is None or run_count <= 0:
                return
            if run_count >= 3:
                candidate_seq += 1
                item = build_span(
                    traj=traj,
                    tokens=tokens,
                    start=run_start,
                    end=end_pos,
                    bucket=current_bucket,
                    score=run_score,
                    token_count=run_count,
                )
                maybe_push_candidate(
                    candidates[current_bucket],
                    score=run_score,
                    seq=candidate_seq,
                    limit=candidate_limit,
                    item=item,
                )
                if run_start >= tail_start:
                    maybe_push_candidate(
                        tail_candidates[current_bucket],
                        score=run_score,
                        seq=candidate_seq,
                        limit=candidate_limit,
                        item=item,
                    )
            current_bucket = None
            run_start = end_pos
            run_score = 0.0
            run_count = 0

        for i, tok in enumerate(tokens):
            d = float(tok.get("delta_logp", 0.0))
            if d > 0:
                direction = "up"
            elif d < 0:
                direction = "down"
            else:
                direction = "same"
            direction_counts_by_correct[correct][direction] += 1
            if direction == "same":
                flush_run(i)
                current_bucket = None
                continue

            bucket = ("correct_" if correct else "wrong_") + direction
            mag = abs(d)
            token_key = f"{int(tok.get('token_id', -1))}:{tok.get('token', '')}"
            in_tail = i >= tail_start
            bucket_stats[bucket].add(
                traj_key=traj_key,
                token_key=token_key,
                signed_delta=d,
                delta_prob=float(tok.get("delta_prob", 0.0)),
                in_tail=in_tail,
            )
            per_traj_mass[bucket] += mag

            if bucket != current_bucket:
                flush_run(i)
                current_bucket = bucket
                run_start = i
                run_score = mag
                run_count = 1
            else:
                run_score += mag
                run_count += 1

        flush_run(total)

        for bucket in BUCKETS:
            is_correct_bucket = bucket.startswith("correct_")
            if is_correct_bucket == correct:
                intensities[bucket].add(per_traj_mass[bucket] / total)

    # Group-level zero-variance usefulness: compute after group labels are known.
    group_kind = {}
    for sample_pos, oks in group_counts.items():
        if oks and all(oks):
            group_kind[sample_pos] = "all_correct"
        elif oks and not any(oks):
            group_kind[sample_pos] = "all_wrong"

    # A second light stream pass attributes bucket mass to all-correct/all-wrong groups.
    if group_kind:
        for pass_idx, traj in enumerate(iter_trajectories(path), start=1):
            if max_trajectories is not None and pass_idx > max_trajectories:
                break
            kind = group_kind.get(int(traj.get("sample_pos", -1)))
            if kind is None:
                continue
            for tok in traj.get("tokens") or []:
                d = float(tok.get("delta_logp", 0.0))
                if d > 0:
                    direction = "up"
                elif d < 0:
                    direction = "down"
                else:
                    continue
                bucket = ("correct_" if bool(traj.get("correct")) else "wrong_") + direction
                zero_var_bucket_mass[kind][bucket] += abs(d)
                zero_var_bucket_mass[kind]["tokens"] += 1.0
                zero_var_bucket_mass[kind]["abs_delta_logp"] += abs(d)

    def heap_to_list(heap: List[Tuple[float, int, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        return [item for _, _, item in sorted(heap, key=lambda x: x[0], reverse=True)]

    total_correct_tokens = sum(direction_counts_by_correct[True].values())
    total_wrong_tokens = sum(direction_counts_by_correct[False].values())
    group_summary = summarize_group_patterns(group_counts)
    result = {
        "source": str(path),
        "trajectory_count": n_traj if max_trajectories is None else min(n_traj, max_trajectories),
        "correct_trajectory_count": n_correct,
        "wrong_trajectory_count": (n_traj if max_trajectories is None else min(n_traj, max_trajectories)) - n_correct,
        "correct_ratio": n_correct / max(1, (n_traj if max_trajectories is None else min(n_traj, max_trajectories))),
        "token_count": n_tokens,
        "direction_counts": {
            "correct": dict(direction_counts_by_correct[True]),
            "wrong": dict(direction_counts_by_correct[False]),
        },
        "direction_token_fracs": {
            "correct_up": direction_counts_by_correct[True]["up"] / total_correct_tokens
            if total_correct_tokens
            else 0.0,
            "correct_down": direction_counts_by_correct[True]["down"] / total_correct_tokens
            if total_correct_tokens
            else 0.0,
            "wrong_up": direction_counts_by_correct[False]["up"] / total_wrong_tokens
            if total_wrong_tokens
            else 0.0,
            "wrong_down": direction_counts_by_correct[False]["down"] / total_wrong_tokens
            if total_wrong_tokens
            else 0.0,
        },
        "completion_length": {
            "correct_mean": sum(len_by_correct[True]) / len(len_by_correct[True]) if len_by_correct[True] else 0.0,
            "wrong_mean": sum(len_by_correct[False]) / len(len_by_correct[False]) if len_by_correct[False] else 0.0,
            "correct_median": median(len_by_correct[True]) if len_by_correct[True] else 0.0,
            "wrong_median": median(len_by_correct[False]) if len_by_correct[False] else 0.0,
        },
        "bucket_stats": {b: bucket_stats[b].as_dict() for b in BUCKETS},
        "trajectory_intensity": {b: intensities[b].as_dict() for b in BUCKETS},
        "group_correctness": group_summary,
        "zero_variance_group_mass": {
            kind: dict(vals) for kind, vals in zero_var_bucket_mass.items()
        },
        "representative_spans": {b: heap_to_list(candidates[b]) for b in BUCKETS},
        "tail_representative_spans": {b: heap_to_list(tail_candidates[b]) for b in BUCKETS},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--max_trajectories", type=int, default=None)
    parser.add_argument("--candidate_limit", type=int, default=8)
    args = parser.parse_args()

    result = analyze(
        args.input_json,
        max_trajectories=args.max_trajectories,
        candidate_limit=max(1, int(args.candidate_limit)),
    )
    out = args.output_json
    if out is None:
        suffix = ".posthoc_report"
        if args.max_trajectories:
            suffix += f".first{args.max_trajectories}"
        out = args.input_json.with_name(args.input_json.stem + suffix + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {out}")
    print(
        json.dumps(
            {
                "trajectory_count": result["trajectory_count"],
                "correct_ratio": result["correct_ratio"],
                "token_count": result["token_count"],
                "direction_token_fracs": result["direction_token_fracs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
