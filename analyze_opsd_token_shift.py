#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import (
    DEFAULT_MATH_INSTRUCTION_SUFFIX,
    coerce_prompt_to_qwen3_user_messages,
    extract_last_user_text,
    load_rlsd_dataset,
    normalize_prompt_to_standard_instruction,
)
from reward_fn import configure_math_reward_extraction, verifiable_math_reward


def _to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse bool value: {x}")


def _resolve_dtype(name: str) -> torch.dtype:
    n = str(name).strip().lower()
    if n in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if n in {"fp16", "float16", "half"}:
        return torch.float16
    if n in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def _safe_jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _safe_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe_jsonable(x) for x in v]
    return str(v)


def apply_prompt_wrapping(prompt: Any, prefix: str, suffix: str) -> Any:
    if not prefix and not suffix:
        return prompt
    if isinstance(prompt, list):
        out = [dict(m) if isinstance(m, dict) else m for m in prompt]
        last_user_idx = None
        for i, msg in enumerate(out):
            if isinstance(msg, dict) and str(msg.get("role", "")).lower() == "user":
                last_user_idx = i
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
                    t = str(part.get("text", ""))
                    if prefix:
                        t = f"{prefix}{t}"
                    if suffix:
                        t = f"{t}{suffix}"
                    new_parts.append({**part, "text": t})
                else:
                    new_parts.append(part)
            user_msg["content"] = new_parts
        else:
            t = str(content)
            if prefix:
                t = f"{prefix}{t}"
            if suffix:
                t = f"{t}{suffix}"
            user_msg["content"] = t
        out[last_user_idx] = user_msg
        return out

    p = prompt.strip() if isinstance(prompt, str) else str(prompt).strip()
    if prefix:
        p = f"{prefix}{p}"
    if suffix:
        p = f"{p}{suffix}"
    return p


def _build_train_like_dataset(
    *,
    dataset_path: str,
    dataset_split: str,
    use_dapo_raw_prompt: bool,
    normalize_math_prompt_to_standard_suffix: bool,
    math_instruction_suffix: str,
    prompt_prefix: str,
    prompt_suffix: str,
):
    ds = load_rlsd_dataset(
        dataset_path,
        split=dataset_split,
        normalize_dapo_prompt=not use_dapo_raw_prompt,
    )

    do_prompt_standardize = bool(normalize_math_prompt_to_standard_suffix) and (not bool(use_dapo_raw_prompt))
    use_raw_prompt_passthrough = (
        bool(use_dapo_raw_prompt)
        and not do_prompt_standardize
        and not prompt_prefix
        and not prompt_suffix
    )

    def _prepare_rollout_prompt(row):
        prompt = row.get("prompt", "")
        if do_prompt_standardize:
            prompt = normalize_prompt_to_standard_instruction(prompt, suffix=math_instruction_suffix)
        if prompt_prefix or prompt_suffix:
            prompt = apply_prompt_wrapping(prompt, prompt_prefix, prompt_suffix)
        if not use_dapo_raw_prompt:
            prompt = coerce_prompt_to_qwen3_user_messages(prompt)
        return {**row, "prompt": prompt}

    if not use_raw_prompt_passthrough:
        ds = ds.map(_prepare_rollout_prompt, desc="Prepare rollout prompt (analysis)")
    return ds


def _prompt_to_teacher_text(prompt: Any) -> str:
    text = extract_last_user_text(prompt)
    if not text:
        return text
    try:
        normalized = normalize_prompt_to_standard_instruction(text)
        if isinstance(normalized, str):
            return normalized
    except Exception:
        pass
    return text


def _apply_chat_no_think(tokenizer, messages: Any, *, enable_thinking: bool = False) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": bool(enable_thinking)}
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _prompt_to_generation_text(tokenizer, prompt: Any, *, enable_thinking: bool = False) -> str:
    messages = coerce_prompt_to_qwen3_user_messages(prompt)
    return _apply_chat_no_think(tokenizer, messages, enable_thinking=enable_thinking)


def _compute_completion_token_logps(
    model,
    *,
    prefix_ids: torch.Tensor,
    completion_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Return per-token log p(y_t | prefix + y_<t) for tokens in completion_ids.
    Shape: [completion_len]
    """
    input_ids = torch.cat([prefix_ids, completion_ids], dim=0).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # Predict token t from position t-1.
    token_logps = torch.log_softmax(logits[:, :-1, :], dim=-1)
    labels = input_ids[:, 1:]
    gathered = token_logps.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(0).squeeze(-1)
    comp_len = int(completion_ids.numel())
    return gathered[-comp_len:].detach().cpu()


def _sample_dataset_indices(n: int, k: int, seed: int) -> List[int]:
    if n <= 0:
        return []
    k = max(1, min(k, n))
    rng = random.Random(seed)
    all_idx = list(range(n))
    rng.shuffle(all_idx)
    return sorted(all_idx[:k])


@dataclass
class TrajSummary:
    sample_idx: int
    completion_idx: int
    reward: float
    correct: bool
    completion_text: str


def _aggregate_token_push_pull(
    traj_token_rows: Iterable[Dict[str, Any]],
    *,
    top_k: int,
) -> Dict[str, Any]:
    by_tok: Dict[str, Dict[str, float]] = {}
    for row in traj_token_rows:
        tok = str(row["token"])
        item = by_tok.get(tok)
        if item is None:
            item = {
                "count": 0.0,
                "delta_logp_sum": 0.0,
                "delta_prob_sum": 0.0,
                "up_count": 0.0,
                "down_count": 0.0,
            }
            by_tok[tok] = item
        item["count"] += 1.0
        dl = float(row["delta_logp"])
        dp = float(row["delta_prob"])
        item["delta_logp_sum"] += dl
        item["delta_prob_sum"] += dp
        if dl > 0:
            item["up_count"] += 1.0
        elif dl < 0:
            item["down_count"] += 1.0

    stats: List[Dict[str, Any]] = []
    for tok, v in by_tok.items():
        c = max(1.0, v["count"])
        stats.append(
            {
                "token": tok,
                "count": int(v["count"]),
                "mean_delta_logp": v["delta_logp_sum"] / c,
                "mean_delta_prob": v["delta_prob_sum"] / c,
                "up_frac": v["up_count"] / c,
                "down_frac": v["down_count"] / c,
            }
        )

    stats_push = sorted(stats, key=lambda x: x["mean_delta_logp"], reverse=True)
    stats_pull = sorted(stats, key=lambda x: x["mean_delta_logp"])
    return {
        "unique_tokens": len(stats),
        "top_push_tokens_by_mean_delta_logp": stats_push[:top_k],
        "top_pull_tokens_by_mean_delta_logp": stats_pull[:top_k],
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample prompts and analyze per-token student vs OPSD-teacher probability shifts. "
            "Supports optional LoRA loading."
        )
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_cache_dir", type=str, default="")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_teacher_prompt_length", type=int, default=3072)
    parser.add_argument("--max_new_tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--presence_penalty", type=float, default=0.2)
    parser.add_argument("--do_sample", type=_to_bool, default=True)
    parser.add_argument("--enable_thinking", type=_to_bool, default=False)

    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")

    parser.add_argument("--prompt_prefix", type=str, default="")
    parser.add_argument("--prompt_suffix", type=str, default="")
    parser.add_argument("--normalize_math_prompt_to_standard_suffix", type=_to_bool, default=False)
    parser.add_argument("--math_instruction_suffix", type=str, default=DEFAULT_MATH_INSTRUCTION_SUFFIX)
    parser.add_argument("--use_dapo_raw_prompt", type=_to_bool, default=True)
    parser.add_argument("--reward_binary_threshold", type=float, default=0.5)
    parser.add_argument("--reward_boxed_last_token_fraction", type=float, default=0.05)
    parser.add_argument(
        "--teacher_prompt_template",
        type=str,
        default="{prompt}\n\n[Reference solution]\n{solution}\n\n[Student response]\n",
    )
    parser.add_argument("--summary_top_k", type=int, default=30)
    args = parser.parse_args()

    if args.dataset_cache_dir:
        import os

        os.environ["HF_DATASETS_CACHE"] = args.dataset_cache_dir

    out_path = Path(args.output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = _resolve_dtype(args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    configure_math_reward_extraction(
        tokenizer=tokenizer,
        boxed_last_token_fraction=float(args.reward_boxed_last_token_fraction),
    )

    model_init_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if args.attn_implementation:
        model_init_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_init_kwargs)
    if args.lora_path.strip():
        model = PeftModel.from_pretrained(model, args.lora_path.strip(), is_trainable=False)
    model.to(args.device)
    model.eval()

    ds = _build_train_like_dataset(
        dataset_path=args.dataset_path,
        dataset_split=args.dataset_split,
        use_dapo_raw_prompt=bool(args.use_dapo_raw_prompt),
        normalize_math_prompt_to_standard_suffix=bool(args.normalize_math_prompt_to_standard_suffix),
        math_instruction_suffix=args.math_instruction_suffix,
        prompt_prefix=args.prompt_prefix,
        prompt_suffix=args.prompt_suffix,
    )
    n_rows = len(ds)
    chosen_indices = _sample_dataset_indices(n_rows, args.sample_size, args.seed)
    rows = [ds[int(i)] for i in chosen_indices]

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    all_traj_records: List[Dict[str, Any]] = []
    all_token_rows: List[Dict[str, Any]] = []
    all_correct_token_rows: List[Dict[str, Any]] = []
    all_wrong_token_rows: List[Dict[str, Any]] = []
    traj_summaries: List[TrajSummary] = []

    for sample_pos, row in enumerate(rows):
        prompt_obj = row.get("prompt", "")
        solution = str(row.get("solution", ""))

        rollout_prompt_text = _prompt_to_generation_text(
            tokenizer,
            prompt_obj,
            enable_thinking=bool(args.enable_thinking),
        )
        teacher_prompt_text = args.teacher_prompt_template.format(
            prompt=_prompt_to_teacher_text(prompt_obj),
            solution=solution,
        )

        prompt_ids = tokenizer(
            rollout_prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=int(args.max_prompt_length),
        )["input_ids"][0].to(args.device)

        gen_kwargs = dict(
            max_new_tokens=int(args.max_new_tokens),
            do_sample=bool(args.do_sample),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
            num_return_sequences=int(args.num_generations),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        if int(args.top_k) > 0:
            gen_kwargs["top_k"] = int(args.top_k)
        if float(args.min_p) > 0.0:
            gen_kwargs["min_p"] = float(args.min_p)
        if float(args.presence_penalty) != 0.0:
            # Transformers generate() does not natively expose presence_penalty for all models.
            # Keep this as metadata only to match CLI intent.
            pass

        with torch.no_grad():
            try:
                outputs = model.generate(
                    input_ids=prompt_ids.unsqueeze(0),
                    attention_mask=torch.ones_like(prompt_ids.unsqueeze(0)),
                    **gen_kwargs,
                )
            except TypeError:
                # Backward-compatible fallback for older Transformers that do not support min_p.
                gen_kwargs.pop("min_p", None)
                outputs = model.generate(
                    input_ids=prompt_ids.unsqueeze(0),
                    attention_mask=torch.ones_like(prompt_ids.unsqueeze(0)),
                    **gen_kwargs,
                )
        # [num_generations, prompt_len + new_len]
        if outputs.dim() != 2:
            raise RuntimeError(f"Unexpected generate output shape: {tuple(outputs.shape)}")

        prompt_len = int(prompt_ids.numel())
        completion_list: List[str] = []
        completion_ids_list: List[torch.Tensor] = []
        for seq in outputs:
            comp_ids = seq[prompt_len:]
            eos_id = tokenizer.eos_token_id
            pad_id = tokenizer.pad_token_id
            # Keep through first EOS (inclusive), matching rollout decoding semantics.
            if eos_id is not None:
                eos_pos = (comp_ids == int(eos_id)).nonzero(as_tuple=False)
                if eos_pos.numel() > 0:
                    cut = int(eos_pos[0].item()) + 1
                    comp_ids = comp_ids[:cut]
            # If pad != eos, strip trailing pad area.
            if pad_id is not None and (eos_id is None or int(pad_id) != int(eos_id)):
                pad_pos = (comp_ids == int(pad_id)).nonzero(as_tuple=False)
                if pad_pos.numel() > 0:
                    cut = int(pad_pos[0].item())
                    comp_ids = comp_ids[:cut]
            comp_ids = comp_ids.to(args.device)
            completion_ids_list.append(comp_ids)
            completion_list.append(tokenizer.decode(comp_ids.tolist(), skip_special_tokens=True))

        rewards = verifiable_math_reward(completion_list, [solution] * len(completion_list))
        rewards = [float(r) for r in rewards]
        correctness = [bool(r > float(args.reward_binary_threshold)) for r in rewards]

        teacher_prefix_ids = tokenizer(
            teacher_prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=int(args.max_teacher_prompt_length),
        )["input_ids"][0].to(args.device)

        for j, (comp_ids, comp_text, rew, ok) in enumerate(
            zip(completion_ids_list, completion_list, rewards, correctness)
        ):
            if comp_ids.numel() == 0:
                continue
            student_logps = _compute_completion_token_logps(
                model,
                prefix_ids=prompt_ids,
                completion_ids=comp_ids,
            )
            teacher_logps = _compute_completion_token_logps(
                model,
                prefix_ids=teacher_prefix_ids,
                completion_ids=comp_ids,
            )
            token_ids = comp_ids.detach().cpu().tolist()
            token_texts = [
                tokenizer.decode([int(tid)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                for tid in token_ids
            ]

            per_tok: List[Dict[str, Any]] = []
            for k, (tid, ttxt, s_lp, t_lp) in enumerate(
                zip(token_ids, token_texts, student_logps.tolist(), teacher_logps.tolist())
            ):
                s_lp = float(s_lp)
                t_lp = float(t_lp)
                d_lp = t_lp - s_lp
                # Clamp exp inputs for numerical stability.
                s_p = float(math.exp(max(-80.0, min(20.0, s_lp))))
                t_p = float(math.exp(max(-80.0, min(20.0, t_lp))))
                d_p = t_p - s_p
                direction = "up" if d_lp > 0 else ("down" if d_lp < 0 else "same")
                row_tok = {
                    "token_pos": int(k),
                    "token_id": int(tid),
                    "token": ttxt,
                    "student_logp": s_lp,
                    "teacher_logp": t_lp,
                    "delta_logp": d_lp,
                    "student_prob": s_p,
                    "teacher_prob": t_p,
                    "delta_prob": d_p,
                    "prob_ratio_teacher_over_student": float(math.exp(max(-40.0, min(40.0, d_lp)))),
                    "direction": direction,
                }
                per_tok.append(row_tok)
                all_token_rows.append(row_tok)
                if ok:
                    all_correct_token_rows.append(row_tok)
                else:
                    all_wrong_token_rows.append(row_tok)

            traj = {
                "sample_pos": int(sample_pos),
                "dataset_index": int(chosen_indices[sample_pos]),
                "completion_idx": int(j),
                "reward": float(rew),
                "correct": bool(ok),
                "solution": solution,
                "prompt_raw": _safe_jsonable(prompt_obj),
                "prompt_for_generation_text": rollout_prompt_text,
                "teacher_prompt_text": teacher_prompt_text,
                "completion_text": comp_text,
                "tokens": per_tok,
            }
            all_traj_records.append(traj)
            traj_summaries.append(
                TrajSummary(
                    sample_idx=int(chosen_indices[sample_pos]),
                    completion_idx=int(j),
                    reward=float(rew),
                    correct=bool(ok),
                    completion_text=comp_text,
                )
            )

    correct_count = sum(1 for x in traj_summaries if x.correct)
    total_count = len(traj_summaries)
    wrong_count = total_count - correct_count

    summary = {
        "num_samples": len(rows),
        "num_trajectories": total_count,
        "num_correct_trajectories": correct_count,
        "num_wrong_trajectories": wrong_count,
        "correct_ratio": (float(correct_count) / float(total_count)) if total_count > 0 else 0.0,
        "all_tokens": _aggregate_token_push_pull(all_token_rows, top_k=int(args.summary_top_k)),
        "correct_tokens": _aggregate_token_push_pull(all_correct_token_rows, top_k=int(args.summary_top_k)),
        "wrong_tokens": _aggregate_token_push_pull(all_wrong_token_rows, top_k=int(args.summary_top_k)),
    }

    out_obj = {
        "config": {
            "model_name_or_path": args.model_name_or_path,
            "lora_path": args.lora_path.strip(),
            "dataset_path": args.dataset_path,
            "dataset_split": args.dataset_split,
            "sample_size": int(args.sample_size),
            "num_generations": int(args.num_generations),
            "seed": int(args.seed),
            "max_prompt_length": int(args.max_prompt_length),
            "max_teacher_prompt_length": int(args.max_teacher_prompt_length),
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "min_p": float(args.min_p),
            "repetition_penalty": float(args.repetition_penalty),
            "presence_penalty": float(args.presence_penalty),
            "do_sample": bool(args.do_sample),
            "enable_thinking": bool(args.enable_thinking),
            "torch_dtype": args.torch_dtype,
            "device": args.device,
            "attn_implementation": args.attn_implementation,
            "teacher_prompt_template": args.teacher_prompt_template,
            "reward_binary_threshold": float(args.reward_binary_threshold),
            "reward_boxed_last_token_fraction": float(args.reward_boxed_last_token_fraction),
            "normalize_math_prompt_to_standard_suffix": bool(args.normalize_math_prompt_to_standard_suffix),
            "math_instruction_suffix": args.math_instruction_suffix,
            "use_dapo_raw_prompt": bool(args.use_dapo_raw_prompt),
            "prompt_prefix": args.prompt_prefix,
            "prompt_suffix": args.prompt_suffix,
        },
        "chosen_dataset_indices": chosen_indices,
        "summary": summary,
        "trajectories": all_traj_records,
    }
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote analysis json -> {out_path}")


if __name__ == "__main__":
    main()
