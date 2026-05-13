import json
import os
from dataclasses import dataclass
from typing import Optional

from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, HfArgumentParser
from trl import GRPOConfig, GRPOTrainer

from data_utils import (
    DEFAULT_MATH_INSTRUCTION_SUFFIX,
    coerce_prompt_to_qwen3_user_messages,
    load_rlsd_dataset,
    normalize_prompt_to_standard_instruction,
)
from reward_fn import (
    configure_math_reward_extraction,
    verifiable_math_reward,
    verifiable_math_reward_with_format_penalties,
)
from run_logging import StructuredJsonMetricsCallback, configure_wandb_offline


@dataclass
class ScriptArguments:
    model_name_or_path: str
    dataset_path: str
    dataset_split: str = "train"
    dataset_cache_dir: Optional[str] = None
    run_config: str = "grpo_strict_4b"
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    normalize_math_prompt_to_standard_suffix: bool = True
    math_instruction_suffix: str = DEFAULT_MATH_INSTRUCTION_SUFFIX
    use_dapo_raw_prompt: bool = False

    max_length: Optional[int] = None
    attn_implementation: Optional[str] = None
    torch_dtype: str = "bfloat16"

    use_peft: bool = False
    strict_lora_only: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_target_modules: str = (
        "q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
    )

    disable_wandb: bool = False
    # Kept for CLI compatibility with strict RLSD script; plain GRPO trainer does not emit rollout snapshots here.
    save_rollout_snapshots: bool = False
    rollout_snapshot_interval_steps: int = 0
    generation_extra_kwargs_json: Optional[str] = None

    reward_format_penalties: bool = True
    reward_no_eos_penalty: float = 0.15
    reward_multi_boxed_penalty: float = 0.15
    reward_min_consecutive_boxed: int = 2
    reward_repeat_triplet_penalty: float = 0.15
    reward_repeat_triplet_levenshtein_threshold: int = 0
    disable_thinking_in_chat_template: bool = True
    reward_boxed_last_token_fraction: float = 0.0


def _to_text_completion(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if completion and isinstance(completion[-1], dict) and "content" in completion[-1]:
            return str(completion[-1]["content"])
    return str(completion)


def build_reward_fn(args: ScriptArguments):
    def reward_fn(completions, solution, ended_with_eos=None, **kwargs):
        text_completions = [_to_text_completion(c) for c in completions]
        if args.reward_format_penalties:
            return verifiable_math_reward_with_format_penalties(
                text_completions,
                solution,
                ended_with_eos=ended_with_eos,
                no_eos_penalty=args.reward_no_eos_penalty,
                multi_boxed_penalty=args.reward_multi_boxed_penalty,
                min_consecutive_boxed=args.reward_min_consecutive_boxed,
                repeat_triplet_penalty=args.reward_repeat_triplet_penalty,
                repeat_triplet_levenshtein_threshold=args.reward_repeat_triplet_levenshtein_threshold,
            )
        return verifiable_math_reward(text_completions, solution)

    return reward_fn


def apply_prompt_wrapping(prompt, prefix: str, suffix: str):
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


def build_peft_config(args: ScriptArguments) -> Optional[LoraConfig]:
    if not args.use_peft:
        return None
    target_modules = [x.strip() for x in args.lora_target_modules.replace(",", " ").split() if x.strip()]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )


def enforce_lora_only_trainable(model) -> None:
    for name, param in model.named_parameters():
        param.requires_grad_("lora_" in name.lower())


def main():
    parser = HfArgumentParser((ScriptArguments, GRPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    if script_args.dataset_cache_dir:
        os.environ["HF_DATASETS_CACHE"] = script_args.dataset_cache_dir
    logging_setup = configure_wandb_offline(
        training_args,
        disable_wandb=bool(script_args.disable_wandb),
        run_name=script_args.run_config if script_args.run_config else None,
        extra_meta={"entrypoint": os.path.basename(__file__)},
    )
    print(f"[wandb] meta_path={logging_setup['meta_path']}", flush=True)

    training_args.remove_unused_columns = False
    if training_args.gradient_checkpointing and getattr(training_args, "gradient_checkpointing_kwargs", None) in (None, {}):
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    model_init_kwargs = dict(training_args.model_init_kwargs or {})
    if script_args.attn_implementation:
        model_init_kwargs["attn_implementation"] = script_args.attn_implementation
    if script_args.torch_dtype:
        model_init_kwargs["torch_dtype"] = script_args.torch_dtype
    training_args.model_init_kwargs = model_init_kwargs

    if script_args.max_length is not None:
        if training_args.max_completion_length is None:
            raise ValueError("When --max_length is set, --max_completion_length must also be set.")
        training_args.max_prompt_length = max(32, script_args.max_length - training_args.max_completion_length)
        print(
            f"[length_budget] max_length={script_args.max_length}, "
            f"max_completion_length={training_args.max_completion_length}, "
            f"computed_max_prompt_length={training_args.max_prompt_length}",
            flush=True,
        )
        if training_args.max_prompt_length <= 64:
            print(
                "[length_budget][warn] max_prompt_length is very small (<=64). "
                "This can cause severe prompt truncation and off-topic generations.",
                flush=True,
            )

    if script_args.generation_extra_kwargs_json and str(script_args.generation_extra_kwargs_json).strip():
        try:
            extra = json.loads(script_args.generation_extra_kwargs_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"--generation_extra_kwargs_json is not valid JSON: {e}") from e
        if not isinstance(extra, dict):
            raise ValueError("--generation_extra_kwargs_json must be a JSON object, e.g. '{\"presence_penalty\": 0.2}'")
        merged = dict(training_args.generation_kwargs or {})
        merged.update(extra)
        training_args.generation_kwargs = merged

    if script_args.disable_thinking_in_chat_template and hasattr(training_args, "chat_template_kwargs"):
        _ct = dict(getattr(training_args, "chat_template_kwargs") or {})
        _ct["enable_thinking"] = False
        training_args.chat_template_kwargs = _ct

    train_dataset = load_rlsd_dataset(
        script_args.dataset_path,
        split=script_args.dataset_split,
        normalize_dapo_prompt=not script_args.use_dapo_raw_prompt,
    )
    if script_args.use_dapo_raw_prompt and script_args.normalize_math_prompt_to_standard_suffix:
        print(
            "[prompt_mode] use_dapo_raw_prompt=True -> skip standard suffix normalization in training map.",
            flush=True,
        )

    do_prompt_standardize = (
        bool(script_args.normalize_math_prompt_to_standard_suffix)
        and not bool(script_args.use_dapo_raw_prompt)
    )
    use_raw_prompt_passthrough = (
        bool(script_args.use_dapo_raw_prompt)
        and not do_prompt_standardize
        and not script_args.prompt_prefix
        and not script_args.prompt_suffix
    )

    def _prepare_rollout_prompt(row):
        prompt = row.get("prompt", "")
        if do_prompt_standardize:
            prompt = normalize_prompt_to_standard_instruction(
                prompt,
                suffix=script_args.math_instruction_suffix,
            )
        if script_args.prompt_prefix or script_args.prompt_suffix:
            prompt = apply_prompt_wrapping(
                prompt,
                script_args.prompt_prefix,
                script_args.prompt_suffix,
            )
        if not script_args.use_dapo_raw_prompt:
            prompt = coerce_prompt_to_qwen3_user_messages(prompt)
        return {**row, "prompt": prompt}

    _steps = []
    if do_prompt_standardize:
        _steps.append("normalize")
    if script_args.prompt_prefix or script_args.prompt_suffix:
        _steps.append("wrap")
    if script_args.use_dapo_raw_prompt:
        _steps.append("raw_prompt_passthrough")
    else:
        _steps.append("qwen3_chat_messages")

    if use_raw_prompt_passthrough:
        print(
            "[prompt_mode] raw DAPO prompt passthrough: skip rollout prompt map.",
            flush=True,
        )
    else:
        train_dataset = train_dataset.map(
            _prepare_rollout_prompt,
            desc=f"Prepare rollout prompt ({' + '.join(_steps)})",
        )

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    configure_math_reward_extraction(
        tokenizer=tokenizer,
        boxed_last_token_fraction=float(script_args.reward_boxed_last_token_fraction),
    )

    if script_args.disable_thinking_in_chat_template:
        _orig_apply_chat = tokenizer.apply_chat_template

        def _apply_chat_no_think(messages, *args, **kwargs):
            kw = dict(kwargs)
            kw["enable_thinking"] = False
            try:
                return _orig_apply_chat(messages, *args, **kw)
            except TypeError:
                kw.pop("enable_thinking", None)
                return _orig_apply_chat(messages, *args, **kw)

        tokenizer.apply_chat_template = _apply_chat_no_think
        _inner = getattr(tokenizer, "tokenizer", None)
        if _inner is not None and _inner is not tokenizer and hasattr(_inner, "apply_chat_template"):
            _orig_inner_apply = _inner.apply_chat_template

            def _inner_apply_no_think(messages, *args, **kwargs):
                kw = dict(kwargs)
                kw["enable_thinking"] = False
                try:
                    return _orig_inner_apply(messages, *args, **kw)
                except TypeError:
                    kw.pop("enable_thinking", None)
                    return _orig_inner_apply(messages, *args, **kw)

            _inner.apply_chat_template = _inner_apply_no_think
        print(
            "[chat_template] enable_thinking=False: tokenizer monkey-patch + "
            f"training_args.chat_template_kwargs={getattr(training_args, 'chat_template_kwargs', {})} "
            "(disable_thinking_in_chat_template=True)",
            flush=True,
        )

    peft_config = build_peft_config(script_args)

    trainer = GRPOTrainer(
        model=script_args.model_name_or_path,
        reward_funcs=build_reward_fn(script_args),
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    metrics_jsonl_path = logging_setup["metrics_jsonl_path"]
    trainer.add_callback(StructuredJsonMetricsCallback(metrics_jsonl_path))
    print(f"[metrics] jsonl_path={metrics_jsonl_path}")
    if script_args.save_rollout_snapshots:
        print("[rollout_snapshot] skipped: plain GRPO trainer does not produce RLSD rollout snapshots.")

    model_for_grad = trainer.model
    if hasattr(trainer, "accelerator"):
        model_for_grad = trainer.accelerator.unwrap_model(model_for_grad)
    if training_args.gradient_checkpointing:
        if hasattr(model_for_grad, "enable_input_require_grads"):
            model_for_grad.enable_input_require_grads()
        if (not script_args.use_peft or not script_args.strict_lora_only) and hasattr(
            model_for_grad, "get_input_embeddings"
        ):
            input_embeddings = model_for_grad.get_input_embeddings()
            if input_embeddings is not None and hasattr(input_embeddings, "weight"):
                input_embeddings.weight.requires_grad_(True)

    if script_args.use_peft and script_args.strict_lora_only:
        enforce_lora_only_trainable(model_for_grad)

    trainable_param_count = sum(p.numel() for p in model_for_grad.parameters() if p.requires_grad)
    total_param_count = sum(p.numel() for p in model_for_grad.parameters())
    lora_trainable_count = sum(
        p.numel()
        for name, p in model_for_grad.named_parameters()
        if p.requires_grad and "lora_" in name.lower()
    )
    non_lora_trainable = [
        name for name, p in model_for_grad.named_parameters() if p.requires_grad and "lora_" not in name.lower()
    ]
    print(
        f"[trainable] trainable_params={trainable_param_count}, "
        f"lora_trainable_params={lora_trainable_count}, "
        f"total_params={total_param_count}, "
        f"use_peft={script_args.use_peft}, strict_lora_only={script_args.strict_lora_only}"
    )
    if trainable_param_count == 0:
        raise RuntimeError(
            "No trainable parameters found. Check --use_peft and --lora_target_modules."
        )
    if script_args.use_peft and script_args.strict_lora_only and non_lora_trainable:
        preview = ", ".join(non_lora_trainable[:8])
        raise RuntimeError(f"Found non-LoRA trainable params under strict_lora_only: {preview}")

    trainer.train()
    trainer.save_model(training_args.output_dir)
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
