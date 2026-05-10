import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, HfArgumentParser, TrainerCallback
from trl import GRPOConfig

from data_utils import load_rlsd_dataset
from reward_fn import verifiable_math_reward, verifiable_math_reward_with_format_penalties
from rlsd_rollout_snapshot import SaveRolloutSnapshotCallback
from rlsd_sign_fallback_strict_trainer import RLSDSignFallbackStrictTrainer


@dataclass
class ScriptArguments:
    model_name_or_path: str
    dataset_path: str
    dataset_split: str = "train"
    dataset_cache_dir: Optional[str] = None
    run_config: str = "rlsd_strict_4b"
    prompt_prefix: str = ""
    prompt_suffix: str = ""

    # mixed-group RLSD
    lmbda: float = 0.5
    lmbda_decay_steps: int = 50
    jsd_token_clip: float = 0.05
    rollout_filter: str = "all"
    fixed_teacher: bool = True
    teacher_prompt_template: str = (
        "{prompt}\n\n[Reference solution]\n{solution}\n\n[Student response]\n"
    )

    # all-correct/all-wrong fallback
    lambda_plus: float = 0.03
    lambda_minus: float = 0.03
    lambda_plus_min: float = 0.0
    lambda_minus_min: float = 0.0
    fallback_decay_steps: int = 200
    fallback_eps0: float = 0.05
    adv_clip_low: float = -1.0
    adv_clip_high: float = 1.0
    suppress_gt_shortcut: bool = True
    answer_token_downweight: float = 0.2
    reward_binary_threshold: float = 0.5
    fallback_tail_tokens: int = 8
    require_eos_for_positive_reward: bool = True
    mask_truncated_advantages: bool = True
    # Penalties applied on top of correctness (see reward_fn.verifiable_math_reward_with_format_penalties).
    reward_format_penalties: bool = True
    reward_no_eos_penalty: float = 0.15
    reward_multi_boxed_penalty: float = 0.15
    reward_min_consecutive_boxed: int = 3
    # DAPO-style asymmetric clipping for positive-advantage samples:
    # upper clip bound becomes (1 + epsilon_high) for adv>0.
    dapo_epsilon_high: Optional[float] = None

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
    # When true, each checkpoint save also writes rollout_snapshot_step_*.json (last mini-batch rollout).
    save_rollout_snapshots: bool = True


def _to_text_completion(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if completion and isinstance(completion[-1], dict) and "content" in completion[-1]:
            return str(completion[-1]["content"])
    return str(completion)


def build_reward_fn(args: ScriptArguments):
    """Closure so format-penalty weights live in ScriptArguments."""

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
            )
        return verifiable_math_reward(text_completions, solution)

    return reward_fn


def apply_prompt_wrapping(prompt: str, prefix: str, suffix: str) -> str:
    p = prompt if isinstance(prompt, str) else str(prompt)
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
    """Freeze all non-LoRA parameters to guarantee adapter-only updates."""
    for name, param in model.named_parameters():
        param.requires_grad_("lora_" in name.lower())


class JsonMetricsCallback(TrainerCallback):
    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path
        os.makedirs(os.path.dirname(self.jsonl_path), exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
        }
        record.update(logs)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


def main():
    parser = HfArgumentParser((ScriptArguments, GRPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    if script_args.dataset_cache_dir:
        os.environ["HF_DATASETS_CACHE"] = script_args.dataset_cache_dir
    if script_args.disable_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        training_args.report_to = []
    if script_args.run_config:
        training_args.run_name = script_args.run_config

    if script_args.dapo_epsilon_high is not None:
        setattr(training_args, "epsilon_high", float(script_args.dapo_epsilon_high))

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

    setattr(training_args, "save_rollout_snapshots", bool(script_args.save_rollout_snapshots))

    train_dataset = load_rlsd_dataset(script_args.dataset_path, split=script_args.dataset_split)
    if script_args.prompt_prefix or script_args.prompt_suffix:
        train_dataset = train_dataset.map(
            lambda row: {
                **row,
                "prompt": apply_prompt_wrapping(
                    row.get("prompt", ""),
                    script_args.prompt_prefix,
                    script_args.prompt_suffix,
                ),
            },
            desc="Applying rollout prompt wrapping",
        )

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = build_peft_config(script_args)

    trainer = RLSDSignFallbackStrictTrainer(
        model=script_args.model_name_or_path,
        reward_funcs=build_reward_fn(script_args),
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        lmbda=script_args.lmbda,
        lmbda_decay_steps=script_args.lmbda_decay_steps,
        jsd_token_clip=script_args.jsd_token_clip,
        fixed_teacher=script_args.fixed_teacher,
        rollout_filter=script_args.rollout_filter,
        teacher_prompt_template=script_args.teacher_prompt_template,
        lambda_plus=script_args.lambda_plus,
        lambda_minus=script_args.lambda_minus,
        lambda_plus_min=script_args.lambda_plus_min,
        lambda_minus_min=script_args.lambda_minus_min,
        fallback_decay_steps=script_args.fallback_decay_steps,
        fallback_eps0=script_args.fallback_eps0,
        adv_clip_low=script_args.adv_clip_low,
        adv_clip_high=script_args.adv_clip_high,
        suppress_gt_shortcut=script_args.suppress_gt_shortcut,
        answer_token_downweight=script_args.answer_token_downweight,
        reward_binary_threshold=script_args.reward_binary_threshold,
        fallback_tail_tokens=script_args.fallback_tail_tokens,
        require_eos_for_positive_reward=script_args.require_eos_for_positive_reward,
        mask_truncated_advantages=script_args.mask_truncated_advantages,
    )

    metrics_jsonl_path = os.path.join(training_args.output_dir, "train_metrics.jsonl")
    trainer.add_callback(JsonMetricsCallback(metrics_jsonl_path))
    print(f"[metrics] jsonl_path={metrics_jsonl_path}")
    if script_args.save_rollout_snapshots:
        trainer.add_callback(SaveRolloutSnapshotCallback(trainer))
        print(f"[rollout_snapshot] enabled -> {training_args.output_dir}/rollout_snapshot_step_*.json on each save")

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
