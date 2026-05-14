import json
import os

from transformers import AutoTokenizer, HfArgumentParser
from trl import GRPOConfig

from data_utils import (
    coerce_prompt_to_qwen3_user_messages,
    load_rlsd_dataset,
    normalize_prompt_to_standard_instruction,
)
from opsd_split_metrics_trainer import OPSDSplitMetricsTrainer
from opsd_train_anchor import (
    ScriptArguments,
    apply_prompt_wrapping,
    build_peft_config,
    build_reward_fn,
    enforce_lora_only_trainable,
)
from reward_fn import configure_math_reward_extraction
from rlsd_rollout_snapshot import SaveRolloutSnapshotCallback
from run_logging import StructuredJsonMetricsCallback, configure_wandb_offline


def main():
    parser = HfArgumentParser((ScriptArguments, GRPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    if getattr(script_args, "run_config", None) == "rlsd_anchor":
        script_args.run_config = "opsd_pure_4b"

    if script_args.dataset_cache_dir:
        os.environ["HF_DATASETS_CACHE"] = script_args.dataset_cache_dir
    logging_setup = configure_wandb_offline(
        training_args,
        disable_wandb=bool(script_args.disable_wandb),
        run_name=script_args.run_config if script_args.run_config else None,
        extra_meta={"entrypoint": os.path.basename(__file__)},
    )
    print(f"[wandb] meta_path={logging_setup['meta_path']}", flush=True)

    if script_args.use_sign_constrained_fallback:
        print(
            "[opsd_pure] ignore --use_sign_constrained_fallback=true and fallback-specific args; "
            "this entrypoint always runs pure OPSD.",
            flush=True,
        )

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

    setattr(training_args, "save_rollout_snapshots", bool(script_args.save_rollout_snapshots))
    setattr(
        training_args,
        "rollout_snapshot_interval_steps",
        int(script_args.rollout_snapshot_interval_steps),
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

    trainer = OPSDSplitMetricsTrainer(
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
        teacher_update_interval_steps=script_args.teacher_update_interval_steps,
        reward_binary_threshold=script_args.reward_binary_threshold,
    )

    metrics_jsonl_path = logging_setup["metrics_jsonl_path"]
    trainer.add_callback(StructuredJsonMetricsCallback(metrics_jsonl_path))
    print(f"[metrics] jsonl_path={metrics_jsonl_path}")
    if script_args.save_rollout_snapshots:
        trainer.add_callback(SaveRolloutSnapshotCallback(trainer))
        _iv = int(getattr(training_args, "rollout_snapshot_interval_steps", 0) or 0)
        _extra = f" + every {_iv} steps" if _iv > 0 else ""
        print(
            f"[rollout_snapshot] enabled -> {training_args.output_dir}/rollout_snapshot_step_*.json "
            f"on checkpoint save{_extra}"
        )

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
