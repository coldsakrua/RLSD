import json
import os
from dataclasses import dataclass
from typing import Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import GRPOConfig

from data_utils import (
    DEFAULT_MATH_INSTRUCTION_SUFFIX,
    coerce_prompt_to_qwen3_user_messages,
    load_rlsd_dataset,
    normalize_prompt_to_standard_instruction,
)
from official_opsd_trainer import (
    DEFAULT_OFFICIAL_TEACHER_PROMPT,
    DEFAULT_TRANSITION_PROMPT,
    OfficialOPSDDataCollator,
    OfficialOPSDTrainer,
)
from opsd_train_anchor import apply_prompt_wrapping, enforce_lora_only_trainable
from reward_fn import configure_math_reward_extraction
from run_logging import StructuredJsonMetricsCallback, configure_wandb_offline


@dataclass
class OfficialOPSDScriptArguments:
    model_name_or_path: str
    dataset_path: str
    dataset_split: str = "train"
    dataset_cache_dir: Optional[str] = None
    run_config: str = "opsd_4b"
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    normalize_math_prompt_to_standard_suffix: bool = True
    math_instruction_suffix: str = DEFAULT_MATH_INSTRUCTION_SUFFIX
    use_dapo_raw_prompt: bool = False

    # Official OPSD objective.
    lmbda: float = 1.0
    jsd_token_clip: float = 1e-6
    top_k_loss: int = 0
    fixed_teacher: bool = True
    teacher_prompt_template: str = DEFAULT_OFFICIAL_TEACHER_PROMPT
    teacher_transition_prompt: str = DEFAULT_TRANSITION_PROMPT
    max_teacher_prompt_length: Optional[int] = None
    student_prompt_as_chat: bool = False
    student_enable_thinking: bool = False
    teacher_enable_thinking: bool = False

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

    vllm_sync_frequency: int = 1
    save_generation_steps: int = 0
    disable_thinking_in_chat_template: bool = True
    reward_boxed_last_token_fraction: float = 0.05
    disable_wandb: bool = False
    generation_extra_kwargs_json: Optional[str] = None


def _resolve_torch_dtype(name: Optional[str]):
    if name is None:
        return None
    value = str(name).strip()
    if not value:
        return None
    if value == "auto":
        return "auto"
    if hasattr(torch, value):
        return getattr(torch, value)
    raise ValueError(f"Unsupported torch dtype: {name}")


def build_peft_config(args: OfficialOPSDScriptArguments) -> Optional[LoraConfig]:
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


def _patch_qwen_thinking(tokenizer) -> None:
    original_apply_chat = tokenizer.apply_chat_template

    def apply_chat_no_think(messages, *args, **kwargs):
        kw = dict(kwargs)
        kw["enable_thinking"] = False
        try:
            return original_apply_chat(messages, *args, **kw)
        except TypeError:
            kw.pop("enable_thinking", None)
            return original_apply_chat(messages, *args, **kw)

    tokenizer.apply_chat_template = apply_chat_no_think
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None and inner is not tokenizer and hasattr(inner, "apply_chat_template"):
        original_inner_apply = inner.apply_chat_template

        def inner_apply_no_think(messages, *args, **kwargs):
            kw = dict(kwargs)
            kw["enable_thinking"] = False
            try:
                return original_inner_apply(messages, *args, **kw)
            except TypeError:
                kw.pop("enable_thinking", None)
                return original_inner_apply(messages, *args, **kw)

        inner.apply_chat_template = inner_apply_no_think


def main() -> None:
    parser = HfArgumentParser((OfficialOPSDScriptArguments, GRPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    if script_args.dataset_cache_dir:
        os.environ["HF_DATASETS_CACHE"] = script_args.dataset_cache_dir

    logging_setup = configure_wandb_offline(
        training_args,
        disable_wandb=bool(script_args.disable_wandb),
        run_name=script_args.run_config if script_args.run_config else None,
        extra_meta={
            "entrypoint": os.path.basename(__file__),
            "algorithm": "official_opsd_compat",
        },
    )
    print(f"[wandb] meta_path={logging_setup['meta_path']}", flush=True)

    training_args.remove_unused_columns = False
    if training_args.gradient_checkpointing and getattr(training_args, "gradient_checkpointing_kwargs", None) in (None, {}):
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    if script_args.max_length is not None:
        max_completion_length = getattr(training_args, "max_completion_length", None)
        if max_completion_length is None:
            raise ValueError("When --max_length is set, --max_completion_length must also be set.")
        training_args.max_prompt_length = max(32, int(script_args.max_length) - int(max_completion_length))
        print(
            f"[length_budget] max_length={script_args.max_length}, "
            f"max_completion_length={max_completion_length}, "
            f"computed_max_prompt_length={training_args.max_prompt_length}",
            flush=True,
        )

    if script_args.generation_extra_kwargs_json and str(script_args.generation_extra_kwargs_json).strip():
        try:
            extra_generation_kwargs = json.loads(script_args.generation_extra_kwargs_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--generation_extra_kwargs_json is not valid JSON: {exc}") from exc
        if not isinstance(extra_generation_kwargs, dict):
            raise ValueError("--generation_extra_kwargs_json must be a JSON object.")
    else:
        extra_generation_kwargs = {}
    merged_generation_kwargs = dict(getattr(training_args, "generation_kwargs", None) or {})
    merged_generation_kwargs.update(extra_generation_kwargs)
    if hasattr(training_args, "generation_kwargs"):
        training_args.generation_kwargs = merged_generation_kwargs

    if script_args.disable_thinking_in_chat_template and hasattr(training_args, "chat_template_kwargs"):
        chat_template_kwargs = dict(getattr(training_args, "chat_template_kwargs") or {})
        chat_template_kwargs["enable_thinking"] = False
        training_args.chat_template_kwargs = chat_template_kwargs

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

    def prepare_rollout_prompt(row):
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

    if use_raw_prompt_passthrough:
        print("[prompt_mode] raw DAPO prompt passthrough: skip rollout prompt map.", flush=True)
    else:
        steps = []
        if do_prompt_standardize:
            steps.append("normalize")
        if script_args.prompt_prefix or script_args.prompt_suffix:
            steps.append("wrap")
        steps.append("raw_prompt_passthrough" if script_args.use_dapo_raw_prompt else "qwen3_chat_messages")
        train_dataset = train_dataset.map(
            prepare_rollout_prompt,
            desc=f"Prepare OPSD rollout prompt ({' + '.join(steps)})",
        )

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    configure_math_reward_extraction(
        tokenizer=tokenizer,
        boxed_last_token_fraction=float(script_args.reward_boxed_last_token_fraction),
    )

    if script_args.disable_thinking_in_chat_template:
        _patch_qwen_thinking(tokenizer)
        print("[chat_template] enable_thinking=False tokenizer patch installed.", flush=True)

    model_init_kwargs = dict(getattr(training_args, "model_init_kwargs", None) or {})
    if script_args.attn_implementation:
        model_init_kwargs["attn_implementation"] = script_args.attn_implementation
    torch_dtype = _resolve_torch_dtype(script_args.torch_dtype)
    if torch_dtype is not None:
        model_init_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        **model_init_kwargs,
    )
    if training_args.gradient_checkpointing:
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    peft_config = build_peft_config(script_args)
    if peft_config is not None:
        model = get_peft_model(model, peft_config)

    if script_args.fixed_teacher and peft_config is None:
        raise ValueError("--fixed_teacher true requires --use_peft true.")

    if script_args.use_peft and script_args.strict_lora_only:
        enforce_lora_only_trainable(model)

    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_param_count = sum(p.numel() for p in model.parameters())
    lora_trainable_count = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and "lora_" in name.lower()
    )
    non_lora_trainable = [
        name for name, p in model.named_parameters() if p.requires_grad and "lora_" not in name.lower()
    ]
    print(
        f"[trainable] trainable_params={trainable_param_count}, "
        f"lora_trainable_params={lora_trainable_count}, "
        f"total_params={total_param_count}, "
        f"use_peft={script_args.use_peft}, strict_lora_only={script_args.strict_lora_only}",
        flush=True,
    )
    if trainable_param_count == 0:
        raise RuntimeError("No trainable parameters found. Check --use_peft and --lora_target_modules.")
    if script_args.use_peft and script_args.strict_lora_only and non_lora_trainable:
        preview = ", ".join(non_lora_trainable[:8])
        raise RuntimeError(f"Found non-LoRA trainable params under strict_lora_only: {preview}")

    max_completion_length = int(getattr(training_args, "max_completion_length", 1024) or 1024)
    max_student_prompt_length = getattr(training_args, "max_prompt_length", None)
    max_teacher_prompt_length = script_args.max_teacher_prompt_length
    if max_teacher_prompt_length is None:
        max_teacher_prompt_length = max_student_prompt_length

    data_collator = OfficialOPSDDataCollator(
        tokenizer,
        student_prompt_as_chat=script_args.student_prompt_as_chat,
        student_thinking=script_args.student_enable_thinking,
        teacher_thinking=script_args.teacher_enable_thinking,
        teacher_prompt_template=script_args.teacher_prompt_template,
        teacher_transition_prompt=script_args.teacher_transition_prompt,
    )

    trainer = OfficialOPSDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        max_student_prompt_length=max_student_prompt_length,
        max_teacher_prompt_length=max_teacher_prompt_length,
        max_completion_length=max_completion_length,
        lmbda=script_args.lmbda,
        beta=float(getattr(training_args, "beta", 0.0) or 0.0),
        temperature=float(getattr(training_args, "temperature", 1.0) or 1.0),
        top_p=float(getattr(training_args, "top_p", 1.0) or 1.0),
        top_k=int(getattr(training_args, "top_k", 0) or 0),
        min_p=float(getattr(training_args, "min_p", 0.0) or 0.0),
        repetition_penalty=float(getattr(training_args, "repetition_penalty", 1.0) or 1.0),
        presence_penalty=float(merged_generation_kwargs.get("presence_penalty", 0.0) or 0.0),
        generation_extra_kwargs=merged_generation_kwargs,
        fixed_teacher=script_args.fixed_teacher,
        top_k_loss=script_args.top_k_loss,
        jsd_token_clip=script_args.jsd_token_clip,
        use_vllm=bool(getattr(training_args, "use_vllm", False)),
        vllm_guided_decoding_regex=getattr(training_args, "vllm_guided_decoding_regex", None),
        vllm_sync_frequency=script_args.vllm_sync_frequency,
        save_generation_steps=script_args.save_generation_steps,
    )

    metrics_jsonl_path = logging_setup["metrics_jsonl_path"]
    trainer.add_callback(StructuredJsonMetricsCallback(metrics_jsonl_path))
    print(f"[metrics] jsonl_path={metrics_jsonl_path}", flush=True)
    print(
        f"[official_opsd] beta={getattr(training_args, 'beta', 0.0)} "
        f"jsd_token_clip={script_args.jsd_token_clip} "
        f"top_k_loss={script_args.top_k_loss} "
        f"fixed_teacher={script_args.fixed_teacher} "
        f"student_prompt_len={max_student_prompt_length} "
        f"teacher_prompt_len={max_teacher_prompt_length} "
        f"completion_len={max_completion_length}",
        flush=True,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
