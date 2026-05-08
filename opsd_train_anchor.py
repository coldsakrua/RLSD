import os
from dataclasses import dataclass, field
from typing import List, Optional

from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, HfArgumentParser
from trl import GRPOConfig

from data_utils import load_rlsd_dataset
from reward_fn import verifiable_math_reward
from rlsd_trainer import RLSDTrainer


@dataclass
class ScriptArguments:
    model_name_or_path: str
    dataset_path: str
    dataset_split: str = "train"
    dataset_cache_dir: Optional[str] = None
    run_config: str = "rlsd_anchor"

    lmbda: float = 0.5
    lmbda_decay_steps: int = 50
    jsd_token_clip: float = 0.2
    rollout_filter: str = "all"
    fixed_teacher: bool = False
    teacher_prompt_template: str = (
        "{prompt}\n\n[Reference solution]\n{solution}\n\n[Student response]\n"
    )

    max_length: Optional[int] = None
    attn_implementation: Optional[str] = None
    torch_dtype: str = "bfloat16"

    use_peft: bool = False
    lora_r: int = 64
    lora_alpha: int = 128
    lora_target_modules: str = (
        "q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
    )

    disable_wandb: bool = False


def _to_text_completion(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if completion and isinstance(completion[-1], dict) and "content" in completion[-1]:
            return str(completion[-1]["content"])
    return str(completion)


def reward_fn(completions, solution, **kwargs):
    text_completions = [_to_text_completion(c) for c in completions]
    return verifiable_math_reward(text_completions, solution)


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

    training_args.remove_unused_columns = False

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

    train_dataset = load_rlsd_dataset(script_args.dataset_path, split=script_args.dataset_split)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = build_peft_config(script_args)

    trainer = RLSDTrainer(
        model=script_args.model_name_or_path,
        reward_funcs=reward_fn,
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
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
