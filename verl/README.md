# veRL RLSD/OPSD/RLRT launchers

This folder contains server-side veRL launch code for the 4B experiments
that exist at the repository root.

## Files

- `verl_rlsd/reward.py`: math reward adapter for veRL custom rewards.
- `verl_rlsd/advantage.py`: custom advantage estimators:
  - `rlsd_grpo`: paper RLSD/OPSD token-gap shaping on GRPO advantages.
  - `rlsd_strict_split_flip`: strict split fallback plus correct-path sign flip.
  - `rlsd_strict_split_flip_wrong_boost`: strict split plus wrong-path positive flip.
  - `rlrt`: RLRT reversed teacher weighting on correct rollouts.
  - `opd_zero`: zero policy-gradient reward for distillation-only OPSD.
- `verl_rlsd/teacher_agent.py`: custom veRL agent loop manager that aligns teacher
  response logprobs under reference-solution, no-reference, identical-student,
  official-OPSD, or successful-rollout teacher prompts.
- `verl_rlsd/teacher_ema.py`: EMA teacher LoRA sync for `cast_ema_4b.sh`.
- `train_scripts/*_4b*.sh`: one fully self-contained sbatch script per veRL training experiment.

## Script mapping

| Root script | veRL script | Main behavior |
| --- | --- | --- |
| `train_scripts/opsd_4b_only.sh` | `verl/train_scripts/grpo_opds_4b.sh` | Pure OPSD-style token-gap shaping with reward, lambda 0.2, decay 50. |
| `train_scripts/opsd_4b.sh` | `verl/train_scripts/opsd_4b.sh` | Official OPSD-style distillation-only run, n=1, official teacher prompt. |
| `train_scripts/grpo_4b_strict.sh` | `verl/train_scripts/grpo_4b.sh` | Strict GRPO baseline, no teacher shaping or distillation. |
| `train_scripts/rlrt_4b.sh` | `verl/train_scripts/rlrt_4b.sh` | RLRT reversed teacher weighting, lambda 0.5, no decay, successful-rollout teacher context. |
| `train_scripts/rlsd_4b_paper.sh` | `verl/train_scripts/rlsd_4b.sh` | Canonical RLSD paper token shaping, lambda 0.5, decay 50. |
| `train_scripts/rlsd_4b_strict_split_flip_nodecay_no_teacher_ref.sh` | `verl/train_scripts/cast_nowrongboost_4b.sh` | Strict split sign flip, lambda 1.0, no decay, no reference solution in teacher prompt. |
| `train_scripts/rlsd_4b_strict_split_flip_wrong_boost_nodecay_teacher_ref.sh` | `verl/train_scripts/cast_4b.sh` | Strict split sign flip plus wrong-path positive boost, lambda 1.0, teacher sees reference solution. |
| — | `verl/train_scripts/cast_ema_4b_256(nogap005).sh` | CAST EMA variant; see `train_scripts/cast_ema_4b*.sh` for other configs. |

## Submit on the server

From the repository root:

```bash
sbatch verl/train_scripts/rlsd_4b.sh
```

Useful overrides:

```bash
BASE_DIR=/gpfs/share/home/2501210611/RLSD \
MODEL_PATH=/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b \
DATASET_PATH=/gpfs/share/home/2501210611/RLSD/data/dapo/dapo-math-17k.parquet \
MAX_STEPS=300 \
sbatch verl/train_scripts/rlsd_4b.sh
```

Most teacher-distillation scripts default to one actor/rollout GPU plus one
teacher GPU:

```bash
ACTOR_GPUS_PER_NODE=1 TEACHER_GPUS_PER_NODE=1 sbatch verl/train_scripts/rlsd_4b.sh
```

`cast_ema_4b.sh` is the exception: its EMA teacher is computed inside the
actor worker, so both GPUs default to the main actor/rollout pool and no
separate teacher GPU is reserved:

```bash
ACTOR_GPUS_PER_NODE=2 DISTILLATION_ENABLED=false sbatch verl/train_scripts/cast_ema_4b_256(nogap005).sh
```

Batch defaults:

- GRPO/RLSD/RLRT/OPSD-only scripts use `TRAIN_BATCH_SIZE=4` prompts and
  `NUM_GENERATIONS=8`, i.e. 32 generated responses per veRL train batch.
- Official OPSD uses `TRAIN_BATCH_SIZE=8` and `NUM_GENERATIONS=1`.
- All scripts default to `PPO_MINI_BATCH_SIZE=4` and
  `PPO_MICRO_BATCH_SIZE_PER_GPU=2` for safer memory use on a 2-GPU A800 node.
- `rlsd_4b.sh` follows the RLSD release/paper sampling and length defaults
  (`TEMPERATURE=1.0`, `TOP_P=1.0`, no top-k cap via `TOP_K=-1`,
  `MAX_PROMPT_LENGTH=4096`, `MAX_COMPLETION_LENGTH=4096`) while keeping local
  card and batch settings unchanged.
- `opsd_4b.sh` follows the OPSD non-thinking 4B release defaults where they
  are not resource/batch choices: `LEARNING_RATE=5e-6`,
  `MAX_GRAD_NORM=0.1`, `SAVE_STEPS=25`, `MAX_COMPLETION_LENGTH=1024`,
  `TEMPERATURE=1.1`, `TOP_P=0.95`, and `TOP_K=20`.

Append raw veRL/Hydra overrides with `VERL_EXTRA_ARGS`, for example:

```bash
VERL_EXTRA_ARGS="trainer.resume_mode=auto actor_rollout_ref.rollout.gpu_memory_utilization=0.85" \
sbatch verl/train_scripts/rlsd_4b.sh
```

W&B defaults to offline mode. Sync later from the output directory if needed.

Eval launchers live under `eval_scripts/` at the repository root, for example:

```bash
CHECKPOINT_DIR=/gpfs/share/home/2501210611/RLSD/outputs/cast_ema_4b_256(nogap005)/job_xxx/global_step_200/actor/lora_adapter \
sbatch eval_scripts/eval_32k_aime24_think.sh
```
