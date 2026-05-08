# RLSD Minimal Implementation

This folder contains a compact implementation of **Self-Distilled RLVR (RLSD)** using `trl==0.22.1`:

- `opsd_train_anchor.py`: training entry.
- `rlsd_trainer.py`: `GRPOTrainer` subclass that injects token-level RLSD credit.
- `reward_fn.py`: verifiable math reward (`math_verify` + fallback string match).
- `data_utils.py`: dataset loading with automatic prompt/solution column inference.
- `train_rlsd_qwen3_4b_1gpu.sh`: 1-GPU `sbatch` launch script.
- `accelerate.yaml`: single-process accelerate config.

## Dataset format

Input file can be `jsonl/json/parquet` with columns containing:

- prompt-like key: `prompt` / `problem` / `question` / `query` / `input` / `instruction`
- solution-like key: `solution` / `answer` / `ground_truth` / `target` / `reference`

For `data/aggregated_l3plus/train.parquet`, loader has an explicit schema adapter:

- reads `problem -> prompt`, `solution -> solution`
- preserves metadata as `problem_level`, `problem_type`, `problem_subject`
- filters empty prompt/solution rows

## Quick launch

```bash
sbatch train_rlsd_qwen3_4b_1gpu.sh
```
