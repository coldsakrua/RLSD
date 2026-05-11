#!/usr/bin/env bash
# Build standardized DAPO math parquet (stem + fixed \boxed{} instruction) from the raw 17k file.
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_DIR}"

export PYTHONPATH="${PYTHONPATH:-}:${BASE_DIR}"
# Avoid writing HuggingFace dataset locks outside the project (HPC-friendly).
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${BASE_DIR}/outputs/hf_cache/datasets_preprocess}"
mkdir -p "${HF_DATASETS_CACHE}"

INPUT_PATH="${INPUT_PATH:-${BASE_DIR}/data/dapo/dapo-math-17k.parquet}"
OUTPUT_PATH="${OUTPUT_PATH:-${BASE_DIR}/data/dapo/dapo-math-17k-standard-boxed.parquet}"
NUM_PROC="${NUM_PROC:-1}"
DRY_RUN="${DRY_RUN:-0}"

EXTRA=()
if [[ "${DRY_RUN}" == "1" ]]; then
  EXTRA+=(--dry_run)
fi

python3 scripts/preprocess_dapo_math_prompts.py \
  --input_path "${INPUT_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --num_proc "${NUM_PROC}" \
  "${EXTRA[@]}"

echo "[hint] Point training at the new file, e.g.:"
echo "  export DATASET_PATH=${OUTPUT_PATH}"
