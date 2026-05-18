#!/bin/bash
#SBATCH -o logs/analyze_opsd_token_shift.%j.out
#SBATCH -p GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=65536M
#SBATCH --time=24:00:00

set -eo pipefail
nvidia-smi

BASE_DIR="/gpfs/share/home/2501210611/RLSD"
cd "${BASE_DIR}"
mkdir -p logs

source activate anchor
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b}
# Leave empty to run base model (no LoRA adapter loaded).
LORA_PATH=${LORA_PATH:-}

# Default: DAPO dataset.
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/dapo/dapo-math-17k.parquet}
DATASET_SPLIT=${DATASET_SPLIT:-train}
DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-${BASE_DIR}/outputs/hf_cache}

OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/outputs/opsd_token_shift}
RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
OUTPUT_JSON=${OUTPUT_JSON:-${OUTPUT_DIR}/opsd_token_shift_${RUN_TAG}.json}
mkdir -p "${OUTPUT_DIR}"

SAMPLE_SIZE=${SAMPLE_SIZE:-256}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
SEED=${SEED:-42}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_TEACHER_PROMPT_LENGTH=${MAX_TEACHER_PROMPT_LENGTH:-3072}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}
MIN_P=${MIN_P:-0.0}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
PRESENCE_PENALTY=${PRESENCE_PENALTY:-0.2}

TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
DEVICE=${DEVICE:-auto}
ALLOW_CPU_FALLBACK=${ALLOW_CPU_FALLBACK:-false}
ATTN_IMPL=${ATTN_IMPL:-sdpa}

# Match strict-style default prompt handling (raw DAPO passthrough).
USE_DAPO_RAW_PROMPT=${USE_DAPO_RAW_PROMPT:-true}
NORMALIZE_MATH_PROMPT_TO_STANDARD_SUFFIX=${NORMALIZE_MATH_PROMPT_TO_STANDARD_SUFFIX:-false}
MATH_INSTRUCTION_SUFFIX=${MATH_INSTRUCTION_SUFFIX:-}
PROMPT_PREFIX=${PROMPT_PREFIX:-}
PROMPT_SUFFIX=${PROMPT_SUFFIX:-}

REWARD_BINARY_THRESHOLD=${REWARD_BINARY_THRESHOLD:-0.5}
REWARD_BOXED_LAST_TOKEN_FRACTION=${REWARD_BOXED_LAST_TOKEN_FRACTION:-0.05}
ENABLE_THINKING=${ENABLE_THINKING:-false}
SUMMARY_TOP_K=${SUMMARY_TOP_K:-30}

# Prompts per model.generate() call (left-padded batch).
BATCH_SIZE=${BATCH_SIZE:-4}

echo "[analyze] model=${MODEL_PATH}"
echo "[analyze] lora=${LORA_PATH:-<none>}"
echo "[analyze] dataset=${DATASET_PATH} split=${DATASET_SPLIT}"
echo "[analyze] output_json=${OUTPUT_JSON}"
echo "[analyze] batch_size=${BATCH_SIZE}"
echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-<unset>}"

python analyze_opsd_token_shift.py \
    --model_name_or_path "${MODEL_PATH}" \
    --lora_path "${LORA_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --dataset_split "${DATASET_SPLIT}" \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --output_json "${OUTPUT_JSON}" \
    --sample_size "${SAMPLE_SIZE}" \
    --num_generations "${NUM_GENERATIONS}" \
    --seed "${SEED}" \
    --max_prompt_length "${MAX_PROMPT_LENGTH}" \
    --max_teacher_prompt_length "${MAX_TEACHER_PROMPT_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --min_p "${MIN_P}" \
    --repetition_penalty "${REPETITION_PENALTY}" \
    --presence_penalty "${PRESENCE_PENALTY}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --device "${DEVICE}" \
    --allow_cpu_fallback "${ALLOW_CPU_FALLBACK}" \
    --attn_implementation "${ATTN_IMPL}" \
    --use_dapo_raw_prompt "${USE_DAPO_RAW_PROMPT}" \
    --normalize_math_prompt_to_standard_suffix "${NORMALIZE_MATH_PROMPT_TO_STANDARD_SUFFIX}" \
    --math_instruction_suffix "${MATH_INSTRUCTION_SUFFIX}" \
    --prompt_prefix "${PROMPT_PREFIX}" \
    --prompt_suffix "${PROMPT_SUFFIX}" \
    --reward_binary_threshold "${REWARD_BINARY_THRESHOLD}" \
    --reward_boxed_last_token_fraction "${REWARD_BOXED_LAST_TOKEN_FRACTION}" \
    --enable_thinking "${ENABLE_THINKING}" \
    --summary_top_k "${SUMMARY_TOP_K}" \
    --batch_size "${BATCH_SIZE}"
