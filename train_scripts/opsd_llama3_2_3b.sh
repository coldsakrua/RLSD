#!/bin/bash
#SBATCH -o logs/opsd_llama3_2_3b.%j.out
#SBATCH -p GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=81920M
#SBATCH --time=72:00:00
#SBATCH --exclude=gpua800n26,gpua800n04,gpua800n11,gpua800n03
set -eo pipefail
nvidia-smi

BASE_DIR="/gpfs/share/home/2501210611/RLSD"
cd "${BASE_DIR}"
mkdir -p logs

source activate anchor
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset ROCR_VISIBLE_DEVICES

MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/llama-3.2-3b-instruct}
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/dapo/dapo-math-17k.parquet}
DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-${BASE_DIR}/outputs/hf_cache}
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/outputs/opsd_llama3_2_3b}
RUN_CONFIG=${RUN_CONFIG:-opsd_llama3_2_3b}
JOB_TAG="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR}/job_${JOB_TAG}"
mkdir -p "${OUTPUT_DIR}"

DISABLE_WANDB="${DISABLE_WANDB:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${OUTPUT_DIR}"
export WANDB_DATA_DIR="${OUTPUT_DIR}/.wandb_data"
mkdir -p "${WANDB_DATA_DIR}"

MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-12949}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-8}
PER_DEVICE_BS=${PER_DEVICE_BS:-2}
MAX_STEPS=${MAX_STEPS:-300}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-3072}
# Keep enough prompt budget: trainer computes max_prompt_length = max_length - max_completion_length.
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_LENGTH=$((MAX_COMPLETION_LENGTH + MAX_PROMPT_LENGTH))
MAX_TEACHER_PROMPT_LENGTH=${MAX_TEACHER_PROMPT_LENGTH:-${MAX_PROMPT_LENGTH}}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
PROMPT_PREFIX=${PROMPT_PREFIX:-}
PROMPT_SUFFIX=${PROMPT_SUFFIX:-}
NORMALIZE_MATH_PROMPT_TO_STANDARD_SUFFIX=${NORMALIZE_MATH_PROMPT_TO_STANDARD_SUFFIX:-false}
MATH_INSTRUCTION_SUFFIX=${MATH_INSTRUCTION_SUFFIX:-}
USE_DAPO_RAW_PROMPT=${USE_DAPO_RAW_PROMPT:-true}

# OPSD paper/repo 4b non-thinking defaults, while keeping this repo's length budget.
LEARNING_RATE=${LEARNING_RATE:-5e-6}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
WARMUP_STEPS=${WARMUP_STEPS:-0}
LR_END=${LR_END:-0}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-polynomial}
if [ -z "${LR_SCHEDULER_KWARGS+x}" ]; then
    LR_SCHEDULER_KWARGS="{\"lr_end\":${LR_END},\"power\":1.0}"
fi

TRAIN_LR_ARGS=(--learning_rate "${LEARNING_RATE}" --lr_scheduler_type "${LR_SCHEDULER_TYPE}")
if [ "${WARMUP_STEPS:-0}" != "0" ]; then
    TRAIN_LR_ARGS+=(--warmup_steps "${WARMUP_STEPS}")
elif [ -n "${WARMUP_RATIO}" ] && [ "${WARMUP_RATIO}" != "0" ]; then
    TRAIN_LR_ARGS+=(--warmup_ratio "${WARMUP_RATIO}")
fi
if [ -n "${LR_SCHEDULER_KWARGS}" ]; then
    TRAIN_LR_ARGS+=(--lr_scheduler_kwargs "${LR_SCHEDULER_KWARGS}")
fi

if [ "${WARMUP_STEPS:-0}" != "0" ]; then
    _WU_DESC="warmup_steps=${WARMUP_STEPS}"
elif [ -n "${WARMUP_RATIO}" ] && [ "${WARMUP_RATIO}" != "0" ]; then
    _WU_STEPS=$(awk -v ms="${MAX_STEPS}" -v r="${WARMUP_RATIO}" 'BEGIN { printf "%d", int(ms * r) }')
    _WU_DESC="warmup_ratio=${WARMUP_RATIO} -> ~${_WU_STEPS} optimizer steps (max_steps=${MAX_STEPS})"
else
    _WU_DESC="no warmup"
fi

VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.9}
TEMPERATURE=${TEMPERATURE:-1.1}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}
MIN_P=${MIN_P:-0.0}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
PRESENCE_PENALTY=${PRESENCE_PENALTY:-0.0}
if [ -z "${GENERATION_KWARGS+x}" ]; then
    GENERATION_KWARGS="{\"presence_penalty\":${PRESENCE_PENALTY}}"
fi
TRAIN_CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES:-0}
GEN_CUDA_VISIBLE_DEVICES=${GEN_CUDA_VISIBLE_DEVICES:-1}
VLLM_SERVER_HOST=${VLLM_SERVER_HOST:-127.0.0.1}
VLLM_SERVER_PORT=${VLLM_SERVER_PORT:-8000}
VLLM_SERVER_BASE_URL=${VLLM_SERVER_BASE_URL:-http://${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT}}
VLLM_SERVER_TIMEOUT=${VLLM_SERVER_TIMEOUT:-300}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-${MAX_LENGTH}}
VLLM_SYNC_FREQUENCY=${VLLM_SYNC_FREQUENCY:-1}
SAVE_GENERATION_STEPS=${SAVE_GENERATION_STEPS:-0}

OPSD_LAMBDA=${OPSD_LAMBDA:-1.0}
OPSD_BETA=${OPSD_BETA:-0}
JSD_TOKEN_CLIP=${JSD_TOKEN_CLIP:-1e-7}
TOP_K_LOSS=${TOP_K_LOSS:-0}
FIXED_TEACHER=${FIXED_TEACHER:-true}
STUDENT_ENABLE_THINKING=${STUDENT_ENABLE_THINKING:-false}
TEACHER_ENABLE_THINKING=${TEACHER_ENABLE_THINKING:-false}
STUDENT_PROMPT_AS_CHAT=${STUDENT_PROMPT_AS_CHAT:-false}
DISABLE_THINKING_IN_CHAT_TEMPLATE=${DISABLE_THINKING_IN_CHAT_TEMPLATE:-false}
REWARD_BOXED_LAST_TOKEN_FRACTION=${REWARD_BOXED_LAST_TOKEN_FRACTION:-0.05}

LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"}
LORA_R=${LORA_R:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
STRICT_LORA_ONLY=${STRICT_LORA_ONLY:-true}

if [ "${TRAIN_CUDA_VISIBLE_DEVICES}" = "${GEN_CUDA_VISIBLE_DEVICES}" ]; then
    echo "[error] TRAIN_CUDA_VISIBLE_DEVICES and GEN_CUDA_VISIBLE_DEVICES must be different."
    exit 1
fi

VLLM_SERVER_LOG="${OUTPUT_DIR}/vllm_server.log"
VLLM_SERVER_PID=""
cleanup() {
    if [ -n "${VLLM_SERVER_PID}" ] && kill -0 "${VLLM_SERVER_PID}" 2>/dev/null; then
        kill "${VLLM_SERVER_PID}" || true
        wait "${VLLM_SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[opsd] fixed_teacher=${FIXED_TEACHER} jsd_token_clip=${JSD_TOKEN_CLIP} beta=${OPSD_BETA} top_k_loss=${TOP_K_LOSS}"
echo "[launch] vLLM server on GPU ${GEN_CUDA_VISIBLE_DEVICES}: ${VLLM_SERVER_BASE_URL}"
echo "[launch] VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN}"
CUDA_VISIBLE_DEVICES="${GEN_CUDA_VISIBLE_DEVICES}" \
PYTORCH_CUDA_ALLOC_CONF="" \
trl vllm-serve \
    --model "${MODEL_PATH}" \
    --host "${VLLM_SERVER_HOST}" \
    --port "${VLLM_SERVER_PORT}" \
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}" \
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    > "${VLLM_SERVER_LOG}" 2>&1 &
VLLM_SERVER_PID=$!

_MATH_SUFFIX_ARGS=()
if [ -n "${MATH_INSTRUCTION_SUFFIX}" ]; then
    _MATH_SUFFIX_ARGS+=(--math_instruction_suffix "${MATH_INSTRUCTION_SUFFIX}")
fi

echo "[launch] trainer on GPU ${TRAIN_CUDA_VISIBLE_DEVICES} lr=${LEARNING_RATE} sched=${LR_SCHEDULER_TYPE} ${_WU_DESC}"
CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}" accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 1 \
    --gradient_accumulation_steps "${GRAD_ACC_STEPS}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    official_opsd_train.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --dataset_split train \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --prompt_prefix "${PROMPT_PREFIX}" \
    --prompt_suffix "${PROMPT_SUFFIX}" \
    --normalize_math_prompt_to_standard_suffix "${NORMALIZE_MATH_PROMPT_TO_STANDARD_SUFFIX}" \
    --use_dapo_raw_prompt "${USE_DAPO_RAW_PROMPT}" \
    "${_MATH_SUFFIX_ARGS[@]}" \
    "${TRAIN_LR_ARGS[@]}" \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size "${PER_DEVICE_BS}" \
    --gradient_accumulation_steps "${GRAD_ACC_STEPS}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_config "${RUN_CONFIG}" \
    --max_steps "${MAX_STEPS}" \
    --num_generations 1 \
    --max_completion_length "${MAX_COMPLETION_LENGTH}" \
    --save_steps 50 \
    --logging_steps 1 \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --torch_dtype bfloat16 \
    --max_length "${MAX_LENGTH}" \
    --max_teacher_prompt_length "${MAX_TEACHER_PROMPT_LENGTH}" \
    --beta "${OPSD_BETA}" \
    --use_vllm \
    --vllm_mode server \
    --vllm_server_base_url "${VLLM_SERVER_BASE_URL}" \
    --vllm_server_timeout "${VLLM_SERVER_TIMEOUT}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEM_UTIL}" \
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
    --use_peft true \
    --strict_lora_only "${STRICT_LORA_ONLY}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_target_modules "${LORA_TARGET_MODULES}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --min_p "${MIN_P}" \
    --repetition_penalty "${REPETITION_PENALTY}" \
    --generation_extra_kwargs_json "${GENERATION_KWARGS}" \
    --lmbda "${OPSD_LAMBDA}" \
    --jsd_token_clip "${JSD_TOKEN_CLIP}" \
    --top_k_loss "${TOP_K_LOSS}" \
    --fixed_teacher "${FIXED_TEACHER}" \
    --student_enable_thinking "${STUDENT_ENABLE_THINKING}" \
    --teacher_enable_thinking "${TEACHER_ENABLE_THINKING}" \
    --student_prompt_as_chat "${STUDENT_PROMPT_AS_CHAT}" \
    --disable_thinking_in_chat_template "${DISABLE_THINKING_IN_CHAT_TEMPLATE}" \
    --reward_boxed_last_token_fraction "${REWARD_BOXED_LAST_TOKEN_FRACTION}" \
    --vllm_sync_frequency "${VLLM_SYNC_FREQUENCY}" \
    --save_generation_steps "${SAVE_GENERATION_STEPS}" \
    --disable_wandb "${DISABLE_WANDB}" \
    --gradient_checkpointing
