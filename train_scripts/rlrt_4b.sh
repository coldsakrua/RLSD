#!/bin/bash
#SBATCH -o logs/rlrt_4b.%j.out
#SBATCH -p GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=81920M
#SBATCH --time=72:00:00
#SBATCH --exclude=gpua800n15,gpua800n05,gpua800n04

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

MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b}
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/dapo/dapo-math-17k.parquet}
DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-${BASE_DIR}/outputs/hf_cache}
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/outputs/rlrt_4b}
RUN_CONFIG=${RUN_CONFIG:-rlrt_4b}
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
PROMPT_PREFIX=${PROMPT_PREFIX:-}
PROMPT_SUFFIX=${PROMPT_SUFFIX:-}
NORMALIZE_MATH_PROMPT_TO_STANDARD_SUFFIX=${NORMALIZE_MATH_PROMPT_TO_STANDARD_SUFFIX:-false}
MATH_INSTRUCTION_SUFFIX=${MATH_INSTRUCTION_SUFFIX:-}
USE_DAPO_RAW_PROMPT=${USE_DAPO_RAW_PROMPT:-true}

# Paper Table 5 (RLRT): lr=1e-6, weight_decay=0.01, warmup_steps=10, rollout T=1.0,
#   lambda_init=0.5, epsilon_w=1.0, no lambda decay, current-policy teacher.
# Where the paper is silent (e.g. LR scheduler), use repo defaults below.
LEARNING_RATE=${LEARNING_RATE:-1e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-}
WARMUP_STEPS=${WARMUP_STEPS:-10}
LR_END=${LR_END:-1e-7}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-linear}
if [ -z "${LR_SCHEDULER_KWARGS+x}" ] && [ "${LR_SCHEDULER_TYPE}" = "polynomial" ]; then
    LR_SCHEDULER_KWARGS="{\"lr_end\":${LR_END},\"power\":1.0}"
elif [ -z "${LR_SCHEDULER_KWARGS+x}" ]; then
    LR_SCHEDULER_KWARGS=""
fi

TRAIN_LR_ARGS=(--learning_rate "${LEARNING_RATE}" --weight_decay "${WEIGHT_DECAY}" --lr_scheduler_type "${LR_SCHEDULER_TYPE}")
if [ "${WARMUP_STEPS:-0}" != "0" ]; then
    TRAIN_LR_ARGS+=(--warmup_steps "${WARMUP_STEPS}")
elif [ -n "${WARMUP_RATIO}" ] && [ "${WARMUP_RATIO}" != "0" ]; then
    TRAIN_LR_ARGS+=(--warmup_ratio "${WARMUP_RATIO}")
fi
if [ -n "${LR_SCHEDULER_KWARGS}" ]; then
    TRAIN_LR_ARGS+=(--lr_scheduler_kwargs "${LR_SCHEDULER_KWARGS}")
fi

# Effective warmup steps for logging only (HuggingFace: warmup_steps = floor(max_steps * warmup_ratio)).
if [ "${WARMUP_STEPS:-0}" != "0" ]; then
    _WU_DESC="warmup_steps=${WARMUP_STEPS}"
elif [ -n "${WARMUP_RATIO}" ] && [ "${WARMUP_RATIO}" != "0" ]; then
    _WU_STEPS=$(awk -v ms="${MAX_STEPS}" -v r="${WARMUP_RATIO}" 'BEGIN { printf "%d", int(ms * r) }')
    _WU_DESC="warmup_ratio=${WARMUP_RATIO} → ~${_WU_STEPS} optimizer steps (max_steps=${MAX_STEPS})"
else
    _WU_DESC="no warmup"
fi

NUM_GENERATIONS=${NUM_GENERATIONS:-8}
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.9}
# Training rollout: paper specifies temperature=1.0 only (eval uses T=0.7, top_p=0.8).
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:-20}
MIN_P=${MIN_P:-0.0}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
PRESENCE_PENALTY=${PRESENCE_PENALTY:-0.0}
if [ -z "${GENERATION_KWARGS+x}" ]; then
  if [ "${PRESENCE_PENALTY}" != "0" ] && [ "${PRESENCE_PENALTY}" != "0.0" ]; then
    GENERATION_KWARGS="{\"presence_penalty\":${PRESENCE_PENALTY}}"
  else
    GENERATION_KWARGS="{}"
  fi
fi
MASK_TRUNCATED_COMPLETIONS=${MASK_TRUNCATED_COMPLETIONS:-true}
TRAIN_CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES:-0}
GEN_CUDA_VISIBLE_DEVICES=${GEN_CUDA_VISIBLE_DEVICES:-1}
VLLM_SERVER_HOST=${VLLM_SERVER_HOST:-127.0.0.1}
VLLM_SERVER_PORT=${VLLM_SERVER_PORT:-8000}
VLLM_SERVER_BASE_URL=${VLLM_SERVER_BASE_URL:-http://${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT}}
VLLM_SERVER_TIMEOUT=${VLLM_SERVER_TIMEOUT:-300}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}

ROLLOUT_FILTER=${ROLLOUT_FILTER:-all}
TOKEN_GAP_LAMBDA=${TOKEN_GAP_LAMBDA:-0.5}
TOKEN_GAP_DECAY_STEPS=${TOKEN_GAP_DECAY_STEPS:-0}
RLRT_WEIGHT_CLIP=${RLRT_WEIGHT_CLIP:-1.0}
RLRT_DELTA=${RLRT_DELTA:-2.0}
RLRT_TEACHER_CONTEXT_MODE=${RLRT_TEACHER_CONTEXT_MODE:-successful_rollout}

ALL_CORRECT_BASE_ADVANTAGE=${ALL_CORRECT_BASE_ADVANTAGE:-1.0}
ALL_WRONG_BASE_ADVANTAGE=${ALL_WRONG_BASE_ADVANTAGE:--1.0}
CORRECT_WEIGHT_CLIP_LOW=${CORRECT_WEIGHT_CLIP_LOW:-0.8}
CORRECT_WEIGHT_CLIP_HIGH=${CORRECT_WEIGHT_CLIP_HIGH:-1.05}
WRONG_WEIGHT_CLIP_LOW=${WRONG_WEIGHT_CLIP_LOW:-0.95}
WRONG_WEIGHT_CLIP_HIGH=${WRONG_WEIGHT_CLIP_HIGH:-1.2}
TEACHER_UPDATE_INTERVAL_STEPS=${TEACHER_UPDATE_INTERVAL_STEPS:-0}
TEACHER_INCLUDE_REFERENCE_SOLUTION=${TEACHER_INCLUDE_REFERENCE_SOLUTION:-false}
ADV_CLIP_LOW=${ADV_CLIP_LOW:--1000000000.0}
ADV_CLIP_HIGH=${ADV_CLIP_HIGH:-1000000000.0}
SUPPRESS_GT_SHORTCUT=${SUPPRESS_GT_SHORTCUT:-true}
ANSWER_TOKEN_DOWNWEIGHT=${ANSWER_TOKEN_DOWNWEIGHT:-1.0}
REWARD_BINARY_THRESHOLD=${REWARD_BINARY_THRESHOLD:-0.5}
FALLBACK_TAIL_TOKENS=${FALLBACK_TAIL_TOKENS:-8}
REWARD_FORMAT_PENALTIES=${REWARD_FORMAT_PENALTIES:-false}
REWARD_NO_EOS_PENALTY=${REWARD_NO_EOS_PENALTY:-0.15}
REWARD_MULTI_BOXED_PENALTY=${REWARD_MULTI_BOXED_PENALTY:-0.15}
REWARD_MIN_CONSECUTIVE_BOXED=${REWARD_MIN_CONSECUTIVE_BOXED:-2}
REWARD_REPEAT_TRIPLET_PENALTY=${REWARD_REPEAT_TRIPLET_PENALTY:-0.0}
REWARD_REPEAT_TRIPLET_LEV_THRESHOLD=${REWARD_REPEAT_TRIPLET_LEV_THRESHOLD:-0}
DISABLE_THINKING_IN_CHAT_TEMPLATE=${DISABLE_THINKING_IN_CHAT_TEMPLATE:-true}
REWARD_BOXED_LAST_TOKEN_FRACTION=${REWARD_BOXED_LAST_TOKEN_FRACTION:-0.05}
DAPO_EPSILON=${DAPO_EPSILON:-0.2}
DAPO_EPSILON_HIGH=${DAPO_EPSILON_HIGH:-0.28}

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

echo "[rlrt] lambda=${TOKEN_GAP_LAMBDA} decay_steps=${TOKEN_GAP_DECAY_STEPS} epsilon_w=${RLRT_WEIGHT_CLIP} delta=${RLRT_DELTA}"
echo "[rlrt] teacher_context_mode=${RLRT_TEACHER_CONTEXT_MODE} teacher_include_reference_solution=${TEACHER_INCLUDE_REFERENCE_SOLUTION}"
echo "[launch] vLLM server on GPU ${GEN_CUDA_VISIBLE_DEVICES}: ${VLLM_SERVER_BASE_URL}"
CUDA_VISIBLE_DEVICES="${GEN_CUDA_VISIBLE_DEVICES}" \
PYTORCH_CUDA_ALLOC_CONF="" \
trl vllm-serve \
    --model "${MODEL_PATH}" \
    --host "${VLLM_SERVER_HOST}" \
    --port "${VLLM_SERVER_PORT}" \
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}" \
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}" \
    > "${VLLM_SERVER_LOG}" 2>&1 &
VLLM_SERVER_PID=$!

_MATH_SUFFIX_ARGS=()
if [ -n "${MATH_INSTRUCTION_SUFFIX}" ]; then
    _MATH_SUFFIX_ARGS+=(--math_instruction_suffix "${MATH_INSTRUCTION_SUFFIX}")
fi
_DAPO_EPS_HIGH_ARGS=()
if [ -n "${DAPO_EPSILON_HIGH}" ]; then
    _DAPO_EPS_HIGH_ARGS+=(--dapo_epsilon_high "${DAPO_EPSILON_HIGH}")
fi

echo "[launch] trainer on GPU ${TRAIN_CUDA_VISIBLE_DEVICES} lr=${LEARNING_RATE} weight_decay=${WEIGHT_DECAY} sched=${LR_SCHEDULER_TYPE} ${_WU_DESC}"
CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}" accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 1 \
    --gradient_accumulation_steps "${GRAD_ACC_STEPS}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    rlrt_train_anchor.py \
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
    --max_grad_norm 1.0 \
    --per_device_train_batch_size "${PER_DEVICE_BS}" \
    --gradient_accumulation_steps "${GRAD_ACC_STEPS}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_config "${RUN_CONFIG}" \
    --max_steps "${MAX_STEPS}" \
    --num_generations "${NUM_GENERATIONS}" \
    --max_completion_length "${MAX_COMPLETION_LENGTH}" \
    --save_steps 50 \
    --logging_steps 1 \
    --attn_implementation sdpa \
    --torch_dtype bfloat16 \
    --max_length "${MAX_LENGTH}" \
    --beta 0 \
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
    --mask_truncated_completions "${MASK_TRUNCATED_COMPLETIONS}" \
    --token_gap_lambda "${TOKEN_GAP_LAMBDA}" \
    --token_gap_decay_steps "${TOKEN_GAP_DECAY_STEPS}" \
    --rlrt_weight_clip "${RLRT_WEIGHT_CLIP}" \
    --rlrt_delta "${RLRT_DELTA}" \
    --rlrt_teacher_context_mode "${RLRT_TEACHER_CONTEXT_MODE}" \
    --fixed_teacher false \
    --teacher_update_interval_steps "${TEACHER_UPDATE_INTERVAL_STEPS}" \
    --teacher_include_reference_solution "${TEACHER_INCLUDE_REFERENCE_SOLUTION}" \
    --rollout_filter "${ROLLOUT_FILTER}" \
    --all_correct_base_advantage "${ALL_CORRECT_BASE_ADVANTAGE}" \
    --all_wrong_base_advantage "${ALL_WRONG_BASE_ADVANTAGE}" \
    --correct_weight_clip_low "${CORRECT_WEIGHT_CLIP_LOW}" \
    --correct_weight_clip_high "${CORRECT_WEIGHT_CLIP_HIGH}" \
    --wrong_weight_clip_low "${WRONG_WEIGHT_CLIP_LOW}" \
    --wrong_weight_clip_high "${WRONG_WEIGHT_CLIP_HIGH}" \
    --adv_clip_low "${ADV_CLIP_LOW}" \
    --adv_clip_high "${ADV_CLIP_HIGH}" \
    --suppress_gt_shortcut "${SUPPRESS_GT_SHORTCUT}" \
    --answer_token_downweight "${ANSWER_TOKEN_DOWNWEIGHT}" \
    --reward_binary_threshold "${REWARD_BINARY_THRESHOLD}" \
    --fallback_tail_tokens "${FALLBACK_TAIL_TOKENS}" \
    --reward_format_penalties "${REWARD_FORMAT_PENALTIES}" \
    --reward_no_eos_penalty "${REWARD_NO_EOS_PENALTY}" \
    --reward_multi_boxed_penalty "${REWARD_MULTI_BOXED_PENALTY}" \
    --reward_min_consecutive_boxed "${REWARD_MIN_CONSECUTIVE_BOXED}" \
    --reward_repeat_triplet_penalty "${REWARD_REPEAT_TRIPLET_PENALTY}" \
    --reward_repeat_triplet_levenshtein_threshold "${REWARD_REPEAT_TRIPLET_LEV_THRESHOLD}" \
    --disable_thinking_in_chat_template "${DISABLE_THINKING_IN_CHAT_TEMPLATE}" \
    --reward_boxed_last_token_fraction "${REWARD_BOXED_LAST_TOKEN_FRACTION}" \
    --epsilon "${DAPO_EPSILON}" \
    "${_DAPO_EPS_HIGH_ARGS[@]}" \
    --disable_wandb "${DISABLE_WANDB}" \
    --gradient_checkpointing
