#!/bin/bash
#SBATCH -o logs/deepseek_logs/rlsd_deepseek_math_7b_rl_strict_split_flip_wrong_boost_nodecay_no_teacher_ref_300.%j.out
#SBATCH -p GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=81920M
#SBATCH --time=72:00:00
#SBATCH --exclude=gpua800n26,gpua800n25
#
# Two-phase schedule (1200 optimizer steps total):
#   Phase 1 (steps 0..299):   RLSD flip_wrong_boost nodecay_no_teacher_ref
#   Phase 2 (steps 300..1199): pure GRPO via opsd_train_grpo_strict.py, resume from checkpoint-300

set -eo pipefail
nvidia-smi

BASE_DIR="/gpfs/share/home/2501210611/RLSD"
cd "${BASE_DIR}"
mkdir -p logs/deepseek_logs

source activate anchor
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset ROCR_VISIBLE_DEVICES

MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/deepseek-math-7b-rl}
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/gsm8k}
DATASET_SPLIT=${DATASET_SPLIT:-train}
DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-${BASE_DIR}/outputs/hf_cache}
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/outputs/rlsd_deepseek_math_7b_rl_strict_split_flip_wrong_boost_nodecay_no_teacher_ref_300}
RUN_CONFIG=${RUN_CONFIG:-rlsd_deepseek_math_7b_rl_strict_split_flip_wrong_boost_nodecay_no_teacher_ref_300}
JOB_TAG="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR}/job_${JOB_TAG}"
mkdir -p "${OUTPUT_DIR}"

DISABLE_WANDB="${DISABLE_WANDB:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${OUTPUT_DIR}"
export WANDB_DATA_DIR="${OUTPUT_DIR}/.wandb_data"
mkdir -p "${WANDB_DATA_DIR}"

MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-12950}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-8}
PER_DEVICE_BS=${PER_DEVICE_BS:-8}
MAX_STEPS=${MAX_STEPS:-1200}
RLSD_PHASE_STEPS=${RLSD_PHASE_STEPS:-300}
GRPO_PHASE_STEPS=${GRPO_PHASE_STEPS:-$((MAX_STEPS - RLSD_PHASE_STEPS))}
# DeepSeek-Math context: prompt + completion <= 4096 (https://github.com/deepseek-ai/DeepSeek-Math)
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-256}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-$((MODEL_MAX_LENGTH - MAX_PROMPT_LENGTH))}
MAX_LENGTH=${MAX_LENGTH:-${MODEL_MAX_LENGTH}}
PROMPT_PREFIX=${PROMPT_PREFIX:-}
PROMPT_SUFFIX=${PROMPT_SUFFIX:-}
NORMALIZE_MATH_PROMPT_TO_STANDARD_SUFFIX=${NORMALIZE_MATH_PROMPT_TO_STANDARD_SUFFIX:-false}
MATH_INSTRUCTION_SUFFIX=${MATH_INSTRUCTION_SUFFIX:-}
USE_DAPO_RAW_PROMPT=${USE_DAPO_RAW_PROMPT:-false}

LEARNING_RATE=${LEARNING_RATE:-1e-6}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
WARMUP_STEPS=${WARMUP_STEPS:-0}
LR_END=${LR_END:-1e-7}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-polynomial}
if [ -z "${LR_SCHEDULER_KWARGS+x}" ]; then
    LR_SCHEDULER_KWARGS="{\"lr_end\":${LR_END},\"power\":1.0}"
fi

# LR schedule is defined for the full ${MAX_STEPS}-step run; phase2 resumes optimizer/scheduler state.
_SCHEDULE_WARMUP_STEPS="${WARMUP_STEPS}"
if [ "${_SCHEDULE_WARMUP_STEPS:-0}" = "0" ] && [ -n "${WARMUP_RATIO}" ] && [ "${WARMUP_RATIO}" != "0" ]; then
    _SCHEDULE_WARMUP_STEPS=$(awk -v ms="${MAX_STEPS}" -v r="${WARMUP_RATIO}" 'BEGIN { printf "%d", int(ms * r) }')
fi

TRAIN_LR_ARGS=(--learning_rate "${LEARNING_RATE}" --lr_scheduler_type "${LR_SCHEDULER_TYPE}")
if [ "${_SCHEDULE_WARMUP_STEPS:-0}" != "0" ]; then
    TRAIN_LR_ARGS+=(--warmup_steps "${_SCHEDULE_WARMUP_STEPS}")
fi
if [ -n "${LR_SCHEDULER_KWARGS}" ]; then
    TRAIN_LR_ARGS+=(--lr_scheduler_kwargs "${LR_SCHEDULER_KWARGS}")
fi

if [ "${_SCHEDULE_WARMUP_STEPS:-0}" != "0" ]; then
    _WU_DESC="warmup_steps=${_SCHEDULE_WARMUP_STEPS} (of max_steps=${MAX_STEPS})"
else
    _WU_DESC="no warmup"
fi

NUM_GENERATIONS=${NUM_GENERATIONS:-8}
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.9}
RLSD_TEMPERATURE=${RLSD_TEMPERATURE:-0.6}
RLSD_TOP_P=${RLSD_TOP_P:-0.9}
GRPO_TEMPERATURE=${GRPO_TEMPERATURE:-0.7}
GRPO_TOP_P=${GRPO_TOP_P:-0.95}
TOP_K=${TOP_K:-20}
MIN_P=${MIN_P:-0.0}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
PRESENCE_PENALTY=${PRESENCE_PENALTY:-0.2}
if [ -z "${GENERATION_KWARGS+x}" ]; then
    GENERATION_KWARGS="{\"presence_penalty\":${PRESENCE_PENALTY}}"
fi
MASK_TRUNCATED_COMPLETIONS=${MASK_TRUNCATED_COMPLETIONS:-true}
TRAIN_CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES:-0}
GEN_CUDA_VISIBLE_DEVICES=${GEN_CUDA_VISIBLE_DEVICES:-1}
VLLM_SERVER_HOST=${VLLM_SERVER_HOST:-127.0.0.1}
VLLM_SERVER_PORT=${VLLM_SERVER_PORT:-8000}
VLLM_SERVER_BASE_URL=${VLLM_SERVER_BASE_URL:-http://${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT}}
VLLM_SERVER_TIMEOUT=${VLLM_SERVER_TIMEOUT:-300}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-${MAX_LENGTH}}

ROLLOUT_FILTER=${ROLLOUT_FILTER:-all}
TOKEN_GAP_LAMBDA=${TOKEN_GAP_LAMBDA:-1.0}
TOKEN_GAP_DECAY_STEPS=${TOKEN_GAP_DECAY_STEPS:-0}

ALL_CORRECT_BASE_ADVANTAGE=${ALL_CORRECT_BASE_ADVANTAGE:-1.0}
ALL_WRONG_BASE_ADVANTAGE=${ALL_WRONG_BASE_ADVANTAGE:--1.0}
CORRECT_WEIGHT_CLIP_LOW=${CORRECT_WEIGHT_CLIP_LOW:-0.8}
CORRECT_WEIGHT_CLIP_HIGH=${CORRECT_WEIGHT_CLIP_HIGH:-1.05}
WRONG_WEIGHT_CLIP_LOW=${WRONG_WEIGHT_CLIP_LOW:-0.95}
WRONG_WEIGHT_CLIP_HIGH=${WRONG_WEIGHT_CLIP_HIGH:-1.2}
TEACHER_UPDATE_INTERVAL_STEPS=${TEACHER_UPDATE_INTERVAL_STEPS:-10}
TEACHER_INCLUDE_REFERENCE_SOLUTION=${TEACHER_INCLUDE_REFERENCE_SOLUTION:-false}
ADV_CLIP_LOW=${ADV_CLIP_LOW:--1.2}
ADV_CLIP_HIGH=${ADV_CLIP_HIGH:-1.2}
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
RELAXED_ANSWER_EXTRACTION=${RELAXED_ANSWER_EXTRACTION:-true}
REWARD_BOXED_LAST_TOKEN_FRACTION=${REWARD_BOXED_LAST_TOKEN_FRACTION:-0.05}
DAPO_EPSILON=${DAPO_EPSILON:-0.2}
DAPO_EPSILON_HIGH=${DAPO_EPSILON_HIGH:-0.28}

LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"}
LORA_R=${LORA_R:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
STRICT_LORA_ONLY=${STRICT_LORA_ONLY:-true}

SKIP_PHASE1=${SKIP_PHASE1:-false}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}

if [ "${TRAIN_CUDA_VISIBLE_DEVICES}" = "${GEN_CUDA_VISIBLE_DEVICES}" ]; then
    echo "[error] TRAIN_CUDA_VISIBLE_DEVICES and GEN_CUDA_VISIBLE_DEVICES must be different."
    exit 1
fi

if [ "${GRPO_PHASE_STEPS}" != "$((MAX_STEPS - RLSD_PHASE_STEPS))" ]; then
    echo "[warn] GRPO_PHASE_STEPS=${GRPO_PHASE_STEPS} != MAX_STEPS-RLSD_PHASE_STEPS=$((MAX_STEPS - RLSD_PHASE_STEPS))"
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

VLLM_SERVE_ARGS=(
    --model "${MODEL_PATH}"
    --host "${VLLM_SERVER_HOST}"
    --port "${VLLM_SERVER_PORT}"
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}"
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}"
)
if [ -n "${VLLM_MAX_MODEL_LEN}" ] && [ "${VLLM_MAX_MODEL_LEN}" != "0" ]; then
    VLLM_SERVE_ARGS+=(--max-model-len "${VLLM_MAX_MODEL_LEN}")
fi

_MATH_SUFFIX_ARGS=()
if [ -n "${MATH_INSTRUCTION_SUFFIX}" ]; then
    _MATH_SUFFIX_ARGS+=(--math_instruction_suffix "${MATH_INSTRUCTION_SUFFIX}")
fi

echo "[schedule] phase1 RLSD steps 0..$((RLSD_PHASE_STEPS - 1)) (${RLSD_PHASE_STEPS} steps)"
echo "[schedule] phase2 pure GRPO steps ${RLSD_PHASE_STEPS}..$((MAX_STEPS - 1)) (${GRPO_PHASE_STEPS} steps)"
echo "[launch] context budget: max_length=${MAX_LENGTH} (prompt<=${MAX_PROMPT_LENGTH}, completion<=${MAX_COMPLETION_LENGTH})"
echo "[launch] vLLM server on GPU ${GEN_CUDA_VISIBLE_DEVICES}: ${VLLM_SERVER_BASE_URL}"
CUDA_VISIBLE_DEVICES="${GEN_CUDA_VISIBLE_DEVICES}" \
PYTORCH_CUDA_ALLOC_CONF="" \
trl vllm-serve "${VLLM_SERVE_ARGS[@]}" \
    > "${VLLM_SERVER_LOG}" 2>&1 &
VLLM_SERVER_PID=$!

if [ "${SKIP_PHASE1}" != "true" ]; then
    echo "[phase1] RLSD flip_wrong_boost nodecay_no_teacher_ref, max_steps=${RLSD_PHASE_STEPS}"
    echo "[phase1] wrong_path_positive_flip=true, teacher_include_reference_solution=${TEACHER_INCLUDE_REFERENCE_SOLUTION}"
    echo "[phase1] trainer on GPU ${TRAIN_CUDA_VISIBLE_DEVICES} lr=${LEARNING_RATE} sched=${LR_SCHEDULER_TYPE} ${_WU_DESC}"
    CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}" accelerate launch \
        --config_file accelerate.yaml \
        --num_processes 1 \
        --gradient_accumulation_steps "${GRAD_ACC_STEPS}" \
        --main_process_port "${MAIN_PROCESS_PORT}" \
        opsd_train_anchor_strict_split_flip_wrong_boost.py \
        --model_name_or_path "${MODEL_PATH}" \
        --dataset_path "${DATASET_PATH}" \
        --dataset_split "${DATASET_SPLIT}" \
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
        --max_steps "${RLSD_PHASE_STEPS}" \
        --num_generations "${NUM_GENERATIONS}" \
        --max_completion_length "${MAX_COMPLETION_LENGTH}" \
        --save_steps "${RLSD_PHASE_STEPS}" \
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
        --temperature "${RLSD_TEMPERATURE}" \
        --top_p "${RLSD_TOP_P}" \
        --top_k "${TOP_K}" \
        --min_p "${MIN_P}" \
        --repetition_penalty "${REPETITION_PENALTY}" \
        --generation_extra_kwargs_json "${GENERATION_KWARGS}" \
        --mask_truncated_completions "${MASK_TRUNCATED_COMPLETIONS}" \
        --token_gap_lambda "${TOKEN_GAP_LAMBDA}" \
        --token_gap_decay_steps "${TOKEN_GAP_DECAY_STEPS}" \
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
        --relaxed_answer_extraction "${RELAXED_ANSWER_EXTRACTION}" \
        --reward_boxed_last_token_fraction "${REWARD_BOXED_LAST_TOKEN_FRACTION}" \
        --epsilon "${DAPO_EPSILON}" \
        --dapo_epsilon_high "${DAPO_EPSILON_HIGH}" \
        --disable_wandb "${DISABLE_WANDB}" \
        --gradient_checkpointing
else
    echo "[phase1] skipped (SKIP_PHASE1=true)"
fi

if [ -n "${RESUME_CHECKPOINT}" ]; then
    PHASE1_CKPT="${RESUME_CHECKPOINT}"
else
    PHASE1_CKPT="${OUTPUT_DIR}/checkpoint-${RLSD_PHASE_STEPS}"
fi

if [ ! -d "${PHASE1_CKPT}" ]; then
    echo "[error] phase1 checkpoint not found: ${PHASE1_CKPT}"
    exit 1
fi
echo "[phase2] resume_from_checkpoint=${PHASE1_CKPT}"

echo "[phase2] pure GRPO (opsd_train_grpo_strict.py), max_steps=${MAX_STEPS} (continues from step ${RLSD_PHASE_STEPS})"
echo "[phase2] trainer on GPU ${TRAIN_CUDA_VISIBLE_DEVICES} temperature=${GRPO_TEMPERATURE} top_p=${GRPO_TOP_P}"
CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}" accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 1 \
    --gradient_accumulation_steps "${GRAD_ACC_STEPS}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    opsd_train_grpo_strict.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --dataset_split "${DATASET_SPLIT}" \
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
    --run_config "${RUN_CONFIG}_grpo_phase" \
    --max_steps "${MAX_STEPS}" \
    --resume_from_checkpoint "${PHASE1_CKPT}" \
    --num_generations "${NUM_GENERATIONS}" \
    --max_completion_length "${MAX_COMPLETION_LENGTH}" \
    --save_steps 100 \
    --logging_steps 1 \
    --attn_implementation sdpa \
    --torch_dtype bfloat16 \
    --max_length "${MAX_LENGTH}" \
    --beta 0 \
    --epsilon "${DAPO_EPSILON}" \
    --epsilon_high "${DAPO_EPSILON_HIGH}" \
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
    --temperature "${GRPO_TEMPERATURE}" \
    --top_p "${GRPO_TOP_P}" \
    --top_k "${TOP_K}" \
    --min_p "${MIN_P}" \
    --repetition_penalty "${REPETITION_PENALTY}" \
    --generation_extra_kwargs_json "${GENERATION_KWARGS}" \
    --mask_truncated_completions "${MASK_TRUNCATED_COMPLETIONS}" \
    --reward_format_penalties "${REWARD_FORMAT_PENALTIES}" \
    --reward_no_eos_penalty "${REWARD_NO_EOS_PENALTY}" \
    --reward_multi_boxed_penalty "${REWARD_MULTI_BOXED_PENALTY}" \
    --reward_min_consecutive_boxed "${REWARD_MIN_CONSECUTIVE_BOXED}" \
    --reward_repeat_triplet_penalty "${REWARD_REPEAT_TRIPLET_PENALTY}" \
    --reward_repeat_triplet_levenshtein_threshold "${REWARD_REPEAT_TRIPLET_LEV_THRESHOLD}" \
    --disable_thinking_in_chat_template "${DISABLE_THINKING_IN_CHAT_TEMPLATE}" \
    --relaxed_answer_extraction "${RELAXED_ANSWER_EXTRACTION}" \
    --reward_boxed_last_token_fraction "${REWARD_BOXED_LAST_TOKEN_FRACTION}" \
    --reward_binary_threshold "${REWARD_BINARY_THRESHOLD}" \
    --disable_wandb "${DISABLE_WANDB}" \
    --gradient_checkpointing

echo "[done] final checkpoint expected at ${OUTPUT_DIR}/checkpoint-${MAX_STEPS}"
