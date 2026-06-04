#!/bin/bash
#SBATCH -o /gpfs/share/home/2501210611/RLSD/verl_logs/grpo_4b.%j.out
#SBATCH -p GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=81920M
#SBATCH --time=72:00:00

set -eo pipefail

# GRPO strict baseline: no teacher token shaping, no distillation.
RUN_CONFIG="${RUN_CONFIG:-grpo_4b}"
REWARD_FUNCTION_NAME="${REWARD_FUNCTION_NAME:-compute_score}"

MODEL_PATH="${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b}"
export MODEL_PATH
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"

NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
MIN_P="${MIN_P:-0.0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0.0}"
MASK_TRUNCATED_COMPLETIONS="${MASK_TRUNCATED_COMPLETIONS:-true}"

# Same batch policy as the 8-answer RLSD/RLRT scripts:
# 4 prompts * 8 rollouts = 32 responses per veRL train batch.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}"

BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/RLSD}}"
if [[ "$(basename "${BASE_DIR}")" == "verl" ]]; then
    BASE_DIR="$(cd "${BASE_DIR}/.." && pwd)"
fi
SCRIPT_DIR="${BASE_DIR}/verl"
cd "${BASE_DIR}"

CONDA_ENV="${CONDA_ENV:-anchor}"
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
else
    source activate "${CONDA_ENV}"
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${SCRIPT_DIR}:${BASE_DIR}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    unset PYTORCH_CUDA_ALLOC_CONF
fi
# Ray AF_UNIX sockets must stay under 107 bytes; avoid long GPFS paths.
_RAY_JOB_TAG="${SLURM_JOB_ID:-$$}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${_RAY_JOB_TAG}}"
export TMPDIR="${TMPDIR:-/tmp/rlsd_${_RAY_JOB_TAG}}"
mkdir -p "${RAY_TMPDIR}" "${TMPDIR}"
unset ROCR_VISIBLE_DEVICES

DATASET_PATH="${DATASET_PATH:-${BASE_DIR}/data/dapo/dapo-math-17k.parquet}"
VAL_DATASET_PATH="${VAL_DATASET_PATH:-${DATASET_PATH}}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-${BASE_DIR}/outputs/hf_cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE_DIR}/outputs/${RUN_CONFIG}}"
JOB_TAG="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/job_${JOB_TAG}}"
mkdir -p "${OUTPUT_DIR}" "${DATASET_CACHE_DIR}"

DISABLE_WANDB="${DISABLE_WANDB:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${OUTPUT_DIR}"
export WANDB_DATA_DIR="${OUTPUT_DIR}/.wandb_data"
export WANDB_CACHE_DIR="${OUTPUT_DIR}/.wandb_cache"
export WANDB_ARTIFACT_DIR="${OUTPUT_DIR}/wandb_artifacts"
export WANDB_PROJECT="${WANDB_PROJECT:-RLSD}"
export WANDB_NAME="${WANDB_NAME:-${RUN_CONFIG}_${JOB_TAG}}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${RUN_CONFIG}}"
mkdir -p "${WANDB_DATA_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_ARTIFACT_DIR}"

MAX_STEPS="${MAX_STEPS:-300}"
SAVE_STEPS="${SAVE_STEPS:-50}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-4096}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_LENGTH="$((MAX_COMPLETION_LENGTH + MAX_PROMPT_LENGTH))"

VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.9}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
LAYERED_SUMMON="${LAYERED_SUMMON:-False}"
ACTOR_GPUS_PER_NODE="${ACTOR_GPUS_PER_NODE:-2}"
AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-8}"

LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]}"
export RLSD_MERGE_LORA_FOR_ASYNC_VLLM="${RLSD_MERGE_LORA_FOR_ASYNC_VLLM:-false}"

DAPO_EPSILON="${DAPO_EPSILON:-0.2}"
DAPO_EPSILON_HIGH="${DAPO_EPSILON_HIGH:-0.28}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
REWARD_BINARY_THRESHOLD="${REWARD_BINARY_THRESHOLD:-0.5}"
PROMPT_PREFIX="${PROMPT_PREFIX:-}"
PROMPT_SUFFIX="${PROMPT_SUFFIX:-}"
NORMALIZE_MATH_PROMPT_TO_STANDARD_SUFFIX="${NORMALIZE_MATH_PROMPT_TO_STANDARD_SUFFIX:-false}"
MATH_INSTRUCTION_SUFFIX="${MATH_INSTRUCTION_SUFFIX:-}"
USE_DAPO_RAW_PROMPT="${USE_DAPO_RAW_PROMPT:-true}"

export REWARD_FORMAT_PENALTIES="${REWARD_FORMAT_PENALTIES:-false}"
export REWARD_NO_EOS_PENALTY="${REWARD_NO_EOS_PENALTY:-0.15}"
export REWARD_MULTI_BOXED_PENALTY="${REWARD_MULTI_BOXED_PENALTY:-0.15}"
export REWARD_MIN_CONSECUTIVE_BOXED="${REWARD_MIN_CONSECUTIVE_BOXED:-2}"
export REWARD_BOXED_LAST_TOKEN_FRACTION="${REWARD_BOXED_LAST_TOKEN_FRACTION:-0.05}"
export DISABLE_THINKING_IN_CHAT_TEMPLATE="${DISABLE_THINKING_IN_CHAT_TEMPLATE:-true}"
STRIP_DAPO_PROMPT_BOILERPLATE="${STRIP_DAPO_PROMPT_BOILERPLATE:-true}"
export STRIP_DAPO_PROMPT_BOILERPLATE
MATH_PROMPT_PREFIX="${MATH_PROMPT_PREFIX:-}"
export MATH_PROMPT_PREFIX
STRIP_EMPTY_THINKING_GENERATION_PROMPT="${STRIP_EMPTY_THINKING_GENERATION_PROMPT:-false}"
export STRIP_EMPTY_THINKING_GENERATION_PROMPT

LOGGER="${LOGGER:-['console','wandb']}"
if [ "${DISABLE_WANDB}" = "true" ]; then
    LOGGER="['console']"
fi

VERL_ARGS=(
    "algorithm.adv_estimator=grpo"
    "algorithm.use_kl_in_reward=false"
    "algorithm.norm_adv_by_std_in_grpo=true"
    "++algorithm.rlsd.reward_binary_threshold=${REWARD_BINARY_THRESHOLD}"
    "++algorithm.rlsd.prompt_prefix=${PROMPT_PREFIX}"
    "++algorithm.rlsd.prompt_suffix=${PROMPT_SUFFIX}"
    "++algorithm.rlsd.normalize_math_prompt_to_standard_suffix=${NORMALIZE_MATH_PROMPT_TO_STANDARD_SUFFIX}"
    "++algorithm.rlsd.math_instruction_suffix=${MATH_INSTRUCTION_SUFFIX}"
    "++algorithm.rlsd.use_dapo_raw_prompt=${USE_DAPO_RAW_PROMPT}"
    "++algorithm.rlsd.mask_truncated_completions=${MASK_TRUNCATED_COMPLETIONS}"
    "data.train_files=${DATASET_PATH}"
    "data.val_files=${VAL_DATASET_PATH}"
    "++data.custom_cls.path=${SCRIPT_DIR}/verl_rlsd/dataset.py"
    "++data.custom_cls.name=RLSDRLHFDataset"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_COMPLETION_LENGTH}"
    "++data.return_raw_chat=true"
    "++data.apply_chat_template_kwargs.enable_thinking=false"
    "custom_reward_function.path=${SCRIPT_DIR}/verl_rlsd/reward.py"
    "custom_reward_function.name=${REWARD_FUNCTION_NAME}"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.model.trust_remote_code=true"
    "actor_rollout_ref.model.use_remove_padding=true"
    "actor_rollout_ref.model.enable_gradient_checkpointing=true"
    "actor_rollout_ref.model.lora_rank=${LORA_R}"
    "actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}"
    "actor_rollout_ref.model.target_modules=${LORA_TARGET_MODULES}"
    "++actor_rollout_ref.model.lora.merge=${RLSD_MERGE_LORA_FOR_ASYNC_VLLM}"
    "actor_rollout_ref.actor.optim.lr=${LEARNING_RATE}"
    "actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY}"
    "actor_rollout_ref.actor.grad_clip=1.0"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}"
    "++actor_rollout_ref.actor.clip_ratio=${DAPO_EPSILON}"
    "++actor_rollout_ref.actor.clip_ratio_low=${DAPO_EPSILON}"
    "++actor_rollout_ref.actor.clip_ratio_high=${DAPO_EPSILON_HIGH}"
    "actor_rollout_ref.actor.use_kl_loss=true"
    "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}"
    "actor_rollout_ref.actor.entropy_coeff=0.0"
    "actor_rollout_ref.actor.use_dynamic_bsz=true"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_LENGTH}"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.mode=async"
    "actor_rollout_ref.rollout.dtype=bfloat16"
    "actor_rollout_ref.rollout.load_format=safetensors"
    "actor_rollout_ref.rollout.layered_summon=${LAYERED_SUMMON}"
    "actor_rollout_ref.rollout.n=${NUM_GENERATIONS}"
    "actor_rollout_ref.rollout.temperature=${TEMPERATURE}"
    "actor_rollout_ref.rollout.top_p=${TOP_P}"
    "actor_rollout_ref.rollout.top_k=${TOP_K}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_GPU_MEM_UTIL}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${VLLM_TENSOR_PARALLEL_SIZE}"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
    "actor_rollout_ref.rollout.max_model_len=${MAX_LENGTH}"
    "++actor_rollout_ref.rollout.engine_kwargs.vllm.enable_lora=true"
    "++actor_rollout_ref.rollout.engine_kwargs.vllm.max_loras=2"
    "++actor_rollout_ref.rollout.engine_kwargs.vllm.max_lora_rank=${LORA_R}"
    "++actor_rollout_ref.rollout.multi_turn.sampling_params.n=${NUM_GENERATIONS}"
    "++actor_rollout_ref.rollout.multi_turn.sampling_params.temperature=${TEMPERATURE}"
    "++actor_rollout_ref.rollout.multi_turn.sampling_params.top_p=${TOP_P}"
    "++actor_rollout_ref.rollout.multi_turn.sampling_params.top_k=${TOP_K}"
    "++actor_rollout_ref.rollout.multi_turn.sampling_params.min_p=${MIN_P}"
    "++actor_rollout_ref.rollout.multi_turn.sampling_params.presence_penalty=${PRESENCE_PENALTY}"
    "++actor_rollout_ref.rollout.agent.num_workers=${AGENT_LOOP_WORKERS}"
    "trainer.nnodes=1"
    "trainer.n_gpus_per_node=${ACTOR_GPUS_PER_NODE}"
    "trainer.total_training_steps=${MAX_STEPS}"
    "trainer.save_freq=${SAVE_STEPS}"
    "trainer.test_freq=-1"
    "trainer.val_before_train=false"
    "trainer.project_name=${WANDB_PROJECT}"
    "trainer.experiment_name=${WANDB_NAME}"
    "trainer.default_local_dir=${OUTPUT_DIR}"
    "trainer.logger=${LOGGER}"
    "trainer.resume_mode=disable"
)

if [ -n "${VERL_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=(${VERL_EXTRA_ARGS})
else
    EXTRA_ARGS=()
fi

nvidia-smi || true

python -m verl_rlsd.launch "${VERL_ARGS[@]}" "${EXTRA_ARGS[@]}"
