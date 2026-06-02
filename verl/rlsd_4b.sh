#!/bin/bash
#SBATCH -o /gpfs/share/home/2501210611/RLSD/verl_logs/rlsd_4b.%j.out
#SBATCH -p GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=81920M
#SBATCH --time=72:00:00
#SBATCH --exclude=gpua800n07,gpua800n04,gpua800n15,gpua800n26

set -eo pipefail

# Canonical RLSD paper objective: token-gap shaping on GRPO advantages.
RUN_CONFIG="${RUN_CONFIG:-rlsd_4b}"
ADV_ESTIMATOR="${ADV_ESTIMATOR:-rlsd_grpo}"
REWARD_FUNCTION_NAME="${REWARD_FUNCTION_NAME:-compute_score}"
TOKEN_GAP_LAMBDA="${TOKEN_GAP_LAMBDA:-0.5}"
TOKEN_GAP_DECAY_STEPS="${TOKEN_GAP_DECAY_STEPS:-50}"
JSD_TOKEN_CLIP="${JSD_TOKEN_CLIP:-0.2}"
TEACHER_PROMPT_MODE="${TEACHER_PROMPT_MODE:-reference_solution}"
OFFICIAL_TEACHER_PROMPT="${OFFICIAL_TEACHER_PROMPT:-false}"
DISTILLATION_LOSS_COEF="${DISTILLATION_LOSS_COEF:-0.0}"

LEARNING_RATE="${LEARNING_RATE:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
LR_END="${LR_END:-1e-7}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-polynomial}"

NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0.0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0.2}"

# 4 prompts * 8 rollouts = 32 responses. Keep mini/micro small for 3072-token rollouts.
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

MODEL_PATH="${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b}"
export MODEL_PATH
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
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-3072}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_LENGTH="$((MAX_COMPLETION_LENGTH + MAX_PROMPT_LENGTH))"
MAX_TEACHER_PROMPT_LENGTH="${MAX_TEACHER_PROMPT_LENGTH:-${MAX_PROMPT_LENGTH}}"

if [ "${WARMUP_STEPS}" != "0" ]; then
    LR_WARMUP_STEPS="${WARMUP_STEPS}"
    LR_WARMUP_RATIO="0.0"
else
    LR_WARMUP_STEPS="-1"
    LR_WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
fi
MIN_LR_RATIO="${MIN_LR_RATIO:-$(awk -v end="${LR_END}" -v lr="${LEARNING_RATE}" 'BEGIN { if (lr > 0) printf "%.8f", end / lr; else print "0.0" }')}"

VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.9}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
ACTOR_GPUS_PER_NODE="${ACTOR_GPUS_PER_NODE:-1}"
TEACHER_GPUS_PER_NODE="${TEACHER_GPUS_PER_NODE:-1}"
TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-1}"
TEACHER_GPU_MEM_UTIL="${TEACHER_GPU_MEM_UTIL:-0.5}"
AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-8}"

LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]}"
export RLSD_MERGE_LORA_FOR_ASYNC_VLLM="${RLSD_MERGE_LORA_FOR_ASYNC_VLLM:-true}"

DAPO_EPSILON="${DAPO_EPSILON:-0.2}"
DAPO_EPSILON_HIGH="${DAPO_EPSILON_HIGH:-0.28}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
DISTILLATION_TOPK="${DISTILLATION_TOPK:-32}"
DISTILLATION_USE_TASK_REWARDS="${DISTILLATION_USE_TASK_REWARDS:-true}"
USE_POLICY_GRAD_DISTILL="${USE_POLICY_GRAD_DISTILL:-false}"
REWARD_BINARY_THRESHOLD="${REWARD_BINARY_THRESHOLD:-0.5}"
export REWARD_FORMAT_PENALTIES="${REWARD_FORMAT_PENALTIES:-false}"
export REWARD_NO_EOS_PENALTY="${REWARD_NO_EOS_PENALTY:-0.15}"
export REWARD_MULTI_BOXED_PENALTY="${REWARD_MULTI_BOXED_PENALTY:-0.15}"
export REWARD_MIN_CONSECUTIVE_BOXED="${REWARD_MIN_CONSECUTIVE_BOXED:-2}"
export REWARD_BOXED_LAST_TOKEN_FRACTION="${REWARD_BOXED_LAST_TOKEN_FRACTION:-0.05}"
export DISABLE_THINKING_IN_CHAT_TEMPLATE="${DISABLE_THINKING_IN_CHAT_TEMPLATE:-true}"

LOGGER="${LOGGER:-['console','wandb']}"
if [ "${DISABLE_WANDB}" = "true" ]; then
    LOGGER="['console']"
fi

VERL_ARGS=(
    "algorithm.adv_estimator=grpo"
    "++algorithm.rlsd.custom_adv_estimator=${ADV_ESTIMATOR}"
    "algorithm.use_kl_in_reward=false"
    "algorithm.norm_adv_by_std_in_grpo=true"
    "++algorithm.rlsd.token_gap_lambda=${TOKEN_GAP_LAMBDA}"
    "++algorithm.rlsd.lmbda=${TOKEN_GAP_LAMBDA}"
    "++algorithm.rlsd.token_gap_decay_steps=${TOKEN_GAP_DECAY_STEPS}"
    "++algorithm.rlsd.lmbda_decay_steps=${TOKEN_GAP_DECAY_STEPS}"
    "++algorithm.rlsd.jsd_token_clip=${JSD_TOKEN_CLIP}"
    "++algorithm.rlsd.reward_binary_threshold=${REWARD_BINARY_THRESHOLD}"
    "++algorithm.rlsd.teacher_prompt_mode=${TEACHER_PROMPT_MODE}"
    "++algorithm.rlsd.official_teacher_prompt=${OFFICIAL_TEACHER_PROMPT}"
    "++algorithm.rlsd.max_teacher_prompt_length=${MAX_TEACHER_PROMPT_LENGTH}"
    "data.train_files=${DATASET_PATH}"
    "data.val_files=${VAL_DATASET_PATH}"
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
    "actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS}"
    "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${LR_WARMUP_RATIO}"
    "actor_rollout_ref.actor.optim.lr_scheduler_type=${LR_SCHEDULER_TYPE}"
    "actor_rollout_ref.actor.optim.min_lr_ratio=${MIN_LR_RATIO}"
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
    "actor_rollout_ref.rollout.n=${NUM_GENERATIONS}"
    "actor_rollout_ref.rollout.temperature=${TEMPERATURE}"
    "actor_rollout_ref.rollout.top_p=${TOP_P}"
    "actor_rollout_ref.rollout.top_k=${TOP_K}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_GPU_MEM_UTIL}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${VLLM_TENSOR_PARALLEL_SIZE}"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
    "actor_rollout_ref.rollout.max_model_len=${MAX_LENGTH}"
    "++actor_rollout_ref.rollout.multi_turn.sampling_params.n=${NUM_GENERATIONS}"
    "++actor_rollout_ref.rollout.multi_turn.sampling_params.temperature=${TEMPERATURE}"
    "++actor_rollout_ref.rollout.multi_turn.sampling_params.top_p=${TOP_P}"
    "++actor_rollout_ref.rollout.multi_turn.sampling_params.top_k=${TOP_K}"
    "++actor_rollout_ref.rollout.multi_turn.sampling_params.min_p=${MIN_P}"
    "++actor_rollout_ref.rollout.multi_turn.sampling_params.presence_penalty=${PRESENCE_PENALTY}"
    "++actor_rollout_ref.rollout.agent.num_workers=${AGENT_LOOP_WORKERS}"
    "++actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl_rlsd.teacher_agent.RLSDTeacherAgentLoopManager"
    "++distillation.enabled=true"
    "++distillation.nnodes=1"
    "++distillation.n_gpus_per_node=${TEACHER_GPUS_PER_NODE}"
    "++distillation.distillation_loss.topk=${DISTILLATION_TOPK}"
    "++distillation.distillation_loss.use_task_rewards=${DISTILLATION_USE_TASK_REWARDS}"
    "++distillation.distillation_loss.distillation_loss_coef=${DISTILLATION_LOSS_COEF}"
    "++distillation.distillation_loss.use_policy_gradient=${USE_POLICY_GRAD_DISTILL}"
    "++distillation.teacher_models.teacher_model.model_path=${MODEL_PATH}"
    "++distillation.teacher_models.teacher_model.num_replicas=1"
    "++distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP_SIZE}"
    "++distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL}"
    "++distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_LENGTH}"
    "++distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
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
