#!/bin/bash
#SBATCH -o logs/verl_rlsd_4b_2gpu.%j.out
#SBATCH -p GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=81920M
#SBATCH --time=72:00:00

set -eo pipefail
nvidia-smi

BASE_DIR="/gpfs/share/home/2501210611/RLSD"
cd "${BASE_DIR}"
mkdir -p logs

source activate anchor
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b}
RAW_DATA_PARQUET=${RAW_DATA_PARQUET:-${BASE_DIR}/data/aggregated_l3plus/train.parquet}
VERL_DATA_DIR=${VERL_DATA_DIR:-${BASE_DIR}/data/verl_opsd}
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/outputs/verl_rlsd_4b_2gpu}
RUN_NAME=${RUN_NAME:-verl_rlsd_4b_2gpu}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-256}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-1024}
MAX_RESP_LEN=${MAX_RESP_LEN:-1536}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-4096}
ROLLOUT_N=${ROLLOUT_N:-8}
PPO_MINIBATCH=${PPO_MINIBATCH:-128}
PPO_MICRO_PER_GPU=${PPO_MICRO_PER_GPU:-4}
LOGPROB_MICRO_PER_GPU=${LOGPROB_MICRO_PER_GPU:-8}
VLLM_MEM_UTIL=${VLLM_MEM_UTIL:-0.50}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-25}

# RLSD (base) knobs
export RLSD_JSD_TOKEN_CLIP=${RLSD_JSD_TOKEN_CLIP:-0.05}
export RLSD_MIXED_LAMBDA=${RLSD_MIXED_LAMBDA:-0.5}
export RLSD_MIXED_DECAY_STEPS=${RLSD_MIXED_DECAY_STEPS:-50}
export RLSD_ROLLOUT_FILTER=${RLSD_ROLLOUT_FILTER:-all}
export RLSD_BINARY_THRESHOLD=${RLSD_BINARY_THRESHOLD:-0.5}
export RLSD_ADV_CLIP_LOW=${RLSD_ADV_CLIP_LOW:--10.0}
export RLSD_ADV_CLIP_HIGH=${RLSD_ADV_CLIP_HIGH:-10.0}

mkdir -p "${VERL_DATA_DIR}" "${OUTPUT_DIR}"

python verl_prepare_opsd_dataset.py \
  --input_parquet "${RAW_DATA_PARQUET}" \
  --output_dir "${VERL_DATA_DIR}" \
  --val_ratio 0.02 \
  --seed 42

python verl_train_main_ppo_rlsd.py \
  algorithm.adv_estimator=rlsd_verl \
  data.train_files="${VERL_DATA_DIR}/train.parquet" \
  data.val_files="${VERL_DATA_DIR}/val.parquet" \
  data.prompt_key=prompt \
  data.reward_fn_key=data_source \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LEN}" \
  data.max_response_length="${MAX_RESP_LEN}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINIBATCH}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_PER_GPU}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.grad_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_PER_GPU}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_MEM_UTIL}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  custom_reward_function.path="${BASE_DIR}/verl_reward_opsd.py" \
  custom_reward_function.name=compute_score \
  trainer.logger='["console"]' \
  trainer.project_name='RLSD-VERL' \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.default_local_dir="${OUTPUT_DIR}"
