#!/bin/bash
#SBATCH -o logs/rlsd_4b_strict.%j.out
#SBATCH -p GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=81920M
#SBATCH --time=72:00:00

set -eo pipefail
nvidia-smi

BASE_DIR="/gpfs/share/home/2501210611/RLSD"
cd "${BASE_DIR}"
mkdir -p logs

source activate anchor
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_HOST_IP=127.0.0.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b}
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/aggregated_l3plus/train.parquet}
DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-${BASE_DIR}/outputs/hf_cache}
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/outputs/rlsd_4b_strict}
RUN_CONFIG=${RUN_CONFIG:-rlsd_4b_strict}
JOB_TAG="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR}/job_${JOB_TAG}"
mkdir -p "${OUTPUT_DIR}"

MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-12949}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-8}
PER_DEVICE_BS=${PER_DEVICE_BS:-1}
MAX_STEPS=${MAX_STEPS:-300}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.6}

ROLLOUT_FILTER=${ROLLOUT_FILTER:-all}
LMBDA=${LMBDA:-0.5}
LMBDA_DECAY_STEPS=${LMBDA_DECAY_STEPS:-50}
JSD_TOKEN_CLIP=${JSD_TOKEN_CLIP:-0.05}

LAMBDA_PLUS=${LAMBDA_PLUS:-0.3}
LAMBDA_MINUS=${LAMBDA_MINUS:-0.3}
LAMBDA_PLUS_MIN=${LAMBDA_PLUS_MIN:-0.0}
LAMBDA_MINUS_MIN=${LAMBDA_MINUS_MIN:-0.0}
FALLBACK_DECAY_STEPS=${FALLBACK_DECAY_STEPS:-150}
FALLBACK_EPS0=${FALLBACK_EPS0:-0.05}
ADV_CLIP_LOW=${ADV_CLIP_LOW:--1.0}
ADV_CLIP_HIGH=${ADV_CLIP_HIGH:-1.0}
SUPPRESS_GT_SHORTCUT=${SUPPRESS_GT_SHORTCUT:-true}
ANSWER_TOKEN_DOWNWEIGHT=${ANSWER_TOKEN_DOWNWEIGHT:-0.2}
REWARD_BINARY_THRESHOLD=${REWARD_BINARY_THRESHOLD:-0.5}
FALLBACK_TAIL_TOKENS=${FALLBACK_TAIL_TOKENS:-8}
DAPO_EPSILON=${DAPO_EPSILON:-0.1}
DAPO_EPSILON_HIGH=${DAPO_EPSILON_HIGH:-0.2}

LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"}
LORA_R=${LORA_R:-64}
LORA_ALPHA=${LORA_ALPHA:-128}

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 1 \
    --gradient_accumulation_steps "${GRAD_ACC_STEPS}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    opsd_train_anchor_strict.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --dataset_split train \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --learning_rate 1e-6 \
    --max_grad_norm 1.0 \
    --per_device_train_batch_size "${PER_DEVICE_BS}" \
    --gradient_accumulation_steps "${GRAD_ACC_STEPS}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_config "${RUN_CONFIG}" \
    --max_steps "${MAX_STEPS}" \
    --num_generations "${NUM_GENERATIONS}" \
    --max_completion_length 4096 \
    --save_steps 25 \
    --logging_steps 2 \
    --attn_implementation sdpa \
    --torch_dtype bfloat16 \
    --max_length 4096 \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEM_UTIL}" \
    --vllm_tensor_parallel_size 1 \
    --use_peft true \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_target_modules "${LORA_TARGET_MODULES}" \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda "${LMBDA}" \
    --lmbda_decay_steps "${LMBDA_DECAY_STEPS}" \
    --fixed_teacher true \
    --jsd_token_clip "${JSD_TOKEN_CLIP}" \
    --rollout_filter "${ROLLOUT_FILTER}" \
    --lambda_plus "${LAMBDA_PLUS}" \
    --lambda_minus "${LAMBDA_MINUS}" \
    --lambda_plus_min "${LAMBDA_PLUS_MIN}" \
    --lambda_minus_min "${LAMBDA_MINUS_MIN}" \
    --fallback_decay_steps "${FALLBACK_DECAY_STEPS}" \
    --fallback_eps0 "${FALLBACK_EPS0}" \
    --adv_clip_low "${ADV_CLIP_LOW}" \
    --adv_clip_high "${ADV_CLIP_HIGH}" \
    --suppress_gt_shortcut "${SUPPRESS_GT_SHORTCUT}" \
    --answer_token_downweight "${ANSWER_TOKEN_DOWNWEIGHT}" \
    --reward_binary_threshold "${REWARD_BINARY_THRESHOLD}" \
    --fallback_tail_tokens "${FALLBACK_TAIL_TOKENS}" \
    --epsilon "${DAPO_EPSILON}" \
    --dapo_epsilon_high "${DAPO_EPSILON_HIGH}" \
    --disable_wandb true \
    --gradient_checkpointing
