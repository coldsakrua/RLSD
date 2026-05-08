#!/bin/bash
#SBATCH -o logs/rlsd_anchor_4b_1gpu.%j.out
#SBATCH -p GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=81920M
#SBATCH --time=72:00:00

set -eo pipefail
nvidia-smi

cd /gpfs/share/home/2501210611/prefernce-learning/OPSD
mkdir -p logs

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="logs/rlsd_anchor_4b_1gpu.${RUN_TS}.out"
exec > >(tee -a "${RUN_LOG}") 2>&1
echo "[INFO] Timestamped log file: ${RUN_LOG}"

source activate anchor
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_HOST_IP=127.0.0.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b}
DATASET_PATH=${DATASET_PATH:-/gpfs/share/home/2501210611/prefernce-learning/preference_learning/data/OPSD}
DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-/gpfs/share/home/2501210611/prefernce-learning/outputs/hf_cache}
OUTPUT_DIR=${OUTPUT_DIR:-/gpfs/share/home/2501210611/prefernce-learning/outputs/rlsd_anchor}
RUN_CONFIG=${RUN_CONFIG:-qwen34b_rlsd_1gpu}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-12949}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-8}
PER_DEVICE_BS=${PER_DEVICE_BS:-4}
MAX_STEPS=${MAX_STEPS:-560}
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.60}
ROLLOUT_FILTER=${ROLLOUT_FILTER:-all}
LMBDA=${LMBDA:-0.5}
LMBDA_DECAY_STEPS=${LMBDA_DECAY_STEPS:-50}
JSD_TOKEN_CLIP=${JSD_TOKEN_CLIP:-0.05}

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 1 \
    --gradient_accumulation_steps "${GRAD_ACC_STEPS}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    opsd_train_anchor.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --dataset_split train \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size "${PER_DEVICE_BS}" \
    --gradient_checkpointing \
    --gradient_accumulation_steps "${GRAD_ACC_STEPS}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_config "${RUN_CONFIG}" \
    --num_train_epochs 3 \
    --max_steps "${MAX_STEPS}" \
    --max_completion_length 1024 \
    --save_steps 25 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 4096 \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEM_UTIL}" \
    --vllm_tensor_parallel_size 1 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda "${LMBDA}" \
    --lmbda_decay_steps "${LMBDA_DECAY_STEPS}" \
    --fixed_teacher true \
    --jsd_token_clip "${JSD_TOKEN_CLIP}" \
    --rollout_filter "${ROLLOUT_FILTER}" \
    --disable_wandb true
