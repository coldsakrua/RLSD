#!/bin/bash
#SBATCH -o logs/deepseek_logs/eval_4k_hmmt25_deepseek_math_7b_rl.%j.out
#SBATCH -p GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=81920M
#SBATCH --time=24:00:00
#SBATCH --exclude=gpua800n04,gpua800n24

set -eo pipefail
nvidia-smi

# sbatch may copy the script into /var/spool/...; prefer SLURM submit dir over BASH_SOURCE.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR}}"
else
    _EVAL_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    BASE_DIR="$(cd "${_EVAL_SCRIPT}/../.." && pwd)"
fi
cd "${BASE_DIR}"

source activate anchor
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
set -u
mkdir -p logs/deepseek_logs outputs

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_HOST_IP=127.0.0.1
export TORCH_CUDA_ARCH_LIST=8.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

model_path=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/deepseek-math-7b-rl}

datasets_csv=${DATASETS:-hmmt25}
data_format=${DATA_FORMAT:-auto}
data_root=${DATA_ROOT:-/gpfs/share/home/2501210611/prefernce-learning/preference_learning/data}
checkpoint_dir=${CHECKPOINT_DIR:-${LORA_PATH:-}}
max_lora_rank=${MAX_LORA_RANK:-${VLLM_MAX_LORA_RANK:-64}}
use_lora=${USE_LORA:-0}
num_samples=${NUM_SAMPLES:-0}
val_n=${VAL_N:-16}
pass_at_k=${PASS_AT_K:-1,4,8,16}
# DeepSeekMath-RL model context is 4096; keep 1024 tokens prompt headroom by default.
max_new_tokens=${MAX_NEW_TOKENS:-3072}
max_model_len=${MAX_MODEL_LEN:-4096}
# Paper-style Pass@K sampling default; set TEMPERATURE=0 for greedy pass@1.
temperature=${TEMPERATURE:-0.7}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:-20}
min_p=${MIN_P:-0.0}
presence_penalty=${PRESENCE_PENALTY:-0.0}
seed=${SEED:-42}
tensor_parallel_size=${TENSOR_PARALLEL_SIZE:-1}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.9}
disable_custom_all_reduce=${DISABLE_CUSTOM_ALL_REDUCE:-1}
generate_batch_size=${GENERATE_BATCH_SIZE:-16}
force_base_tokenizer=${FORCE_BASE_TOKENIZER:-1}

stamp=$(date -u +%Y%m%d_%H%M%S)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  run_tag="${stamp}_job${SLURM_JOB_ID}"
else
  run_tag="${stamp}"
fi
output_json=${OUTPUT_JSON:-outputs/eval_4k_hmmt25_deepseek_math_7b_rl/cot_4k/eval_${run_tag}.json}

mkdir -p "$(dirname "${output_json}")"
echo "[EVAL] model_path=${model_path}"
echo "[EVAL] checkpoint_dir=${checkpoint_dir:-<none>}"
echo "[EVAL] USE_LORA=${use_lora} (1=use LoRA, 0=base model)"
echo "[EVAL] DATASETS=${datasets_csv}"
echo "[EVAL] DATA_ROOT=${data_root}"
echo "[EVAL] MAX_NEW_TOKENS=${max_new_tokens}"
echo "[EVAL] MAX_MODEL_LEN=${max_model_len}"
echo "[EVAL] FORCE_BASE_TOKENIZER=${force_base_tokenizer} (1=base tokenizer/chat_template)"
echo "[EVAL] DISABLE_CUSTOM_ALL_REDUCE=${disable_custom_all_reduce} (1=disable vLLM custom all-reduce)"
echo "[EVAL] TEMPERATURE=${temperature} TOP_P=${top_p} TOP_K=${top_k}"
echo "[EVAL] GENERATE_BATCH_SIZE=${generate_batch_size} VAL_N=${val_n}"
echo "[EVAL] output_json=${output_json}"

cmd=(
  python eval_math_vllm_local.py
  --model-path "${model_path}"
  --data-root "${data_root}"
  --data-format "${data_format}"
  --output-json "${output_json}"
  --num-samples "${num_samples}"
  --val-n "${val_n}"
  --pass-at-k "${pass_at_k}"
  --generate-batch-size "${generate_batch_size}"
  --max-new-tokens "${max_new_tokens}"
  --temperature "${temperature}"
  --top-p "${top_p}"
  --top-k "${top_k}"
  --min-p "${min_p}"
  --presence-penalty "${presence_penalty}"
  --seed "${seed}"
  --tensor-parallel-size "${tensor_parallel_size}"
  --gpu-memory-utilization "${gpu_memory_utilization}"
  --max-model-len "${max_model_len}"
)

IFS=',' read -ra _ds <<< "${datasets_csv}"
for _n in "${_ds[@]}"; do
  _n="${_n#"${_n%%[![:space:]]*}"}"
  _n="${_n%"${_n##*[![:space:]]}"}"
  [[ -z "${_n}" ]] && continue
  cmd+=(--dataset="${_n}")
done

if [[ "${use_lora}" == "1" ]]; then
  if [[ -n "${checkpoint_dir}" ]]; then
    cmd+=(--lora-path "${checkpoint_dir}")
  fi
  if [[ -n "${checkpoint_dir}" && -n "${max_lora_rank}" && "${max_lora_rank}" != "0" ]]; then
    cmd+=(--max-lora-rank "${max_lora_rank}")
  fi
fi

if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  cmd+=(--enforce-eager)
fi

if [[ "${disable_custom_all_reduce}" == "1" ]]; then
  cmd+=(--disable-custom-all-reduce)
fi

if [[ "${force_base_tokenizer}" == "1" ]]; then
  cmd+=(--force-base-tokenizer)
fi

"${cmd[@]}"

echo "[EVAL] done -> ${output_json}"
