#!/bin/bash
#SBATCH --job-name=wait_eval_1726486
#SBATCH -p C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=10:00:00
#SBATCH -o /gpfs/share/home/2501210611/RLSD/logs/wait_eval_1726486/submit_eval.%j.out
#
# CPU-only waiter: does NOT request GPU. After WAIT_HOURS, checks checkpoint and sbatch's eval jobs.
#
# Submit from login node:
#   cd /gpfs/share/home/2501210611/RLSD
#   sbatch submit_eval_32k_after_300step_1726486.sh
#
# Cancel:
#   scancel <job_id>

set -euo pipefail

BASE_DIR="/gpfs/share/home/2501210611/RLSD"
LOG_DIR="${BASE_DIR}/logs/wait_eval_1726486"
cd "${BASE_DIR}"
mkdir -p "${LOG_DIR}"

WAIT_HOURS="${WAIT_HOURS:-5}"
WAIT_SEC=$((WAIT_HOURS * 3600))
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-300}"
POLL_MAX_EXTRA_SEC="${POLL_MAX_EXTRA_SEC:-7200}"

NEW_LORA_PATH="/gpfs/share/home/2501210611/RLSD/outputs/rlsd_4b_strict_split_300step/job_1726486/checkpoint-300"
OLD_LORA_PATH="/gpfs/share/home/2501210611/RLSD/outputs/rlsd_4b_strict_split_250step/job_1722677/checkpoint-300"

EVAL_SCRIPTS=(
  eval_32k_aime24.sh
  eval_32k_aime25.sh
  eval_32k_hmmt25.sh
  eval_32k_math500.sh
)

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_DIR}/submit_eval.log"
}

checkpoint_ready() {
  [[ -d "${NEW_LORA_PATH}" ]] && [[ -f "${NEW_LORA_PATH}/adapter_config.json" ]]
}

update_eval_scripts() {
  local script
  for script in "${EVAL_SCRIPTS[@]}"; do
    if [[ ! -f "${BASE_DIR}/${script}" ]]; then
      log "[error] missing ${script}"
      exit 1
    fi
    if grep -qF "${NEW_LORA_PATH}" "${BASE_DIR}/${script}"; then
      log "[skip] ${script} already points to ${NEW_LORA_PATH}"
      continue
    fi
    if grep -qF "${OLD_LORA_PATH}" "${BASE_DIR}/${script}"; then
      sed -i "s|${OLD_LORA_PATH}|${NEW_LORA_PATH}|g" "${BASE_DIR}/${script}"
      log "[updated] ${script}: ${OLD_LORA_PATH} -> ${NEW_LORA_PATH}"
    else
      log "[warn] ${script} does not contain OLD_LORA_PATH; patching checkpoint_dir line directly"
      sed -i "s|checkpoint_dir=\${CHECKPOINT_DIR:-\${LORA_PATH:-[^}]*}}|checkpoint_dir=\${CHECKPOINT_DIR:-\${LORA_PATH:-${NEW_LORA_PATH}}}|" "${BASE_DIR}/${script}"
      log "[updated] ${script} checkpoint_dir default -> ${NEW_LORA_PATH}"
    fi
  done
}

submit_eval_jobs() {
  local script job_id
  : > "${LOG_DIR}/eval_jobs_submitted.txt"
  for script in "${EVAL_SCRIPTS[@]}"; do
    log "[submit] sbatch ${script}"
    job_id=$(sbatch "${script}" | awk '{print $NF}')
    log "[submit] ${script} -> Slurm job ${job_id}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') ${script} ${job_id}" >> "${LOG_DIR}/eval_jobs_submitted.txt"
  done
}

log "slurm_job_id=${SLURM_JOB_ID:-<local>}"
log "slurm_partition=${SLURM_JOB_PARTITION:-<unknown>}"
log "log_dir=${LOG_DIR}"
log "BASE_DIR=${BASE_DIR}"
log "NEW_LORA_PATH=${NEW_LORA_PATH}"
log "Waiting ${WAIT_HOURS} hour(s) (${WAIT_SEC}s) on CPU node (no GPU)..."
sleep "${WAIT_SEC}"

if ! checkpoint_ready; then
  log "checkpoint not ready after initial wait; polling every ${POLL_INTERVAL_SEC}s (max extra ${POLL_MAX_EXTRA_SEC}s)"
  elapsed=0
  while ! checkpoint_ready; do
    if (( elapsed >= POLL_MAX_EXTRA_SEC )); then
      log "[error] checkpoint still missing: ${NEW_LORA_PATH}"
      log "[error] expected adapter_config.json under checkpoint-300"
      exit 1
    fi
    sleep "${POLL_INTERVAL_SEC}"
    elapsed=$((elapsed + POLL_INTERVAL_SEC))
    log "still waiting... (${elapsed}s extra elapsed)"
  done
fi

log "checkpoint ready: ${NEW_LORA_PATH}"
update_eval_scripts
submit_eval_jobs
log "done. eval job ids listed in ${LOG_DIR}/eval_jobs_submitted.txt"
