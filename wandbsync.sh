#!/bin/bash
set -euo pipefail

# Usage:
#   1) On a machine with Internet and after `wandb login`
#   2) Ensure WANDB_MODE is not offline:
#      unset WANDB_MODE
#   3) Run:
#      bash wandbsync.sh [ROOT_DIR]
#
# Default ROOT_DIR is ./outputs, and the script auto-discovers:
#   */wandb/offline-run-*

ROOT_DIR="${1:-outputs}"
EXCLUDE_GLOBS="${EXCLUDE_GLOBS:-*media/images/*,*media/videos/*,*.png,*.jpg,*.jpeg,*.bmp,*.gif,*.webp,*.mp4,*.avi,*.mov,*.mkv,*.webm,*.npy}"

if [ "${WANDB_MODE:-}" = "offline" ]; then
    echo "[warn] WANDB_MODE=offline. Run 'unset WANDB_MODE' before syncing." >&2
fi

if [ ! -d "${ROOT_DIR}" ]; then
    echo "[error] ROOT_DIR does not exist: ${ROOT_DIR}" >&2
    exit 1
fi

mapfile -t RUN_DIRS < <(find "${ROOT_DIR}" -type d -name 'offline-run-*' -path '*/wandb/*' | sort)

if [ "${#RUN_DIRS[@]}" -eq 0 ]; then
    echo "[info] no offline runs found under ${ROOT_DIR}" >&2
    exit 0
fi

echo "[info] syncing ${#RUN_DIRS[@]} offline runs from ${ROOT_DIR}"
for d in "${RUN_DIRS[@]}"; do
    echo "  - ${d}"
done

wandb sync \
    --exclude-globs "${EXCLUDE_GLOBS}" \
    "${RUN_DIRS[@]}"

