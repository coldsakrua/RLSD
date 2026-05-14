#!/bin/bash
set -euo pipefail

# Usage:
#   1) On a machine with Internet and after `wandb login`
#   2) Ensure WANDB_MODE is not offline:
#      unset WANDB_MODE
#   3) Run:
#      bash wandbsync.sh [ROOT_DIR [PROJECT]]
#
#   Optional env (override cloud destination / resync):
#     WANDB_PROJECT / WANDB_ENTITY — passed as wandb sync -p / -e
#     WANDB_INCLUDE_SYNCED=true — if runs were already synced elsewhere and you
#       need to upload again (e.g. push a copy to another project)
#
# Default ROOT_DIR is ./outputs, and the script auto-discovers:
#   */wandb/offline-run-*

ROOT_DIR="${1:-outputs}"
PROJECT="${2:-${WANDB_PROJECT:-}}"
ENTITY="${WANDB_ENTITY:-}"
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
if [ -n "${ENTITY}" ]; then
    echo "[info] entity: ${ENTITY}"
fi
if [ -n "${PROJECT}" ]; then
    echo "[info] project: ${PROJECT}"
fi
if [ "${WANDB_INCLUDE_SYNCED:-false}" = "true" ]; then
    echo "[info] WANDB_INCLUDE_SYNCED=true (including already-synced runs)"
fi
for d in "${RUN_DIRS[@]}"; do
    echo "  - ${d}"
done

SYNC_CMD=(wandb sync --exclude-globs "${EXCLUDE_GLOBS}")
if [ -n "${ENTITY}" ]; then
    SYNC_CMD+=(-e "${ENTITY}")
fi
if [ -n "${PROJECT}" ]; then
    SYNC_CMD+=(-p "${PROJECT}")
fi
if [ "${WANDB_INCLUDE_SYNCED:-false}" = "true" ]; then
    SYNC_CMD+=(--include-synced)
fi
"${SYNC_CMD[@]}" "${RUN_DIRS[@]}"

