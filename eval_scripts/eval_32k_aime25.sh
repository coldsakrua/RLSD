#!/bin/bash
# Deprecated: use eval_32k_aime25_think.sh or eval_32k_aime25_nothink.sh
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    _EVAL_DIR="${SLURM_SUBMIT_DIR}/eval_scripts"
else
    _EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
exec bash "${_EVAL_DIR}/eval_32k_aime25_think.sh" "$@"
