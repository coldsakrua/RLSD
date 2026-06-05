#!/bin/bash
# Deprecated: use eval_32k_aime25_think.sh or eval_32k_aime25_nothink.sh
exec bash "$(dirname "${BASH_SOURCE[0]}")/eval_32k_aime25_think.sh" "$@"
