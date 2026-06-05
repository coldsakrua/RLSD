#!/bin/bash
# Deprecated: use eval_32k_aime24_think.sh or eval_32k_aime24_nothink.sh
exec bash "$(dirname "${BASH_SOURCE[0]}")/eval_32k_aime24_think.sh" "$@"
