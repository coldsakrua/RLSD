#!/bin/bash
# On a machine with Internet and after `wandb login`:
#   unset WANDB_MODE
# Then sync offline runs (metrics only; large media excluded).
#
# Training jobs (grpo_4b_strict.sh / rlsd_*.sh) write runs under:
#   ${OUTPUT_DIR}/wandb/offline-run-*
# Example after a job finishes (replace JOB_ID):
#   wandb sync --exclude-globs "*media/images/*,..." "outputs/grpo_4b_strict/job_JOB_ID/wandb/offline-run-*"

wandb sync \
	--exclude-globs "*media/images/*,*media/videos/*,*.png,*.jpg,*.jpeg,*.bmp,*.gif,*.webp,*.mp4,*.avi,*.mov,*.mkv,*.webm,*.npy"\
	results/ttt_tanks/*/wandb/offline-run-*

# Optional: jsonl replay output (see scripts/jsonl_metrics_to_wandb.py)
# wandb sync \
# 	--exclude-globs "*media/images/*,*media/videos/*,*.png,*.jpg,*.jpeg,*.bmp,*.gif,*.webp,*.mp4,*.avi,*.mov,*.mkv,*.webm,*.npy"\
# 	wandb_jsonl_replay/wandb/offline-run-*

# wandb sync \
# 	--exclude-globs "*media/images/*,*media/videos/*,*.png,*.jpg,*.jpeg,*.bmp,*.gif,*.webp,*.mp4,*.avi,*.mov,*.mkv,*.webm,*.npy"\
# 	results/ttt_tanks_onlyrecon/*/wandb/offline-run-*
