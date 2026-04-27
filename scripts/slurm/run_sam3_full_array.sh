#!/bin/bash
# SLURM array template for masking a long sequence with SAM3.
#
# Splits the frame range across N parallel A100 tasks, model loaded once
# per task. Set TOTAL_FRAMES, N_TASKS, DATA before submission.
#
# Usage:
#   sbatch --array=0-$((N_TASKS-1)) \
#          --export=ALL,TOTAL_FRAMES=1434,N_TASKS=10,DATA=/scratch/yubo/cow_1/9148_10581 \
#          run_sam3_full_array.sh

#SBATCH --account=rrg-vislearn
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=cow_sam3_array
#SBATCH --output=/scratch/yubo/jobs/logs/%x_%A_%a.out
#SBATCH --error=/scratch/yubo/jobs/logs/%x_%A_%a.err

set -euo pipefail
mkdir -p /scratch/yubo/jobs/logs

PER=$(( (TOTAL_FRAMES + N_TASKS - 1) / N_TASKS ))   # ceil division
START=$(( SLURM_ARRAY_TASK_ID * PER ))
END=$(( START + PER ))
[ $END -gt $TOTAL_FRAMES ] && END=$TOTAL_FRAMES
echo "[task $SLURM_ARRAY_TASK_ID/$N_TASKS] frames $START:$END  data=$DATA"

module load StdEnv/2023 gcc/12.3 cuda/12.9 python/3.12 opencv/4.13.0
source $HOME/envs/sam3/bin/activate
cd /home/yubo/github/Motion-Capture

echo "=== task $SLURM_ARRAY_TASK_ID start $(date) ==="
# --no_overlay to keep the file-count manageable (otherwise 11×N×2 jpgs)
time python -m apps.reconstruction.preprocess_segment_sam3 \
    "$DATA" --frames ${START}:${END} --no_overlay
echo "=== task $SLURM_ARRAY_TASK_ID end $(date) ==="
