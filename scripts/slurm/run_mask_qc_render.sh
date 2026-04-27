#!/bin/bash
# SLURM array template: render mask-QC grid JPGs in parallel chunks.
#
# Required env vars (via --export=ALL,...):
#   TOTAL_FRAMES, N_TASKS, DATA, OUT_DIR

#SBATCH --account=def-vislearn
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --job-name=cow_qc_render
#SBATCH --output=/scratch/yubo/jobs/logs/%x_%A_%a.out
#SBATCH --error=/scratch/yubo/jobs/logs/%x_%A_%a.err

set -euo pipefail
mkdir -p /scratch/yubo/jobs/logs

PER=$(( (TOTAL_FRAMES + N_TASKS - 1) / N_TASKS ))
START=$(( SLURM_ARRAY_TASK_ID * PER ))
END=$(( START + PER ))
[ $END -gt $TOTAL_FRAMES ] && END=$TOTAL_FRAMES
echo "[task $SLURM_ARRAY_TASK_ID] frames $START:$END  data=$DATA  out=$OUT_DIR"

module load python/3.11 opencv/4.9.0
cd /home/yubo/github/Motion-Capture
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index numpy >/dev/null

time python -m apps.reconstruction.viz.render_mask_qc_video \
    "$DATA" --frames ${START}:${END} --output_dir "$OUT_DIR"
echo "=== task $SLURM_ARRAY_TASK_ID end $(date) ==="
