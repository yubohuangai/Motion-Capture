#!/bin/bash
# Final encode: ffmpeg the per-frame JPGs into one MP4 < 100 MB.
#
# Required env vars: DATA, OUT_DIR, OUT_VIDEO

#SBATCH --account=def-vislearn
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --job-name=cow_qc_encode
#SBATCH --output=/scratch/yubo/jobs/logs/%x_%j.out
#SBATCH --error=/scratch/yubo/jobs/logs/%x_%j.err

set -euo pipefail
mkdir -p /scratch/yubo/jobs/logs

module load python/3.11 opencv/4.9.0 ffmpeg/7.1.1
cd /home/yubo/github/Motion-Capture
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip >/dev/null
pip install --no-index numpy >/dev/null

time python -m apps.reconstruction.viz.render_mask_qc_video \
    "$DATA" --output_dir "$OUT_DIR" --output_video "$OUT_VIDEO" \
    --encode_only --fps 30 --crf 28
ls -lh "$OUT_VIDEO"
