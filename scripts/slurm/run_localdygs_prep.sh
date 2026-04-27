#!/bin/bash
# Stage 1 → Stage 2 data prep: convert per-frame COLMAP outputs into
# LocalDyGS scene layout. CPU-only — no GPU needed (file shuffling +
# Open3D voxel downsample on point clouds).
#
# Submit:
#   sbatch /home/yubo/github/Motion-Capture/scripts/slurm/run_localdygs_prep.sh
#
# Override defaults via env vars:
#   STAGE_A_ROOT  path to stage_a/colmap_4d (default: cow_1/9148_10581 60-frame)
#   FRAMES_START  start index into discovered frames (default: 0)
#   FRAMES_END    end index, exclusive (default: 60)

#SBATCH --account=def-vislearn
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --job-name=localdygs_prep
#SBATCH --output=/scratch/yubo/jobs/logs/%x_%j.out
#SBATCH --error=/scratch/yubo/jobs/logs/%x_%j.err

set -euo pipefail
mkdir -p /scratch/yubo/jobs/logs

STAGE_A_ROOT="${STAGE_A_ROOT:-/scratch/yubo/cow_1/9148_10581_output/stage_a/colmap_4d}"
FRAMES_START="${FRAMES_START:-0}"
FRAMES_END="${FRAMES_END:-60}"

echo "=== localdygs prep starting at $(date) ==="
echo "stage_a_root: $STAGE_A_ROOT"
echo "frames:       [$FRAMES_START, $FRAMES_END)"
echo

# Use cleanply venv (numpy + open3d + plyfile) — NOT localdygs.
# Reason: open3d's wheelhouse wheel pins torch==2.6, which downgrades the
# 2.10 install in localdygs and ABI-breaks the CUDA extensions
# (simple_knn / tinycudann / diff_gaussian_rasterization). The prep script
# only needs numpy/open3d/plyfile and zero torch, so use the dedicated env.
module --force purge
module load StdEnv/2023 python/3.11
source ~/envs/cleanply/bin/activate

# Sanity: verify env can import what the prep needs (open3d for voxel downsample)
python -c "import open3d, plyfile, numpy; print('prep imports OK — open3d', open3d.__version__)"

cd /home/yubo/github/Motion-Capture
time python -m apps.reconstruction.stage_b_localdygs.prepare_localdygs_data \
    "$STAGE_A_ROOT" \
    --frames-start-end "$FRAMES_START" "$FRAMES_END"

echo
echo "=== localdygs prep finished at $(date) ==="
