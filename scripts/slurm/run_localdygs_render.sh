#!/bin/bash
# Render trained LocalDyGS model — outputs train/test view PNGs and video.
# Lets us visually verify what the trained Gaussians actually represent.
#
# Submit:
#   MODEL_DIR=/scratch/.../stage_b/train_<timestamp> \
#       sbatch /home/yubo/github/Motion-Capture/scripts/slurm/run_localdygs_render.sh
#
# Override defaults via env vars:
#   MODEL_DIR  trained model dir (REQUIRED — must contain point_cloud/iteration_*/)
#   ITERATION  which checkpoint to render (default: -1 = pick highest)
#   CONFIG     LocalDyGS config (default: upstream basketball.py — must match training)

#SBATCH --account=rrg-vislearn
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:30:00
#SBATCH --job-name=localdygs_render
#SBATCH --output=/scratch/yubo/jobs/logs/%x_%j.out
#SBATCH --error=/scratch/yubo/jobs/logs/%x_%j.err

set -euo pipefail
mkdir -p /scratch/yubo/jobs/logs

if [ -z "${MODEL_DIR:-}" ]; then
    echo "ERROR: MODEL_DIR not set. Pass via:"
    echo "  MODEL_DIR=/scratch/.../train_<ts> sbatch $0"
    exit 1
fi
ITERATION="${ITERATION:--1}"
CONFIG="${CONFIG:-/home/yubo/github/LocalDyGS/arguments/vrugz/basketball.py}"

echo "=== localdygs render starting at $(date) ==="
echo "model:     $MODEL_DIR"
echo "iteration: $ITERATION"
echo "config:    $CONFIG"
echo

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.9 python/3.11 opencv/4.13.0
source ~/envs/localdygs/bin/activate

python -c "
import torch, simple_knn, tinycudann, diff_gaussian_rasterization, cv2
print('CUDA available:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0))
print('extensions OK')
"

cd ~/github/LocalDyGS
time python render.py \
    -m "$MODEL_DIR" \
    --iteration "$ITERATION" \
    --frames_start_end 0 60 \
    --configs "$CONFIG" \
    --skip_video

echo
echo "=== localdygs render finished at $(date) ==="
echo "--- render output tree ---"
find "$MODEL_DIR" -type d \( -name "ours_*" -o -name "train" -o -name "test" \) 2>/dev/null
echo "--- sample images ---"
find "$MODEL_DIR" -name "*.png" -type f 2>/dev/null | head -10
echo "--- size ---"
du -sh "$MODEL_DIR" 2>/dev/null
