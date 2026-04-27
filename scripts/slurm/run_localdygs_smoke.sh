#!/bin/bash
# LocalDyGS smoke-test: 500-iteration training run on the prepared
# Stage 2 scene. Verifies env + scene layout actually train without
# errors before committing to a 30000-iter full run.
#
# Submit:
#   sbatch /home/yubo/github/Motion-Capture/scripts/slurm/run_localdygs_smoke.sh
#
# Override defaults via env vars:
#   SCENE      Stage 2 scene dir (default: cow_1/9148_10581 60-frame)
#   OUT_NAME   model output subdir under stage_b/ (default: smoke_<timestamp>)

#SBATCH --account=rrg-vislearn
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --job-name=localdygs_smoke
#SBATCH --output=/scratch/yubo/jobs/logs/%x_%j.out
#SBATCH --error=/scratch/yubo/jobs/logs/%x_%j.err

set -euo pipefail
mkdir -p /scratch/yubo/jobs/logs

SCENE="${SCENE:-/scratch/yubo/cow_1/9148_10581_output/stage_b/localdygs_scene}"
OUT_NAME="${OUT_NAME:-smoke_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="/scratch/yubo/cow_1/9148_10581_output/stage_b/$OUT_NAME"
CONFIG="/home/yubo/github/Motion-Capture/apps/reconstruction/stage_b_localdygs/configs/cow_smoke.py"

echo "=== localdygs smoke starting at $(date) ==="
echo "scene:  $SCENE"
echo "out:    $OUT_DIR"
echo "config: $CONFIG"
echo

# opencv module BEFORE venv activate (see SETUP.md ordering rule)
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.9 python/3.11 opencv/4.13.0
source ~/envs/localdygs/bin/activate

# Pre-flight: verify CUDA extensions resolve on this GPU node.
# tinycudann probes compute capability at import — fails on CPU-only login
# nodes but should succeed on an A100 (SM 8.0 = what we built for).
python -c "
import torch
print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
import simple_knn, tinycudann, diff_gaussian_rasterization, cv2
print('CUDA extensions + cv2: all import OK')
"

cd ~/github/LocalDyGS
time python train.py \
    -s "$SCENE" \
    -m "$OUT_DIR" \
    --frames_start_end 0 60 \
    --configs "$CONFIG"

echo
echo "=== localdygs smoke finished at $(date) ==="
echo "model output dir: $OUT_DIR"
echo "--- output tree ---"
find "$OUT_DIR" -maxdepth 3 2>/dev/null | head -30
echo "--- size ---"
du -sh "$OUT_DIR" 2>/dev/null
