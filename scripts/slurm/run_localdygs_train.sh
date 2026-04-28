#!/bin/bash
# LocalDyGS full training: 30000-iteration run on the prepared Stage 2
# scene with the upstream basketball.py preset. First real Stage 2
# training on cow data — output is the canonical deformable Gaussian
# representation that Stage 3 will eventually mine for articulation.
#
# Submit:
#   sbatch /home/yubo/github/Motion-Capture/scripts/slurm/run_localdygs_train.sh
#
# Override defaults via env vars:
#   SCENE      Stage 2 scene dir (default: cow_1/9148_10581 60-frame)
#   OUT_NAME   model output subdir under stage_b/ (default: train_<timestamp>)
#   CONFIG     LocalDyGS config (default: upstream basketball.py)

#SBATCH --account=rrg-vislearn
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --job-name=localdygs_train
#SBATCH --output=/scratch/yubo/jobs/logs/%x_%j.out
#SBATCH --error=/scratch/yubo/jobs/logs/%x_%j.err

set -euo pipefail
mkdir -p /scratch/yubo/jobs/logs

SCENE="${SCENE:-/scratch/yubo/cow_1/9148_10581_output/stage_b/localdygs_scene}"
OUT_NAME="${OUT_NAME:-train_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="/scratch/yubo/cow_1/9148_10581_output/stage_b/$OUT_NAME"
CONFIG="${CONFIG:-/home/yubo/github/LocalDyGS/arguments/vrugz/basketball.py}"
FRAMES_START="${FRAMES_START:-0}"
FRAMES_END="${FRAMES_END:-60}"

echo "=== localdygs full training starting at $(date) ==="
echo "scene:  $SCENE"
echo "out:    $OUT_DIR"
echo "config: $CONFIG"
echo

# opencv module BEFORE venv activate (see SETUP.md ordering rule)
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.9 python/3.11 opencv/4.13.0
source ~/envs/localdygs/bin/activate

# Pre-flight: CUDA extensions resolve
python -c "
import torch
print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
import simple_knn, tinycudann, diff_gaussian_rasterization, cv2
print('CUDA extensions + cv2: all import OK')
"

# Pre-flight: pretrained weights cached (LPIPS+VGG16; see SETUP.md #7)
VGG=$HOME/.cache/torch/hub/checkpoints/vgg16-397923af.pth
LPIPS_VGG=$(python -c "import lpips, os; print(os.path.join(os.path.dirname(lpips.__file__), 'weights/v0.1/vgg.pth'))")
for f in "$VGG" "$LPIPS_VGG"; do
    if [ ! -s "$f" ]; then
        echo "ERROR: required pretrained weight missing: $f"
        echo "Run on login node first:"
        echo "  source ~/envs/localdygs/bin/activate"
        echo "  python -c 'import lpips; lpips.LPIPS(net=\"vgg\")'"
        exit 1
    fi
done
echo "pretrained weights pre-flight OK"

cd ~/github/LocalDyGS
time python train.py \
    -s "$SCENE" \
    -m "$OUT_DIR" \
    --frames_start_end "$FRAMES_START" "$FRAMES_END" \
    --configs "$CONFIG"

echo
echo "=== localdygs full training finished at $(date) ==="
echo "model output dir: $OUT_DIR"
echo "--- output tree ---"
find "$OUT_DIR" -maxdepth 4 -type f 2>/dev/null | head -50
echo "--- size ---"
du -sh "$OUT_DIR" 2>/dev/null
