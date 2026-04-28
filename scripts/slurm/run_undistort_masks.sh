#!/bin/bash
# Re-run COLMAP image_undistorter on the SAM3 binary masks for all
# frames currently present in stage_a/colmap_4d/work/. Output goes to
# work/frame_<NNNNNN>/dense_masks/images/<cam>.jpg (parallel structure
# to dense_unmasked/images/).
#
# These undistorted masks are used by LocalDyGS's patch-0006 mask-aware
# loss to restrict optimization to cow pixels only.
#
# Submit:
#   sbatch /home/yubo/github/Motion-Capture/scripts/slurm/run_undistort_masks.sh
#
# Override defaults via env vars:
#   DATA       raw dataset root (default: cow_1/9148_10581)
#   WORK       Stage 1 work/ dir (default: cow_1/9148_10581_output/stage_a/colmap_4d/work)
#   FRAMES     space-separated frame indices (default: all currently in WORK/)
#   PARALLEL   xargs concurrency (default: 16)

#SBATCH --account=def-vislearn
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --job-name=undistort_masks
#SBATCH --output=/scratch/yubo/jobs/logs/%x_%j.out
#SBATCH --error=/scratch/yubo/jobs/logs/%x_%j.err

set -euo pipefail
mkdir -p /scratch/yubo/jobs/logs

DATA="${DATA:-/scratch/yubo/cow_1/9148_10581}"
WORK="${WORK:-/scratch/yubo/cow_1/9148_10581_output/stage_a/colmap_4d/work}"
PARALLEL="${PARALLEL:-16}"

# Default to all frames currently in work/ (skip nothing — we want masks
# for everything for any future re-prep)
if [ -z "${FRAMES:-}" ]; then
    FRAMES=$(ls "$WORK" | grep -oP 'frame_\K[0-9]+' | sed 's/^0*//' | sed 's/^$/0/' | tr '\n' ' ')
fi

echo "=== undistort_masks starting at $(date) ==="
echo "DATA:     $DATA"
echo "WORK:     $WORK"
echo "FRAMES:   ($(echo $FRAMES | wc -w) total)"
echo "PARALLEL: $PARALLEL"
echo

module --force purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.6 colmap/3.12.6

do_frame() {
    local frame=$1
    local frame_pad=$(printf "%06d" "$frame")
    local frame_dir=$WORK/frame_${frame_pad}
    local sparse=$frame_dir/sparse/0
    local out=$frame_dir/dense_masks

    if [ ! -d "$sparse" ]; then
        echo "[frame $frame] WARN: missing sparse model at $sparse, skipping"
        return 0
    fi

    # Build flat mask-images dir for this frame: 01.png, 02.png, ..., 11.png
    local raw_in=${SLURM_TMPDIR:-/tmp}/masks_${frame_pad}
    rm -rf "$raw_in"
    mkdir -p "$raw_in"
    local found=0
    for cam in 01 02 03 04 05 06 07 08 09 10 11; do
        local src=$DATA/masks/${cam}/${frame_pad}.png
        if [ -f "$src" ]; then
            ln -sf "$src" "$raw_in/${cam}.png"
            found=$((found+1))
        else
            echo "[frame $frame] WARN: missing mask $src"
        fi
    done
    if [ "$found" -eq 0 ]; then
        echo "[frame $frame] no masks found, skipping"
        rm -rf "$raw_in"
        return 0
    fi

    if [ -d "$out" ]; then
        rm -rf "$out"
    fi

    echo "[frame $frame] starting image_undistorter on $found masks at $(date +%T)"
    colmap image_undistorter \
        --image_path "$raw_in" \
        --input_path "$sparse" \
        --output_path "$out" \
        --output_type COLMAP \
        > "$out.log" 2>&1
    local n_out=$(ls $out/images 2>/dev/null | wc -l)
    echo "[frame $frame] done at $(date +%T)  (mask images=$n_out)"

    rm -rf "$raw_in"
}
export -f do_frame
export DATA WORK SLURM_TMPDIR

echo "$FRAMES" | tr ' ' '\n' | grep -v '^$' \
    | xargs -P "$PARALLEL" -n 1 -I{} bash -c 'do_frame "$@"' _ {}

echo
echo "=== undistort_masks finished at $(date) ==="
echo "summary of dense_masks/images counts per frame:"
total=$(echo $FRAMES | wc -w)
ok=0
for frame in $FRAMES; do
    pad=$(printf "%06d" "$frame")
    count=$(ls "$WORK/frame_${pad}/dense_masks/images" 2>/dev/null | wc -l)
    if [ "$count" -eq 11 ]; then
        ok=$((ok+1))
    else
        echo "  frame $pad: $count (expected 11)"
    fi
done
echo "frames with all 11 cam-masks: $ok / $total"
