#!/bin/bash
# Re-undistort all 60 cow frames using UNMASKED raw images.
# Reuses the existing per-frame sparse models in
# work/frame_<NNNNNN>/sparse/0/ from Stage 1; produces
# work/frame_<NNNNNN>/dense_unmasked/images/<cam>.jpg.
#
# Why: Stage 1's dense/images/ have masks applied (cow-only, black bg),
# which makes LocalDyGS converge to "render all black" as the trivial
# loss minimum. LocalDyGS expects normal multi-view images.
#
# Submit:
#   sbatch /home/yubo/github/Motion-Capture/scripts/slurm/run_undistort_unmasked.sh
#
# Override defaults:
#   DATA       raw dataset root      (default: cow_1/9148_10581)
#   WORK       Stage 1 work/ dir     (default: cow_1/9148_10581_output/stage_a/colmap_4d/work)
#   FRAMES     space-separated frame indices (default: 0 5 10 ... 295)
#   PARALLEL   how many to run at once (default: 16)

#SBATCH --account=def-vislearn
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --job-name=undistort_unmasked
#SBATCH --output=/scratch/yubo/jobs/logs/%x_%j.out
#SBATCH --error=/scratch/yubo/jobs/logs/%x_%j.err

set -euo pipefail
mkdir -p /scratch/yubo/jobs/logs

DATA="${DATA:-/scratch/yubo/cow_1/9148_10581}"
WORK="${WORK:-/scratch/yubo/cow_1/9148_10581_output/stage_a/colmap_4d/work}"
FRAMES="${FRAMES:-$(seq 0 5 295 | tr '\n' ' ')}"
PARALLEL="${PARALLEL:-16}"

echo "=== undistort_unmasked starting at $(date) ==="
echo "DATA:     $DATA"
echo "WORK:     $WORK"
echo "FRAMES:   $FRAMES"
echo "PARALLEL: $PARALLEL"
echo

module --force purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.6 colmap/3.12.6

# Per-frame routine. One frame = one COLMAP image_undistorter call.
# We point image_undistorter at a temporary dir of symlinks named "01.jpg"
# .. "11.jpg" — matches what's in the sparse model's images.txt (flat
# per-camera filenames, no subdirs).
do_frame() {
    local frame=$1
    local frame_pad=$(printf "%06d" "$frame")
    local frame_dir=$WORK/frame_${frame_pad}
    local sparse=$frame_dir/sparse/0
    local out=$frame_dir/dense_unmasked

    if [ ! -d "$sparse" ]; then
        echo "[frame $frame] WARN: missing sparse model at $sparse, skipping"
        return 0
    fi

    # Build a flat raw-images dir for this frame (symlinks; no copies)
    local raw_in=${SLURM_TMPDIR:-/tmp}/raw_${frame_pad}
    rm -rf "$raw_in"
    mkdir -p "$raw_in"
    for cam in 01 02 03 04 05 06 07 08 09 10 11; do
        local src=$DATA/images/${cam}/${frame_pad}.jpg
        if [ -f "$src" ]; then
            ln -sf "$src" "$raw_in/${cam}.jpg"
        else
            echo "[frame $frame] WARN: missing raw $src"
        fi
    done

    # Wipe and rebuild dense_unmasked/
    if [ -d "$out" ]; then
        rm -rf "$out"
    fi

    echo "[frame $frame] starting image_undistorter at $(date +%T)"
    colmap image_undistorter \
        --image_path "$raw_in" \
        --input_path "$sparse" \
        --output_path "$out" \
        --output_type COLMAP \
        > "$out.log" 2>&1
    echo "[frame $frame] done at $(date +%T)  (images=$(ls $out/images 2>/dev/null | wc -l))"

    # Cleanup the symlink dir
    rm -rf "$raw_in"
}
export -f do_frame
export DATA WORK SLURM_TMPDIR

# Run frames in parallel; xargs -P respects $PARALLEL.
echo "$FRAMES" | tr ' ' '\n' | grep -v '^$' \
    | xargs -P "$PARALLEL" -n 1 -I{} bash -c 'do_frame "$@"' _ {}

echo
echo "=== undistort_unmasked finished at $(date) ==="
echo "summary of dense_unmasked/images counts per frame:"
for frame in $FRAMES; do
    pad=$(printf "%06d" "$frame")
    count=$(ls "$WORK/frame_${pad}/dense_unmasked/images" 2>/dev/null | wc -l)
    if [ "$count" -ne 11 ]; then
        echo "  frame $pad: $count (EXPECTED 11)"
    fi
done | head -20
echo "total frames with all 11 cams: $(for frame in $FRAMES; do
    pad=$(printf "%06d" "$frame")
    count=$(ls "$WORK/frame_${pad}/dense_unmasked/images" 2>/dev/null | wc -l)
    [ "$count" -eq 11 ] && echo "ok"
done | wc -l) / $(echo $FRAMES | wc -w)"
