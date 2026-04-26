#!/bin/bash
# Launcher: submit a SLURM array job over an explicit frame list.
#
# Writes the frames to a temp file (avoiding SLURM's comma-in-value parsing
# bug) and submits one array task per frame.
#
# Usage:
#   ./launch_4d_array.sh <data_root> <frames-spec>
# where <frames-spec> is either:
#   - 'START:END[:STEP]' range (end-exclusive): 0:50:5  → 0,5,10,...,45
#   - explicit comma list:                      0,30,60,90,120
#
# Example:
#   ./launch_4d_array.sh /scratch/yubo/cow_1/9148_10581 0:50:5

set -euo pipefail
DATA="${1:?usage: $0 <data_root> <frames-spec>}"
SPEC="${2:?usage: $0 <data_root> <frames-spec>}"

# Expand spec into one-frame-per-line file
TS=$(date +%s)
FRAMES_FILE="/scratch/yubo/jobs/frames_${TS}.txt"
mkdir -p /scratch/yubo/jobs

if [[ "$SPEC" == *:* ]]; then
    IFS=':' read -ra P <<< "$SPEC"
    if [ ${#P[@]} -eq 2 ]; then
        seq ${P[0]} 1 $((${P[1]}-1)) > "$FRAMES_FILE"
    else
        seq ${P[0]} ${P[2]} $((${P[1]}-1)) > "$FRAMES_FILE"
    fi
else
    tr ',' '\n' <<< "$SPEC" > "$FRAMES_FILE"
fi

N=$(wc -l < "$FRAMES_FILE")
echo "[launch_4d_array] data=$DATA"
echo "[launch_4d_array] frames file=$FRAMES_FILE (n=$N)"
echo "[launch_4d_array] frames: $(paste -sd, $FRAMES_FILE)"
echo "[launch_4d_array] each task: ~10 min on full A100"

HERE="$(cd "$(dirname "$0")" && pwd)"
sbatch --array=0-$((N-1)) \
       --export=ALL,FRAMES_FILE="$FRAMES_FILE",DATA="$DATA" \
       "$HERE/run_4d_array.sh.template"
