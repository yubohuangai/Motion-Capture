#!/bin/bash
# Launcher: submit a SLURM array job over an explicit frame list.
#
# Usage:
#   ./launch_4d_array.sh <data_root> <frames-spec>
# where <frames-spec> is either:
#   - 'START:END:STEP' range (end-exclusive): 0:50:5  → 0,5,10,...,45
#   - explicit comma list:                    0,30,60,90,120
#
# Example:
#   ./launch_4d_array.sh /scratch/yubo/cow_1/9148_10581 0:50:5

set -euo pipefail
DATA="${1:?usage: $0 <data_root> <frames-spec>}"
SPEC="${2:?usage: $0 <data_root> <frames-spec>}"

# Expand spec into a comma-separated list
if [[ "$SPEC" == *:* ]]; then
    IFS=':' read -ra P <<< "$SPEC"
    if [ ${#P[@]} -eq 2 ]; then
        FRAMES=$(seq -s, ${P[0]} 1 $((${P[1]}-1)))
    else
        FRAMES=$(seq -s, ${P[0]} ${P[2]} $((${P[1]}-1)))
    fi
else
    FRAMES="$SPEC"
fi

N=$(awk -F, '{print NF}' <<< "$FRAMES")
echo "[launch_4d_array] data=$DATA"
echo "[launch_4d_array] frames=$FRAMES (n=$N)"
echo "[launch_4d_array] each task: ~10 min on full A100"

HERE="$(cd "$(dirname "$0")" && pwd)"
sbatch --array=0-$((N-1)) \
       --export=ALL,FRAMES="$FRAMES",DATA="$DATA" \
       "$HERE/run_4d_array.sh.template"
