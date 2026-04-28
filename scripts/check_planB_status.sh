#!/bin/bash
# Wake-up helper: summarize Plan B overnight chain status.
# Run this when you start a new Claude session in the morning.

set -uo pipefail

source /scratch/yubo/jobs/planB_chain.txt 2>/dev/null || {
    echo "ERROR: /scratch/yubo/jobs/planB_chain.txt not found."
    echo "Either no Plan B was submitted, or scratch was purged."
    exit 1
}

echo "=== Plan B chain (submitted overnight) ==="
echo

format_state() {
    local jobid=$1
    local label=$2
    local state=$(sacct -j "$jobid" -X -n -o State 2>/dev/null | head -1 | tr -d ' ')
    local sq_state=$(squeue -j "$jobid" -h -o '%T %r' 2>/dev/null | head -1)
    local elapsed=$(sacct -j "$jobid" -X -n -o Elapsed 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "$sq_state" ]; then
        printf "  %-40s [%-12s] %s\n" "$label" "$state" "($sq_state)"
    elif [ -n "$state" ]; then
        printf "  %-40s [%-12s] elapsed=%s\n" "$label" "$state" "$elapsed"
    else
        printf "  %-40s [no record]\n" "$label"
    fi
}

format_state "$STAGE1" "Stage 1 array (76 frames missing)"
format_state "$U1"     "U1 undistort_unmasked (new 76 frames)"
format_state "$M1"     "M1 undistort_masks (all frames)"
echo
echo "  --- Exp 2 (136 frames, NO mask): control ---"
format_state "$P2"     "P2 prep"
format_state "$T2"     "T2 train"
format_state "$R2"     "R2 render"
echo
echo "  --- Exp 1 (136 frames + mask-aware loss): MAIN ---"
format_state "$P1"     "P1 prep"
format_state "$T1"     "T1 train"
format_state "$R1"     "R1 render"
echo
echo "  --- Exp 3 (60 frames + mask-aware loss): control ---"
format_state "$P3"     "P3 prep"
format_state "$T3"     "T3 train"
format_state "$R3"     "R3 render"
echo

echo "=== Train metrics if available ==="
for E in e1 e2 e3; do
    LOG=/scratch/yubo/cow_1/9148_10581_output/stage_b/train_planB_$E/outputs.log
    if [ -f "$LOG" ]; then
        echo "  $E:"
        grep "ITER\|PSNR" "$LOG" | grep -E "^\[ITER" | tail -3 | sed 's/^/    /'
    fi
done
echo

echo "=== Failed/cancelled jobs (need investigation) ==="
sacct -j "$STAGE1,$U1,$M1,$P2,$T2,$R2,$P1,$T1,$R1,$P3,$T3,$R3" -X -n -P -o JobID,JobName,State,ExitCode \
    | awk -F'|' '$3 != "COMPLETED" && $3 != "PENDING" && $3 != "RUNNING" && $3 != ""' \
    | head -10
echo

echo "=== render output PNG counts ==="
for E in e1 e2 e3; do
    DIR=/scratch/yubo/cow_1/9148_10581_output/stage_b/train_planB_$E
    if [ -d "$DIR" ]; then
        T=$(ls "$DIR/train/ours_30000/renders/"*.png 2>/dev/null | wc -l)
        Te=$(ls "$DIR/test/ours_30000/renders/"*.png 2>/dev/null | wc -l)
        printf "  %s: train=%d test=%d  (dir=%s)\n" "$E" "$T" "$Te" "$DIR"
    fi
done
