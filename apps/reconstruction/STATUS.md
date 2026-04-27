# Reconstruction Project — Live Status

> **Purpose**: single source of truth for "where are we *right now*". Updated
> after every completed unit of work and committed to git. If a Claude session
> dies (bus error etc.), the next session reads this file and continues
> without losing context.
>
> **Architectural decisions, pipeline overview, design rationale** → `PROJECT.md`.
> **Live progress, last completed step, next command** → this file.

---

## Current stage

**Stage 1 → Stage 2 handoff.** Stage 1 (per-frame COLMAP MVS) is complete on
the 60-frame `cow_1/9148_10581` sweep. Stage 2 (LocalDyGS) is fully set up:
env built, both upstream patches landed, prep script written. The remaining
hop is **executing `prepare_localdygs_data.py` on real Stage 1 outputs for
the first time**, then a smoke-test LocalDyGS training run.

## Last completed

**Visual review of the 60-frame aggregated PLY** (Yubo, on Mac, 2026-04-27):
`/scratch/yubo/cow_1/9148_10581_output/stage_a/colmap_4d/human/aggregated_4d.ply`
looks good. Cleared to proceed to Stage 2 data prep.

Prior milestones:
- Stage 1 60-frame sweep finished 2026-04-27. 60/60 frames produced fused clouds. 8.6 M points aggregated, 129 MB. Per-frame point counts 25 K–250 K.
- LocalDyGS env at `~/envs/localdygs/`. Patches `0001-simple_knn-include-cfloat` + `0002-dataset_readers-downsample-1` applied to the local LocalDyGS clone. Both committed (`362781b`, `ec9e3be`).

## Next concrete step

Run Stage 1 → Stage 2 data prep on the existing 60-frame Stage 1 output. CPU-only job (no GPU needed for the prep script — it's file shuffling + Open3D voxel downsample).

```bash
sbatch /home/yubo/github/Motion-Capture/scripts/slurm/run_localdygs_prep.sh
```

(Script will be written next — see "Recent activity".)

After prep completes successfully, the sibling step is a smoke-test LocalDyGS training run on a single A100 (~10–20 min, 500–1000 iters) to verify the env + scene layout actually train without errors.

## Recent activity

Newest first.

| Date | Event | Detail |
|---|---|---|
| 2026-04-27 | Visual review approved | `aggregated_4d.ply` cleared on Mac; proceed to Stage 2 prep |
| 2026-04-27 | STATUS.md created | This file. Establishes session-resilience pattern. |
| 2026-04-27 | Round-2 patches landed | commit `ec9e3be` — patch 0002 (downsample 2.0→1.0) + finalized SETUP.md |
| 2026-04-27 | Round-1 stage_b scaffolding landed | commit `362781b` — env recipe + Stage 1→LocalDyGS data prep + simple_knn cfloat patch |
| 2026-04-27 | Stage 1 60-frame sweep complete | 60/60 frames, 8.6 M aggregated points, /scratch/.../9148_10581_output/stage_a/colmap_4d/ |
| 2026-04-27 | LocalDyGS env built | ~/envs/localdygs/, torch 2.10+CUDA 12.9, tinycudann@SM 8.0, two non-obvious patches |
