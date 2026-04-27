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

**Round-3 env fix landed** (commit `fbaf6a4`): prep sbatch switched from
`~/envs/localdygs/` to `~/envs/cleanply/`. Reason: open3d's wheelhouse
wheel pins torch==2.6, which downgrades from our 2.10 install and
ABI-breaks the localdygs CUDA extensions. SETUP.md lesson #6 added.

Prior milestones:
- Visual review of `aggregated_4d.ply` cleared on Mac (Yubo, 2026-04-27).
- Stage 1 60-frame sweep finished. 60/60 frames, 8.6 M points aggregated, 129 MB. Per-frame point counts 25 K–250 K.
- LocalDyGS env at `~/envs/localdygs/`. Patches `0001-simple_knn-include-cfloat` + `0002-dataset_readers-downsample-1` applied. Committed (`362781b`, `ec9e3be`).

## Next concrete step

**In flight: prep job 59959992** (resubmission after the open3d-missing failure of `59959671`). Watcher running in background; will dump logs + scene-layout inspection on completion.

After prep succeeds, smoke-test LocalDyGS training on a single A100 (~10–20 min, 500–1000 iters) to verify env + scene layout actually train without errors.

```bash
# (after smoke test plan exists)
sbatch scripts/slurm/run_localdygs_smoke.sh   # not yet written
```

## Recent activity

Newest first.

| Date | Event | Detail |
|---|---|---|
| 2026-04-27 | Prep resubmitted | job 59959992, uses cleanply venv (no torch deps) |
| 2026-04-27 | Stale scene wiped | `/scratch/.../localdygs_scene/` from 16:51 (mystery origin) removed |
| 2026-04-27 | localdygs venv restored | `pip install --force-reinstall torch==2.10.0 torchvision`; simple_knn import OK; tinycudann OK only on GPU node |
| 2026-04-27 | Prep env fix committed | `fbaf6a4` — prep sbatch uses ~/envs/cleanply/, SETUP.md lesson #6 |
| 2026-04-27 | Prep job 59959671 FAILED | `ModuleNotFoundError: No module named 'open3d'` in localdygs venv (4 s) |
| 2026-04-27 | Visual review approved | `aggregated_4d.ply` cleared on Mac; proceed to Stage 2 prep |
| 2026-04-27 | STATUS.md created + first prep submitted | commit `c2ce86d` |
| 2026-04-27 | Round-2 patches landed | commit `ec9e3be` — patch 0002 (downsample 2.0→1.0) + finalized SETUP.md |
| 2026-04-27 | Round-1 stage_b scaffolding landed | commit `362781b` — env recipe + Stage 1→LocalDyGS data prep + simple_knn cfloat patch |
| 2026-04-27 | Stage 1 60-frame sweep complete | 60/60 frames, 8.6 M aggregated points |
| 2026-04-27 | LocalDyGS env built | ~/envs/localdygs/, torch 2.10+CUDA 12.9, tinycudann@SM 8.0 |
