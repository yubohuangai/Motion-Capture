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

**Full training COMPLETED** (job `59960861`, 1:46:01 wall, exit 0).
30K iters at 4.75 it/s on A100, peak 36.96 GB RAM. Output:
`/scratch/yubo/cow_1/9148_10581_output/stage_b/train_20260427_190608/`
(1.0 GB, both `point_cloud/iteration_15000/` and `iteration_30000/`).

**Metrics are concerning** (need investigation):
| Iter | Test L1 | Test PSNR | Train L1 | Train PSNR |
|---|---|---|---|---|
| 15000 | 0.143 | 13.66 | 0.391 | 8.94 |
| 30000 | 0.143 | 13.66 | 0.391 | 8.94 |

Three flags: (1) test PSNR > train PSNR (unusual — typically train
overfits first); (2) metrics flat across 15K iters (model has
plateaued — `position_lr=0` by LocalDyGS scaffold design means anchors
can't move; learning is stuck on the offset/MLP/hash params alone);
(3) PSNR 13-14 is low — well-conditioned 3DGS gets 25-30+.

## Next concrete step

**In flight: render job 59964986** (commit `bc0f4f8` — patch 0004 +
render sbatch). Renders the iter-30000 model on both train and test
cams. Output goes into the existing `train_20260427_190608/{train,test}/ours_30000/`
subdirs. ~30 min walltime; expect ~10 min real time for 660 images.

After render completes:
- visually inspect a few train/test cam renders
- if "looks like a cow but blurry" → next move is more iterations
  (basketball.py supports up to 200K) or denser init pcd (re-prep with
  `--target-points 250000`)
- if "complete garbage / not a cow" → hyperparams likely wrong; revisit
  basketball.py choice or LocalDyGS preset

## Recent activity

Newest first.

| Date | Event | Detail |
|---|---|---|
| 2026-04-27 | Render job 59964986 submitted | iter 30000, MODEL_DIR env override |
| 2026-04-27 | Patch 0004 + render sbatch landed | `bc0f4f8` — render.py hardcoded CUDA_VISIBLE_DEVICES=2; SETUP lesson #6b |
| 2026-04-27 | **Full training 59960861 COMPLETED** | 1:46:01, 30K iters @ 4.75 it/s. Test PSNR 13.66, train PSNR 8.94 — flat across iters, concerning |
| 2026-04-27 | Full training job 59960861 submitted | 30K iters, 4h walltime, basketball.py config, A100 |
| 2026-04-27 | Full training sbatch committed | `3838825` — `scripts/slurm/run_localdygs_train.sh` |
| 2026-04-27 | **Smoke job 59960670 SUCCEEDED** | 3:10 wall, 500/500 iters, loss 0.53→0.17, PSNR 6.83→16.87. Stage 2 pipeline validated end-to-end. |
| 2026-04-27 | Smoke job 59960670 resubmitted | with patch 0003 applied |
| 2026-04-27 | Patch 0003 landed | commit `1e53592` — bounds test_num to in-range indices; SETUP lesson #2c |
| 2026-04-27 | Smoke job 59960468 FAILED | 25 s, IndexError in Colmap_Dataset.load_images_path — upstream test_num=[0,10,20,30] assumes ≥31 cams; ours has 11 |
| 2026-04-27 | Smoke job 59960468 resubmitted | with VGG16 + LPIPS weights pre-cached |
| 2026-04-27 | LPIPS/VGG16 weights pre-cached + fail-fast guard | commit `b1304d4`. VGG16 ~528 MB to ~/.cache; lpips bundled weights wget'd from upstream GitHub (wheelhouse lpips missing them) |
| 2026-04-27 | Smoke job 59960152 FAILED | 9 min wall on `URLError [Errno 101]` — train.py downloads VGG16 at import; compute node no internet |
| 2026-04-27 | Smoke job submitted | `59960152` — single A100, 500 iters, cow_smoke.py config |
| 2026-04-27 | Smoke artifacts committed | `51b985b` — `configs/cow_smoke.py` + `scripts/slurm/run_localdygs_smoke.sh` |
| 2026-04-27 | Prep job 59959992 SUCCEEDED | 17 s, 60 frames, 73,784-pt init pcd, 4K cameras |
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
