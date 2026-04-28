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

**Render job `59965124` completed** (12:27 wall, with patch 0005 OOM
fix). All 660 PNGs written to `train_20260427_190608/{train,test}/ours_30000/`
(540 train + 120 test). Job exit was SIGPIPE (13:0) from a post-render
shell pipe, but the render itself succeeded.

**Critical finding from inspection**: renders are **completely black**.
- GT images: 3840×2160, 7.7 MB, pixel mean 174.7 (cow on background)
- Renders: 3840×2160 ✓, 25 KB, pixel min=0 max=1 mean=0.0

The model converged to render-pure-black for every view and frame.

## Root cause (high confidence)

Stage 1's `dense/images/` are **mask-blackened** (post-`apply_masks`,
the COLMAP image_undistorter input has cow regions only; background
pixels are zeroed). Our prep script symlinked these into the LocalDyGS
scene. So LocalDyGS sees images that are ~95% black background + ~5%
cow.

L1+SSIM loss treats every pixel equally. With 95% of pixels at value 0,
a model rendering all-zeros achieves MSE = (cow_region_area) × (cow_pixel_variance)
≈ 0.05 × 150² ≈ 1125 → PSNR ≈ 17, matching observed test PSNR 13.66.

The model has zero pressure to learn the cow — "render black" is the
trivial-minimum solution. The plateau in metrics across 15K iters
confirms convergence to this minimum.

## Plan A — COMPLETE end-to-end ✓

| # | Step | Wall | Job ID | Result |
|---|---|---|---|---|
| 1 | Undistort 60 unmasked frames | 12 s | `59967114` | 60×11 = 660 unmasked images at `dense_unmasked/images/` |
| 2 | Re-prep LocalDyGS scene | 30 s | (in-watcher) | 73,784-pt init pcd, scene at `localdygs_scene/` |
| 3 | Re-train (30K iters) | 1:59:18 | `59967435` | Train PSNR **22.94** (was 8.94 with masked imgs) |
| 4 | Re-render iter-30000 | 57 min | `59972470` | All 660 PNGs at `train/test/ours_30000/` (5.7-8.7 MB each) |

**Total Plan A wall time**: ~3.5 h.

**Output**: `/scratch/yubo/cow_1/9148_10581_output/stage_b/train_20260427_220326/`
(12 GB — 1 GB model + 11 GB rendered PNGs).

## Result quality

**Train views** (5 samples, 540 total): render mean matches GT within
<1%, L1 ~0.04-0.06. Model has memorized training views.

**Test views** (5 samples, 120 total): render mean roughly matches
GT (within ~10%), but L1 ~0.16-0.27 — 4-5× higher than train. The
model produces cow-shaped renders but not pixel-aligned with held-out
GT. This is the half-circle rig limitation: cams 0 and 10 view angles
the model never trained on, with no articulation prior to interpolate.

## Next concrete step

**Decision pending — Yubo's call**:

1. **Visually inspect renders on Mac.** Path:
   `/scratch/yubo/cow_1/9148_10581_output/stage_b/train_20260427_220326/{train,test}/ours_30000/{renders,gt}/`
   (rsync or scp). The numerical metrics say "cow-shaped, not pixel-perfect" — eyeballing tells us if "cow-shaped" is good enough.

2. **If visually good**: Stage 2 is done for `cow_1/9148_10581`. Can pivot to
   Stage 3 (articulation discovery from per-Gaussian trajectories) or to
   another dataset.

3. **If not good enough**: options to consider —
   - longer training (basketball.py supports up to 200K iters; another ~10 h)
   - denser init pcd (re-prep with `--target-points 250000`)
   - tune hyperparams (basketball.py is for outdoor sport, not cattle)
   - SAM2-tracked masks → mask-aware loss (combine A + B for cow-only output without loss collapse)

Ready to commit Plan A as a milestone and pause for Yubo's review.

**Plan A worked** — model now actually learns training views.

| Metric | Iter 30000, before (masked) | Iter 30000, after (unmasked) |
|---|---|---|
| Train L1 | 0.391 | **0.047** (8× better) |
| Train PSNR | 8.94 | **22.94** (+14 dB) |
| Test L1 | 0.143 | 0.217 |
| Test PSNR | 13.66 | 11.46 |

Train metrics now make sense (train < test L1, train > test PSNR — normal
overfitting). Test PSNR dropped because the previous "render all black"
scored coincidentally well against masked GT; with unmasked GT, all-black
gets 0 PSNR. Generalization to held-out cams 0/10 is genuinely hard
(half-circle rig, novel viewpoints), but that's a model-quality issue
not a pipeline-broken issue.

**Validation: Plan A is materially different** (sampled across 4 frames × 3 cams):

| Frame | Cam | Masked mean / black% | Unmasked mean / black% |
|---|---|---|---|
| 0 | 01/05/11 | ~144-165 / 0% | ~144-165 / 0% (frame 0 was already unmasked — coincidence; cow filled frame) |
| 100 | 01/05/11 | 7-16 / **84-95%** | 158-170 / 0% |
| 200 | 01/05/11 | 6-17 / **87-95%** | 162-170 / 0% |
| 295 | 01/05/11 | 4-13 / **89-97%** | 161-170 / 0% |

Confirms: the previous training was seeing 87-97% black images for
most frames, drove the model to "render zero" minimum. With unmasked
images, the loss landscape now has the cow + background actually
visible — model has a normal multi-view scene to fit.

## Next concrete step

Watcher will fire on retrain completion → submit render → inspect rendered
PNGs (expecting actual cow content this time).

## Recent activity

Newest first.

| Date | Event | Detail |
|---|---|---|
| 2026-04-28 | **Plan A render 59972470 DONE** | 57 min wall, all 660 PNGs (5.7-8.7 MB each). Train L1 ~0.05, test L1 ~0.20. Model produces real cow renders. |
| 2026-04-28 | Render 59972470 resubmitted | walltime 90 min |
| 2026-04-28 | Render 59971560 timed out @ 30 min | 372/540 train rendered before kill. **Renders contain real cow content** (mean ~175 matches GT, vs old all-black mean 0). Slower because non-trivial Gaussian splat work. |
| 2026-04-28 | Plan A render 59971560 submitted | for the new train output dir |
| 2026-04-28 | **Plan A retrain 59967435 DONE** | 1:59:18 wall. Train PSNR 22.94 (vs 8.94 before). Pipeline works. |
| 2026-04-27 | Plan A retrain 59967435 submitted | with unmasked images |
| 2026-04-27 | Plan A prep DONE | scene rebuilt with `--image-subdir dense_unmasked/images` |
| 2026-04-27 | Plan A undistort 59967114 DONE | 12 s wall, 60×11 unmasked images, validated cross-frame |
| 2026-04-27 | Plan A undistort job 59967114 submitted | xargs -P 16 over 60 frames |
| 2026-04-27 | Plan A landed | `32b2379` — `run_undistort_unmasked.sh` + `--image-subdir` flag in prep |
| 2026-04-27 | **Render diagnosis: renders are pure black** | Pixel mean 0.0 vs GT 174.7. Root cause: training fed masked images (95% black bg) → trivial minimum. Decision needed before retraining. |
| 2026-04-27 | Render job 59965124 completed | 540+120 PNGs at 3840×2160, but pixel content all-zero |
| 2026-04-27 | Render job 59965124 resubmitted | with patch 0005, mem=32G |
| 2026-04-27 | Patch 0005 + render OOM lesson | `2a8fa30` — incremental save in render.py; SETUP #6c |
| 2026-04-27 | Render job 59964986 OOM-killed | 191/540 train views, CPU 25 GB / 24 GB. Render buffers all tensors before writing. |
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
