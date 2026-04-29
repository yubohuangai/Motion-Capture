# Cow Canonical Deformation Model — Project Overview

> Living document. **Last updated: 2026-04-28.**
> If you are an AI agent or engineer joining this project, **read this first**.
> Update the "Update log" at the bottom whenever the plan, repo layout, or
> decisions change.
>
> **For live runtime state (in-flight jobs, latest checkpoints, today's
> next step): see [`STATUS.md`](./STATUS.md)** — that file is updated
> after every completed unit of work. PROJECT.md is the durable
> architecture / decisions doc; STATUS.md is the running journal.

---

## 1. North Star

Build a **canonical deformable model** of a captured cow, **using
foundational methods (no category-specific priors like SMAL or
DensePose-CSE)**:
> a rest-pose 3D shape  +  a per-frame deformation field
> that together explain every observed video frame from every camera.

This canonical model is what lets you *re-pose* the cow, *retarget* its
motion, *compare body shape across animals*, and downstream — fit
body-condition scoring, weight estimation, gait analysis. None of that
is possible from a sequence of independent per-frame point clouds.

**Term notes (since these recur):**
- *canonical* = one rest-pose reference shape (not N per-frame meshes)
- *deformable* = a general per-frame deformation field; we explicitly
  allow but do not require this to factor into a skeleton + skinning
  (i.e., articulation). The cow's dominant motion *is* skeletal, but
  forcing an explicit skeleton into the model with off-the-shelf methods
  (BANMo, RAC, SMAL) requires category-specific priors that conflict
  with the foundational constraint. See §6.
- *foundational* (constraint) = no priors specific to cattle / animals /
  humans. Pure geometric + photometric optimization. Outputs whatever
  the data alone justifies; articulation is then a *discovery* problem,
  not a *fitting* problem. Discovering articulation is a follow-on
  research direction (§9).

## 2. Why this is hard (and why we don't have it yet)

1. **L-shaped rig**: 11 cameras placed along two perpendicular edges of a
   rectangle (cams 01–06 along one edge, 07–11 along the perpendicular
   edge; the corner where the edges meet is empty). At any instant, only
   ~two adjacent sides of the cow are observed (the other two sides are
   invisible). Static reconstruction is fundamentally incomplete per
   frame. **However**, the cow rotates and walks through the rig over
   the full 1434-frame (~48 s @ 30 fps) sequence — so over time, every
   camera eventually sees all sides of the cow. Motion is non-uniform:
   the cow stops, walks, and rotates at varying speeds, so a frame
   subset must span the full sequence (not just the first 10 s) to
   leverage this temporal 360°-coverage property.
2. **The cow articulates** (legs, head, neck, tail rotate around joints)
   plus small non-rigid residuals (breathing, muscle bulge). Naive "stack
   all per-frame clouds" doesn't merge into one coherent surface because
   corresponding points have moved. Foundational methods model this as
   a general deformation field; articulation is left implicit.
3. **Classical MVS has no notion of time**: each frame is reconstructed
   independently. To leverage temporal info you need a *deformation
   model*, which classical methods don't provide.
4. **Holstein fur** has very low contrast in the dark patches — even SIFT +
   ZNCC struggles. Our CLAHE + masking work mitigates this but doesn't fully
   solve it.

## 3. Pipeline

```
Multi-view 30-fps RGB  +  calibration (intri.yml, extri.yml)
                          │
        ┌─────────────────▼──────────────────┐
        │ SAM3 foreground masks              │  per-frame, per-camera cow
        │ preprocess_segment_sam3.py         │  → data/masks/<cam>/<frame>.png
        │ (--frames START:END[:STEP])        │
        └─────────────────┬──────────────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │ STAGE 1: per-frame dense MVS       │  N independent 3D clouds, one
        │ stage_a_colmap_4d/                 │  per time frame. Each is partial
        │ (NB: "_4d" naming refers to the    │  (only the rig-visible side of the
        │  output dataset structure          │  cow). NO temporal modeling at
        │  3D-points × time, NOT to the      │  this stage — see §5 lesson on
        │  reconstruction method, which is   │  why per-frame is correct here.
        │  per-frame independent COLMAP MVS) │
        │   sparse:  full image (no mask)    │
        │   dense:   masked, cow-only        │  Sparse uses unmasked: tight cow
        │                                    │  mask leaves <100 SIFT pts and
        │                                    │  breaks COLMAP depth bounds.
        │ Output: <out>/human/, work/        │
        │   frame_<NNNNNN>_fused.ply         │
        │   frame_<NNNNNN>_sparse.ply        │
        │   frame_<NNNNNN>_thumb.jpg         │
        │   work/frame_<NNNNNN>/{sparse,     │
        │     dense, dense_unmasked,         │
        │     dense_masks}/                  │
        └─────────────────┬──────────────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │ STAGE 2: LocalDyGS canonical       │  ✓ IMPLEMENTED + RUN
        │ deformable model (foundational)    │  Multi-view dynamic 3DGS with
        │ stage_b_localdygs/                 │  seed-anchored local spaces +
        │                                    │  static/dynamic decoupling.
        │ Driver:                            │  Best result on cow_1/9148_10581:
        │   prepare_localdygs_data.py →      │  136fr (stride-mixed full
        │   LocalDyGS train.py →             │  rotation) + mask-aware loss
        │   LocalDyGS render.py              │  → train PSNR 22.49,
        │ +  6 patches in patches/           │     test  PSNR 12.14 (best).
        │                                    │
        │ Output: stage_b/train_<name>/      │  See §6 for results table; see
        │   point_cloud/iteration_30000/     │  STATUS.md for the live state of
        │   {train,test}/ours_30000/         │  any in-flight retrains.
        │     {renders,gt}/                  │
        └─────────────────┬──────────────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │ STAGE 3: articulation discovery    │  ⚠ FUTURE (research)
        │ (cluster per-Gaussian trajectories │  Cluster trajectories from
        │  → fit skeleton, no priors)        │  Stage 2 output into rigid
        │                                    │  parts; fit joint graph by
        │                                    │  RANSAC over part-pair offsets.
        │                                    │  Deferred until Stage 2 is
        │ See §6.                            │  producing something to mine.
        └────────────────────────────────────┘
```

## 4. Stage 1 — per-frame COLMAP MVS

**Status: complete** for `cow_1/9148_10581`. 136 frames reconstructed
end-to-end (60 stride-5 over 0..295 + 76 stride-15 over 300..1425; the
mixed stride spans the cow's full 360° rotation, ~48 s of capture).

### Inputs
- `<data_root>/intri.yml`, `<data_root>/extri.yml` — EasyMocap calibration
- `<data_root>/images/<cam>/<frame:06d>.jpg` — multi-view sequence
- `<data_root>/masks/<cam>/<frame:06d>.png` — produced by SAM3 (run first)

### Driver
`apps/reconstruction/stage_a_colmap_4d/run_stage_a_colmap_4d.py` —
loops over frames; per-frame calls the proven single-frame
`stage_a_colmap.run_stage_a_colmap` with `--no-mask-sparse` (essential —
see §5 lesson on sparse vs dense masking).

The directory name `stage_a_colmap_4d` is misleading: the "_4d" refers
to the 4D output dataset structure (per-frame 3D clouds × time), NOT
to a 4D reconstruction method. Each frame is reconstructed independently
in 3D. Renaming to `stage_a_colmap_perframe` is on the long-term cleanup
list but defers until other refactors settle.

### Output layout
```
<data>_output/stage_a/colmap_4d/
├── human/                            ← per-frame scp/rsync target
│   ├── aggregated_4d.ply             (color-coded union, viz only)
│   ├── frame_<NNNNNN>_fused.ply      (per-frame dense cloud)
│   ├── frame_<NNNNNN>_sparse.ply     (per-frame SIFT 3D points)
│   └── frame_<NNNNNN>_thumb.jpg      (one input view as visual ref)
└── work/                             ← machine intermediates
    └── frame_<NNNNNN>/
        ├── sparse/0/                 (COLMAP sparse model)
        ├── dense/                    (PatchMatch workspace + fused.ply)
        ├── dense_unmasked/images/    (Stage 2 training input)
        └── dense_masks/images/       (Stage 2 mask-aware loss input)
```

The `dense_unmasked/` and `dense_masks/` subdirs are produced by a
separate undistort pass (see `scripts/slurm/run_undistort_unmasked.sh`,
`run_undistort_masks.sh`); they exist because the original `dense/images/`
are mask-blackened, which traps Stage 2's loss in a degenerate
"render-all-zeros" minimum (see §5 lesson on mask-aware loss).

### Resource numbers (cow_1, 11 cams @ 4K, A100)
- SAM3 masking (full 1434 frames): ~30 min wall, single GPU
- Stage 1 per frame: ~10 min on full A100 (PatchMatch dominates)
- 60-frame stride-5 SLURM array: ~12 min wall (parallel)
- 76-frame stride-15 array: ~10 min wall (parallel, smaller batch)
- Per-frame point counts: 25K–250K masked-cow points (varies with cow
  position in frame; later frames where cow fills more of the rig give
  denser clouds)

### Failure modes (all hit and fixed during development)
1. **Wrong depth bounds** → fusion gets 0 points despite full depth maps.
   *Fix*: per-frame triangulation gives COLMAP per-camera bounds.
2. **Tight mask kills sparse triangulation** → too few SIFT pts →
   "no sparse model in workspace" error.
   *Fix*: `--no-mask-sparse` (default in 4D driver).
3. **Comma-in-value SLURM env var** → array tasks 1..N silently fail.
   *Fix*: launcher writes a per-task frames file.
4. **Empty-array reduction crash** in dense-reconstruction stats reporter.
   *Fix*: guard with `if n_pts > 0`.

### How to run

For a fresh dataset, the canonical sequence:

```bash
# 1. Mask the sequence (once per dataset + frame stride)
sbatch /scratch/yubo/jobs/run_sam3_seq_<dataset>.sh

# 2. Per-frame MVS via SLURM array (frames spec: START:END:STEP, end-exclusive)
./scripts/slurm/launch_4d_array.sh /scratch/yubo/cow_1/<dataset> 0:1426:15

# 3. Aggregate visualization
python -m apps.reconstruction.viz.aggregate_4d_clouds \
    /scratch/yubo/cow_1/<dataset>_output/stage_a/colmap_4d/human

# 4. Undistort pass for Stage 2 inputs (unmasked images + undistorted masks)
sbatch scripts/slurm/run_undistort_unmasked.sh
sbatch scripts/slurm/run_undistort_masks.sh
```

### Masking semantics — read carefully (common confusion)

**Masks are always applied at the dense step**, regardless of sparse policy.
`dense_reconstruct.py:apply_masks` blacks out background pixels in the dense
workspace before PatchMatch, so the resulting `fused.ply` is **cow-only in
all cases**.

The choice is only about *whether SIFT/triangulation also uses the mask*:

| Policy | Sparse pts source | Effect on dense |
|---|---|---|
| `masked` | SIFT inside cow mask only | tighter per-cam depth bounds → **2× more cow pts** when it works |
| `unmasked` | SIFT on full image | wider depth bounds → fewer cow pts but always works |
| `auto` (default since 2026-04-26) | tries masked first; falls back if empty | best of both, ~10 min retry penalty per failed frame |

The `masked` policy fails when too few SIFT points survive the mask
(<100 → COLMAP can't auto-derive depth bounds → 0 fused pts). On
`9148_10581`: early frames (cow far/oblique) fail; end frames
(cow well-centered) give ~700 sparse pts and 400K–500K dense pts.

## 5. Lessons learned (do not relitigate)

### Stage 1 / classical MVS

- **Plane-sweep MVS produces 2× more points than COLMAP but they are
  visually noisier.** COLMAP MVS is the default Stage A backend.
- **Auto-derived depth bounds from camera centroid are unreliable** for
  rigs with asymmetric camera placement. Use per-frame triangulation
  (gives COLMAP per-camera bounds) or pass explicit bounds.
- **Mask sparse and dense separately.** SIFT needs the full image for
  enough matches; PatchMatch should see only the cow.
- **`--no-undistort` + `--triangulate` is incompatible** in the COLMAP
  wrapper (`triangulate_points` hardcodes PINHOLE camera model). Pre-
  undistort with cv2 if you need both.
- **SLURM `--export=ALL,FRAMES=0,5,10`** splits on value commas. Use a
  per-task frames file instead.
- **Holstein cow black fur** needs CLAHE on LAB-L for plane-sweep MVS
  (does not affect COLMAP MVS, which uses NCC on raw pixels).

### Why Stage 1 is per-frame, not temporal

**Temporal optimization at Stage A is not worth it for a deforming
subject.** The cow surface deforms, so cross-frame depth smoothing
*blurs out* exactly the deformation Stage 2 needs to capture. The
background, where temporal smoothing would help, is masked out anyway.
Stage A produces independent per-frame clouds; all temporal work
belongs in Stage 2 (canonical deformable model that learns its own
per-frame deformation field). Possible exception worth considering
later: temporally consistent SAM masks (SAM2 with track propagation)
— that's a *pre-Stage-A input* improvement, not Stage A reconstruction.

### LocalDyGS / Stage 2 — six upstream patches

LocalDyGS's vendored code needs six patches for modern toolchains and
small-rig captures. Full recipe + diffs in
`stage_b_localdygs/SETUP.md`; patches in `stage_b_localdygs/patches/`.

| # | What it fixes |
|---|---|
| 0001 | `simple_knn.cu` missing `#include <cfloat>` (modern CUDA/gcc don't transitively pull it from `<cmath>`; vanilla 3DGS's simple-knn has it, LocalDyGS's vendored fork doesn't). |
| 0002 | `dataset_readers.py` hardcodes `downsample = 2.0`. Cow data is 4K specifically not to be downscaled (fur texture). Patched to 1.0. |
| 0003 | `dataset_readers.py` hardcodes `test_num = [0, 10, 20, 30]` assuming ≥31 cams. Our 11-cam rig crashes with `IndexError` because `Colmap_Dataset.cameras = len(self.split) = 4` mismatches `len(test_exr) = 2`. Patched to bound by `len(cam_extrinsics)`. |
| 0004 | `render.py` hardcodes `os.environ['CUDA_VISIBLE_DEVICES'] = '2'`. SLURM exposes a single GPU at index 0 → CUDA init fails. Patched out. |
| 0005 | `render.py` accumulates ALL rendered tensors and GTs in lists before writing — OOMs at 4K × hundreds of views (53 GB GPU + 13 GB CPU). Patched to incremental `torchvision.utils.save_image`. |
| 0006 | Mask-aware loss: optional L1+SSIM weighted by mask before reduction. Loads `<scene>/frame<N>/masks/` if present; falls back to standard loss if not. |

Two more recipe-level lessons (also in SETUP.md):

- **Don't install `open3d` into the localdygs venv.** Wheelhouse
  `open3d-0.19.0+computecanada` pins `torch==2.6.0`, downgrading from
  the 2.10 install and ABI-breaking the CUDA extensions. Use the
  separate `~/envs/cleanply/` venv for prep (numpy + open3d + plyfile,
  no torch).
- **Pre-cache pretrained weights on a login node.** `train.py` imports
  `lpips` at module scope, which downloads VGG16 (~528 MB) from
  `download.pytorch.org`. Compute nodes have no internet — without
  pre-cache, train.py hangs ~9 min on TCP retry timeouts then crashes.
  Also: wheelhouse `lpips==0.1.4+computecanada` is missing its bundled
  `weights/v0.1/*.pth` calibration files; fetch from upstream GitHub.

### Stage 2 — what we learned from running it

- **Mask-blacked images break the loss.** Training LocalDyGS on the
  Stage 1 dense/images/ (cow-only with black background) collapses to
  "render all zeros" — the trivial minimum at PSNR ~17 against a 95%-
  black GT. Fix: train on UNMASKED images (preserves the real scene)
  and apply mask only to the loss (so gradients focus on the cow).
  This is patch 0006 + the `dense_unmasked/images/` step in Stage 1.
- **The cow's full rotation needs to be in view, not a contiguous
  short window.** A 60-frame stride-5 sample over frames 0..295 covers
  only the first ~10 s of mostly-walking motion; the cow rotates 360°
  only across the full 48 s. Stride-15 over 0..1425 (76 new frames
  on top of the existing 60) gave ~+0.5 dB test PSNR vs the narrow
  window — more than mask-aware loss alone (+0.4 dB).
- **LocalDyGS has fixed anchors** (`position_lr = 0` in basketball.py).
  The init pcd quality determines spatial coverage — anchors don't
  move during training. Build the init pcd from the union of per-frame
  fused.ply files for full motion-extent coverage.
- **No multi-GPU** in upstream LocalDyGS `train.py`. Adding DDP is a
  multi-day project (custom CUDA rasterizer doesn't trivially
  parallelize across views). For "as fast as possible" on Alliance,
  run multiple experiments in parallel on separate single-GPU jobs
  rather than parallelizing one job across GPUs.

Many of these lessons are also ingested into the personal knowledge
base at `~/github/yubo-brain/wiki/concepts/` (mask-aware-loss-3dgs,
3dgs-render-oom-pattern, temporal-rotation-coverage, etc.) so future
sessions on any device that pulls yubo-brain inherit them.

## 6. Stage 2 — LocalDyGS (chosen, built, run)

**Method**: [LocalDyGS](https://wujh2001.github.io/LocalDyGS/) (Wu et al.,
ICCV 2025). Multi-view dynamic 3DGS with seed-anchored local spaces +
static/dynamic feature decoupling. Designed for multi-view sync rigs
at large scale; closest 2025 paper to the cow problem.

**Implementation**: see `stage_b_localdygs/SETUP.md` for the full
recipe (six patches + two venvs + pretrained-weight pre-cache). Driver
chain: `prepare_localdygs_data.py` → LocalDyGS `train.py` →
`render.py` → `postprocess_render.py`.

### Why we ruled out articulated methods (durable decision)

The foundational constraint (no category-specific priors) rules out
every off-the-shelf articulated method. Recorded for posterity so a
future session doesn't relitigate this:

| Method | Category prior? | Skeleton prior? | Surface features used | Verdict |
|---|---|---|---|---|
| **SMAL** (Zuffi 2017) | yes — fixed skeleton + shape PCA over 5 species | yes (predefined) | none | category-specific by design |
| **BANMo** (Yang 2022) | claims template-free | discovers via neural blend skinning, but `# bones` is a hyperparam | uses **DensePose-CSE** in the official codebase (cat/dog/sheep CSE models exist; cattle isn't one) | **practically category-specific** in shipped code |
| **RAC** (Yang 2023) | yes — explicitly per-category | yes (per-category) | yes (CSE) | category-specific by design |
| **3D-Fauna** (Li 2024) | yes — pan-species learned prior | yes | yes | learned prior over 100+ species |
| **Lab4D** (BANMo successor) | optional | optional | can run without CSE in "no prior" mode | most flexible; revisit *after* foundational baseline is in |

LocalDyGS satisfies the foundational constraint cleanly. Output is a
deformable Gaussian representation queryable for per-point trajectories
— enough to call "Stage 2 done" and feed Stage 3 articulation
discovery.

### Results so far (cow_1/9148_10581)

Train PSNR is the *fit* metric (memorization of training cams); test
PSNR is the *generalization* metric (held-out cams 01 + 11 — at rig
extremes, hard split — see
`yubo-brain/wiki/analyses/3dgs-train-test-evaluation-semantics.md`
for why "extreme" splits read artificially low).

| Run | Frames | Loss | Train PSNR | **Test PSNR** | Notes |
|---|---|---|---|---|---|
| Plan A | 60 (stride-5 over 0..295) | standard L1+SSIM | 22.94 | 11.46 | baseline |
| Exp 3 | same 60 | mask-aware | 23.09 | 11.82 | +0.36 over Plan A |
| Exp 2 | 136 (mixed stride, full rotation) | standard | 22.49 | 11.99 | +0.53 over Plan A |
| **Exp 1** | **136** | **mask-aware** | **22.49** | **12.14** | **+0.68 (best)** ⭐ |

Both interventions help; the wider time window (full rotation) helps
more than mask-aware loss alone, and they compound. Train PSNR drops
slightly on 136-frame runs because the same 30K iters are split across
2.27× more images — less per-image memorization, better generalization
to held-out cams. That's the right direction.

### Stage 3 (deferred)

Foundational articulation discovery — cluster Stage 2's per-Gaussian
trajectories into rigid parts; fit a skeleton graph by RANSAC over
part-pair offsets. No category prior, all discovered from observed
motion. Concrete TODOs in §9. Deliberately deferred until Stage 2 is
producing something to discover from.

If LocalDyGS proves limiting downstream (e.g. for articulation
extraction), fallback methods to consider are **Dynamic 3D Gaussians**
(Luiten 3DV 2024 — explicit per-Gaussian SE(3) trajectories, possibly
more amenable to clustering) and **4D Gaussian Splatting** (Wu CVPR
2024 — HexPlane decomposition).

## 7. Repo layout — what lives where

```
apps/reconstruction/
├── PROJECT.md                       ← this file (architecture, decisions, lessons)
├── STATUS.md                        ← live state (jobs, latest checkpoints, next step)
├── README.md                        ← user-facing technical docs
├── STAGES_EXPLAINED.md              ← deep dive on Stage A internals
├── preprocess_segment_sam3.py       ← SAM3 masking (single frame OR --frames)
├── run_stage_a.py                   ← Stage A dispatcher (--backend colmap|plane_sweep)
├── run_stage_b.py                   ← NeuS Stage B driver (legacy, not used)
├── run_pipeline.py                  ← Stage A → Stage B in one shot (legacy)
├── common/                          ← shared utilities (cameras, images, IO)
├── stage_a_plane_sweep/             ← hand-rolled MVS backend (CPU)
├── stage_a_colmap/                  ← COLMAP MVS backend (default)
├── stage_a_colmap_4d/               ← per-frame loop driver — Stage 1 of this project
├── stage_b_neus/                    ← NeuS implicit surface (legacy, not used)
├── stage_b_3dgs/                    ← 3D Gaussian Splatting (one-shot, legacy)
├── stage_b_localdygs/               ← Stage 2 driver (THE active Stage 2)
│   ├── SETUP.md                     ← reproducible env-build recipe + 6 patches
│   ├── prepare_localdygs_data.py    ← Stage 1 → LocalDyGS scene layout
│   ├── postprocess_render.py        ← rename + JPEG-compress GT for review
│   ├── configs/cow_smoke.py         ← 500-iter smoke config
│   └── patches/                     ← 6 upstream patches (see §5)
├── tools/                           ← post-processing utilities (clean, mesh)
└── viz/                             ← visualization scripts
    └── aggregate_4d_clouds.py       ← color-coded union PLY for 4D output viz

scripts/slurm/                       ← SLURM job templates
├── launch_4d_array.sh               ← Stage 1 array launcher
├── run_4d_array.sh.template         ← per-frame Stage 1 task
├── run_undistort_unmasked.sh        ← produces dense_unmasked/images for Stage 2
├── run_undistort_masks.sh           ← produces dense_masks/images for mask-aware loss
├── run_localdygs_prep.sh            ← Stage 1 → Stage 2 prep (CPU job)
├── run_localdygs_smoke.sh           ← 500-iter LocalDyGS smoke test
├── run_localdygs_train.sh           ← 30K-iter LocalDyGS training
├── run_localdygs_render.sh          ← LocalDyGS render at iter N
├── run_sam3_full_array.sh           ← SAM3 mask generation
├── run_mask_qc_render.sh            ← QC video pipeline
└── run_mask_qc_encode.sh            ← QC video encode

External (per-machine; rebuild on each cluster):
~/envs/localdygs/                    ← LocalDyGS training/rendering env (torch + CUDA exts)
~/envs/cleanply/                     ← lightweight prep env (numpy + open3d + plyfile)
~/envs/3dgs/                         ← 3DGS legacy
~/envs/sam3/                         ← SAM 3 masking
~/envs/globus-cli/                   ← inter-cluster transfers
~/github/LocalDyGS/                  ← upstream clone + 6 applied patches
~/github/tiny-cuda-nn/               ← upstream clone (built at SM 8.0 for A100, 9.0 for H100)
~/github/gaussian-splatting/         ← legacy 3DGS clone
~/github/sam3/                       ← SAM 3 clone
~/github/yubo-brain/                 ← personal knowledge base (cross-cluster)

Data (Narval — being migrated to Rorqual):
/scratch/yubo/cow_1/<dataset>/                       ← input (purged after 60 days!)
/scratch/yubo/cow_1/<dataset>_output/stage_a/        ← Stage 1 outputs
/scratch/yubo/cow_1/<dataset>_output/stage_b/        ← Stage 2 outputs
```

## 8. Datasets

| ID | Path | Use |
|---|---|---|
| `cow_1/10465` | `/scratch/yubo/cow_1/10465/` | Single-frame, 11 cams @ 4K. Validated single-frame Stage A. |
| `cow_1/9148_10581` | `/scratch/yubo/cow_1/9148_10581/` | 1434-frame sequence, 30 fps, 11 cams @ 4K. Same calibration as `10465` (symlinked). The primary dataset for Stage 1 + Stage 2. Stage 1 done on 136 stride-mixed frames (60 stride-5 over 0..295 + 76 stride-15 over 300..1425). Stage 2 has four trained models (Plan A + Plan B Exp 1/2/3); see §6 results table. |

Detailed dataset properties — L-shaped 11-cam rig, non-uniform cow
motion, mask source, etc. — are documented at
`yubo-brain/wiki/entities/cow-1-9148-10581-dataset.md`.

## 9. Active TODOs

For the live, day-by-day to-do list, see `STATUS.md`. This section
tracks larger architectural items only.

### Now / near term

- [ ] **Inspect Plan B Exp 1 renders** (best model, test PSNR 12.14)
  on Mac when the in-flight render finishes. Validate the model
  visually, not just by PSNR.
- [ ] **Migration to Rorqual** (in progress — see STATUS.md). Code +
  yubo-brain already cloned; data transfers in flight; venvs built.
  Once data lands, smoke-test the full pipeline on H100.
- [ ] **Try a denser init pcd** (`--target-points 250000` instead of
  the default 90K) and see if test PSNR improves further. LocalDyGS's
  fixed-anchor design makes init density a real lever.
- [ ] **Try a "fair" test split** (test cams in middle of each rig
  edge, e.g. cam 04 + cam 09) to get a meaningful diagnostic number,
  not the artificially-low extremes split.

### Stage 3 (research follow-on, post-Stage 2)

Foundational articulation discovery — turn Stage 2's deformable Gaussians
into an explicit skeleton + skinning, with no category prior:

- [ ] Track per-point trajectories across frames (Stage 2 output).
- [ ] Cluster trajectories into rigid parts (e.g., affinity over
  relative motion; no fixed `# bones` — let the data say).
- [ ] Fit a skeleton graph over the discovered parts (RANSAC over
  part-pair offsets that stay constant → joint locations).
- [ ] Validate by re-rendering with the discovered articulation and
  comparing photometric error to the unconstrained Stage 2 output.

This is research, not engineering. Deferred until Stage 2 is producing
something to discover from. With Plan B Exp 1 in hand, that's now.

### Cleanup / hygiene (optional, when convenient)

- [ ] Rename `stage_a_colmap_4d/` → `stage_a_colmap_perframe/`. The
  current name implies temporal modeling; the implementation is
  per-frame independent. Touches multiple files + active /scratch
  paths; defer until other refactors settle.
- [ ] Retire the legacy Stage 2 dirs (`stage_b_neus/`, `stage_b_3dgs/`,
  `run_stage_b.py`, `run_pipeline.py`) once we're confident Stage 2 =
  LocalDyGS for the foreseeable future.

## 10. Constraints to remember

- **Compute**: Narval (A100-40G) currently; migrating to Rorqual
  (H100-80G) per `yubo-brain/wiki/analyses/alliance-cluster-comparison-yubo.md`.
  Both are Alliance HPC. Compute nodes have **no internet** —
  pre-stage everything on login. See per-machine `~/.claude/CLAUDE.md`
  for cluster-specific rules.
- **Accounts**: `rrg-vislearn` for GPU jobs (priority allocation),
  `def-vislearn` for CPU-only jobs (rrg-vislearn has no CPU
  membership; CPU sbatch is rejected under it).
- **Storage**: `/scratch` auto-purged after 60 days; treat as
  ephemeral. Permanent artifacts → `/project/rrg-vislearn/yubo/`
  (currently unused by this project; should be once results stabilize).
- **Cow captures must stay at native 4K resolution** — no downscaling.
  Destroys fur texture, breaks MVS. (See `yubo-brain` memory entry
  `feedback_never_downscale_cow_captures.md`.)
- **Always commit + push every cohesive change to GitHub** — Yubo's
  preference; also enables cross-device continuity (Rorqual session
  pulls latest immediately).
- **STATUS.md updates after every completed unit of work**, before
  moving on (`feedback_status_md_protocol`). A killed Claude session
  resumes cleanly from the latest STATUS.md commit.
- **For SLURM jobs in this project: estimate and submit without
  asking** (`feedback_resource_sizing_autonomy`). Overrides the global
  confirm-first rule.

The yubo-brain repo at `~/github/yubo-brain/` is the cross-device
knowledge base — durable lessons (env recipes, methodology insights,
upstream patches) get ingested there so future sessions on any device
inherit them without copy-paste.

## 11. Update log

| Date | Change | By |
|---|---|---|
| 2026-04-26 | Initial draft. Documents Stage 1 (working) + Stage 2 (open). | Claude |
| 2026-04-26 | §4 masking-semantics clarification (dense always applies masks regardless of sparse policy); add `auto` sparse policy as default; lesson on temporal-at-Stage-A being not worth it. | Claude |
| 2026-04-27 | Replace "canonical deformable" with "canonical articulated"; recommend BANMo/RAC. (REVERSED next entry.) | Claude |
| 2026-04-27 | **Reverse §6**: re-add foundational constraint (no category priors). All articulated methods use CSE etc. in shipped code, conflict with foundational. Pick LocalDyGS; defer articulation to Stage 3. | Claude |
| 2026-04-27 | LocalDyGS env built at `~/envs/localdygs/`. Two env-build lessons (§5); add `stage_b_localdygs/SETUP.md` + `patches/0001-simple_knn-include-cfloat.patch`. | Claude |
| 2026-04-27 | Add `STATUS.md` for live runtime state; PROJECT.md becomes the durable architecture/decisions doc. | Claude |
| 2026-04-28 | §2 rig correction: L-shaped 11-cam rig (not half-circle); document the cow's non-uniform rotation across the full 1434-frame sequence. | Claude |
| 2026-04-28 | Stage 2 fully implemented + first results: Plan A + Plan B Exp 1/2/3. Six patches landed (cfloat, downsample-1.0, test_num bounds, render GPU, render incremental save, mask-aware loss). Best model: Exp 1 (136fr + mask-aware) test PSNR 12.14. | Claude |
| 2026-04-28 | yubo-brain integration: bidirectional ingestion workflow + per-machine canonical install via `bin/install-claude-md.sh`. Motion-Capture lessons distilled into 8 wiki pages (entities, concepts, analyses). | Claude |
| 2026-04-28 | Onboard Rorqual (H100-80G) via yubo-brain's "Onboard a new machine" workflow; data migration via Globus in flight. PROJECT.md now references both clusters. | Claude |
| 2026-04-28 | **PROJECT.md cleanup**: refresh Stage 1 narrative (was "10-frame sweep", now 136-frame stride-mixed); promote Stage 2 from "design space" to "chosen, built, run" with results table; expand §5 lessons to all 6 patches; trim §9 TODOs (refer to STATUS.md); add §11 entry note about cleanup. | Claude |

> When you change the pipeline, the layout, or a decision: add a row here
> with the date and a one-line description of what changed.
