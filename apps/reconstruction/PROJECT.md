# Cow Canonical Deformation Model — Project Overview

> Living document. **Last updated: 2026-04-26.**
> If you are an AI agent or engineer joining this project, **read this first**.
> Update the "Update log" at the bottom whenever the plan, repo layout, or
> decisions change.

---

## 1. North Star

Build a **canonical deformable model** of a captured cow:
> a rest-pose 3D shape  +  per-frame deformation field
> that together explain every observed video frame from every camera.

A canonical model is what lets you *re-pose* the cow, *retarget* its motion,
*compare body shape across animals*, and downstream — fit body-condition
scoring, weight estimation, gait analysis. None of that is possible from a
sequence of independent per-frame point clouds.

## 2. Why this is hard (and why we don't have it yet)

1. **Half-circle rig**: at any instant, only one side of the cow is observed.
   Static reconstruction is fundamentally incomplete per frame.
2. **The cow deforms**: legs, head, neck, breathing — non-rigid. Naive
   "stack all per-frame clouds" doesn't merge into one coherent surface
   because corresponding points have moved.
3. **Classical MVS has no notion of time**: each frame is reconstructed
   independently. To leverage temporal info you need a *deformation model*,
   which classical methods don't provide.
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
        │ STAGE 1: per-frame dense MVS       │  N independent dense clouds
        │ stage_a_colmap_4d                  │  Each is partial (one side of cow)
        │   sparse:  full image (no mask)    │  Sparse must be unmasked: tight
        │   dense:   masked, cow-only        │  cow mask leaves <100 SIFT pts,
        │                                    │  breaks COLMAP depth bounds.
        │ Output: <out>/human/               │  No temporal info leveraged.
        │   aggregated_4d.ply (color-coded)  │
        │   frame_<NNNNNN>_fused.ply         │
        │   frame_<NNNNNN>_sparse.ply        │
        │   frame_<NNNNNN>_thumb.jpg         │
        └─────────────────┬──────────────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │ STAGE 2: canonical model           │  ⚠ NOT YET IMPLEMENTED
        │ (method to be picked, see §6)      │  Solves the deformation field.
        │                                    │  Inputs: Stage 1 outputs +
        │                                    │  optionally SMAL prior.
        │ Output:                            │  Outputs: canonical mesh +
        │   canonical_cow.{obj,ply}          │  per-frame SE3/skinning weights
        │   deformation_field/<frame>.npz    │  + a renderer.
        └────────────────────────────────────┘
```

## 4. Stage 1 — current state

**Status: working** end-to-end on `cow_1/9148_10581` (10-frame stride-5 sweep).

### Inputs
- `<data_root>/intri.yml`, `<data_root>/extri.yml` — EasyMocap calibration
- `<data_root>/images/<cam>/<frame:06d>.jpg` — multi-view sequence
- `<data_root>/masks/<cam>/<frame:06d>.png` — produced by SAM3 (run first)

### Driver
`apps/reconstruction/stage_a_colmap_4d/run_stage_a_colmap_4d.py`

Loops over frames; per-frame calls the proven single-frame
`stage_a_colmap.run_stage_a_colmap` with `--no-mask-sparse` (essential —
see §5 lesson 3).

### Output layout (flat, single rsync target)
```
<data>_output/stage_a/colmap_4d/
├── human/                            ← scp/rsync THIS dir
│   ├── aggregated_4d.ply             (color-coded union of all frames)
│   ├── frame_<NNNNNN>_fused.ply      (per-frame dense cloud)
│   ├── frame_<NNNNNN>_sparse.ply     (per-frame SIFT 3D points)
│   └── frame_<NNNNNN>_thumb.jpg      (one input view as visual reference)
└── work/                             ← machine intermediates, deletable
    └── frame_<NNNNNN>/{sparse, dense, images, masks, database.db}
```

### Resource numbers (cow_1, 11 cams @ 4K, single A100, 10 frames stride 5)
- SAM3 masking (sequence): ~5 min wall (one job, model loaded once)
- Stage 1 per frame: ~10 min wall (full A100, mostly PatchMatch)
- 10-frame sweep via SLURM array: **~12 min wall** (parallel, all on full A100s)
- Per-frame point counts at this scale: **150K–250K** masked-cow points

### Failure modes (all hit and fixed during development)
1. **Wrong depth bounds** → fusion gets 0 points despite full depth maps
   *Fix*: per-frame triangulation gives COLMAP per-camera bounds.
2. **Tight mask kills sparse triangulation** → too few SIFT pts →
   "no sparse model in workspace" error.
   *Fix*: `--no-mask-sparse` (default in 4D driver).
3. **Comma-in-value SLURM env var** → array tasks 1..N silently fail.
   *Fix*: launcher writes a per-task frames file.
4. **Empty-array reduction crash** in dense reconstruction stats reporter.
   *Fix*: guard with `if n_pts > 0`.

### How to run today
```bash
# 1. Mask the sequence (once per data set + frame stride)
sbatch /scratch/yubo/jobs/run_sam3_seq_<dataset>.sh

# 2. After SAM3 finishes, launch parallel per-frame MVS
./scripts/slurm/launch_4d_array.sh /scratch/yubo/cow_1/<dataset> 0:50:5

# 3. After array completes, build aggregated PLY
python -m apps.reconstruction.viz.aggregate_4d_clouds \
    /scratch/yubo/cow_1/<dataset>_output/stage_a/colmap_4d/human
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
| `auto` (default) | tries masked first; falls back if empty | best of both, ~10 min retry penalty per failed frame |

The `masked` policy fails when too few SIFT points survive the mask
(<100 → COLMAP can't auto-derive depth bounds → 0 fused pts). On the
`9148_10581` sequence: early frames (cow far/oblique) fail, end frames
(cow well-centered) give ~700 sparse pts and 400K–500K dense pts.

`auto` is the default since 2026-04-26 — get the dense lift when possible,
graceful fallback otherwise.

## 5. Lessons learned (do not relitigate)

- **Plane-sweep MVS produces 2× more points than COLMAP but they are
  visually noisier.** COLMAP MVS is the default Stage A backend.
- **Auto-derived depth bounds from camera centroid are unreliable** for
  rigs with asymmetric camera placement. Use per-frame triangulation
  (which gives COLMAP per-camera bounds) or pass explicit bounds.
- **Mask sparse and dense separately.** SIFT needs the full image for
  enough matches; PatchMatch should see only the cow.
- **`--no-undistort` + `--triangulate` is incompatible** in the COLMAP
  wrapper (`triangulate_points` hardcodes PINHOLE camera model). If you
  need both, pre-undistort with cv2 or rewrite the rewrite step.
- **`--export=ALL,FRAMES=0,5,10`** in SLURM splits on the value commas.
  Use a file.
- **3DGS auto-downscales to 1.6K** unless `--resolution 1`. Honor the
  no-downscale rule for cow data.
- **Holstein cow black fur** needs CLAHE on LAB-L for plane-sweep MVS
  (does not affect COLMAP MVS, which uses NCC on raw pixels).
- **Temporal optimization at Stage A is not worth it for a deforming
  subject.** The cow surface deforms, so cross-frame depth smoothing
  *blurs out* exactly the deformation Stage 2 needs to capture. The
  background, where temporal smoothing would help, is masked out anyway.
  Stage A produces independent per-frame clouds; all temporal work
  belongs in Stage 2 (canonical model with explicit deformation field).
  Possible exception worth considering later: temporally consistent SAM
  masks (SAM2 with track propagation) — that's a *pre-Stage-A input*
  improvement, not Stage A reconstruction.

## 6. Stage 2 — design space (open question)

Three candidates, pick one based on Stage 1 quality + lab priorities.

| Method | What it is | Pros | Cons |
|---|---|---|---|
| **BANMo / RAC / Lab4D** | Articulated neural body: skeleton + skinning + canonical SDF, fit to images via differentiable rendering | Animal-specific (cats/dogs/horses already shown). Outputs reusable canonical + skinning weights. Skeleton enables intuitive editing/retargeting. | Paper expects monocular casual video; multi-view + known calibration needs adaptation. Training is multi-day. |
| **LocalDyGS** (cloned at `~/github/LocalDyGS`) | Multi-view dynamic 3DGS with seed-anchored local spaces, static-vs-dynamic feature decoupling | Designed for our exact setup (multi-view sync). Handles large motion. Repo present. | Output is Gaussians, not a mesh; canonical "rest pose" isn't a first-class concept. Env build is risky on Narval (tinycudann). |
| **Deformable 3DGS** | Per-Gaussian time-conditioned deformation MLP over a single static set | Simpler than LocalDyGS. Active community. | Designed for monocular video. Adapting multi-view + cow scale unproven. |

**Recommendation when we get there**: try LocalDyGS first since (a) the
repo is already cloned, (b) the data layout matches our multi-view rig,
(c) Stage 1's COLMAP per-frame outputs slot directly into LocalDyGS's
seed initialization. BANMo is the "real" answer for an articulated cow
model but is more research and less infrastructure.

## 7. Repo layout — what lives where

```
apps/reconstruction/
├── PROJECT.md                       ← this file
├── README.md                        ← user-facing technical docs
├── STAGES_EXPLAINED.md              ← deep dive on Stage A internals
├── preprocess_segment_sam3.py       ← SAM3 masking (single frame OR --frames)
├── run_stage_a.py                   ← Stage A dispatcher (--backend colmap|plane_sweep)
├── run_stage_b.py                   ← NeuS Stage B driver
├── run_pipeline.py                  ← Stage A → Stage B in one shot
├── common/                          ← shared utilities (cameras, images, IO)
├── stage_a_plane_sweep/             ← hand-rolled MVS backend (CPU)
├── stage_a_colmap/                  ← COLMAP MVS backend (default)
├── stage_a_colmap_4d/               ← per-frame loop driver — Stage 1 of this project
├── stage_b_neus/                    ← NeuS implicit surface
├── stage_b_3dgs/                    ← 3D Gaussian Splatting (one-shot, not 4D)
├── tools/                           ← post-processing utilities (clean, mesh)
└── viz/                             ← visualization scripts
    └── aggregate_4d_clouds.py       ← combined color-coded PLY for 4D output

scripts/slurm/                       ← SLURM job templates
├── run_4d_array.sh.template
└── launch_4d_array.sh

External:
~/envs/3dgs/                         ← persistent venv for 3D Gaussian Splatting
~/envs/sam3/                         ← persistent venv for SAM 3
~/envs/cleanply/                     ← lightweight venv: numpy + open3d + plyfile
~/github/gaussian-splatting/         ← cloned 3DGS repo
~/github/LocalDyGS/                  ← cloned 4D Gaussian repo (Stage 2 candidate)
~/github/sam3/                       ← cloned SAM 3 repo
/scratch/yubo/cow_1/<dataset>/       ← input data (purged after 60 days!)
/scratch/yubo/cow_1/<dataset>_output/ ← outputs (also purged)
```

## 8. Datasets

| ID | Path | Description |
|---|---|---|
| `cow_1/10465` | `/scratch/yubo/cow_1/10465/` | Single-frame, 11 cams @ 4K. Used to validate single-frame Stage A. |
| `cow_1/9148_10581` | `/scratch/yubo/cow_1/9148_10581/` | 1434-frame sequence, 30 fps, 11 cams @ 4K. Same calibration as `10465` (symlinked). Used for Stage 1 sequence work. |

## 9. Active TODOs (descending priority)

- [ ] **Visual review** of the 10-frame `cow_1/9148_10581` sweep on Mac:
  do per-frame masked clouds look clean? Does the aggregated PLY show the
  cow at distinct poses?
- [ ] If Stage 1 looks good: scale up to 60-frame sweep (`0:300:5`) so the
  cow makes a more substantial rotation through the rig.
- [ ] **Decide Stage 2 method** (§6). Likely needs an offline meeting with
  Yubo before committing.
- [ ] If LocalDyGS picked: build the env on Narval. Known risks: tinycudann,
  mmcv 1.6.0, the differing diff-gaussian-rasterization fork.
- [ ] Evaluate whether per-frame stride 5 is enough vs stride 1 for Stage 2.
  Smaller stride = less per-frame deformation = easier registration.

## 10. Constraints to remember

- Compute is **Narval** (Alliance HPC). Compute nodes have **no internet**.
  Pre-stage everything on login.
- Account: `rrg-vislearn` for GPU jobs, `def-vislearn` for CPU-only jobs.
- `/scratch` is auto-purged after 60 days. Treat as ephemeral. Permanent
  artifacts go to `/project/rrg-vislearn/yubo/` (not yet used by this
  project — should be when results stabilize).
- Cow captures must be processed at **native 4K resolution** (no
  downscaling — destroys fur texture, breaks MVS).
- Always commit + push every cohesive change to GitHub (per Yubo's
  preference — this is automated guidance for AI agents in the project).

## 11. Update log

| Date | Change | By |
|---|---|---|
| 2026-04-26 | Initial draft. Documents Stage 1 (working) + Stage 2 (open). | Claude (under Yubo's direction) |
| 2026-04-26 | Add §4 masking-semantics clarification: dense always applies masks regardless of sparse policy. Add `auto` sparse policy as default. Add lesson on temporal-at-Stage-A being not worth it. | Claude |

> When you change the pipeline, the layout, or a decision: add a row here
> with the date and a one-line description of what changed.
