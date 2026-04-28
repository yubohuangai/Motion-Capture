# Cow Canonical Deformation Model — Project Overview

> Living document. **Last updated: 2026-04-27.**
> If you are an AI agent or engineer joining this project, **read this first**.
> Update the "Update log" at the bottom whenever the plan, repo layout, or
> decisions change.

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
        │ STAGE 2: canonical deformable      │  ⚠ NOT YET IMPLEMENTED
        │ model (foundational, no priors)    │  Solves a per-frame deformation
        │ (method TBD, see §6)               │  field + a canonical shape.
        │                                    │  Inputs: Stage 1 outputs.
        │ Output:                            │  Outputs: canonical
        │   canonical_cow.{ply,…}            │  representation +
        │   per_frame/<frame>.{npz,…}        │  per-frame deformation
        │                                    │  state (Gaussian positions,
        │                                    │  warp field, etc — depends
        │                                    │  on chosen method).
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
  belongs in Stage 2 (canonical deformable model that learns its own
  per-frame deformation field).
  Possible exception worth considering later: temporally consistent SAM
  masks (SAM2 with track propagation) — that's a *pre-Stage-A input*
  improvement, not Stage A reconstruction.
- **LocalDyGS env on Narval — two non-obvious patches.**
  (1) `tinycudann` `setup.py` imports `pkg_resources` → pin
  `setuptools<81` before install. (2) LocalDyGS's vendored
  `submodules/simple-knn/simple_knn.cu` uses `FLT_MAX` without
  `#include <cfloat>` → modern CUDA/gcc no longer transitively pulls it
  in. The vanilla 3DGS simple-knn fork has the include; only LocalDyGS's
  doesn't. Patch lives at
  `apps/reconstruction/stage_b_localdygs/patches/0001-simple_knn-include-cfloat.patch`.
  Full reproducible recipe: `apps/reconstruction/stage_b_localdygs/SETUP.md`.

## 6. Stage 2 — design space (foundational only)

We've **dropped explicit articulation** from the immediate Stage 2 goal
because every off-the-shelf articulated method depends on category
priors that conflict with the foundational constraint. Articulation
becomes a follow-on research direction (§9).

### Why we ruled out articulated methods

| Method | Category prior? | Skeleton prior? | Surface features used | Verdict |
|---|---|---|---|---|
| **SMAL** (Zuffi 2017) | yes — fixed skeleton + shape PCA over 5 species | yes (predefined) | none | category-specific by design |
| **BANMo** (Yang 2022) | claims template-free | discovers via neural blend skinning, but `# bones` is a hyperparam | uses **DensePose-CSE** in the official codebase (cat/dog/sheep models exist; cattle isn't one) | **practically category-specific** in shipped code |
| **RAC** (Yang 2023) | yes — explicitly per-category | yes (per-category) | yes (CSE) | category-specific by design |
| **3D-Fauna** (Li 2024) | yes — pan-species learned prior | yes | yes | learned prior over 100+ species |
| **Lab4D** (BANMo successor) | optional | optional | can run without CSE in "no prior" mode | most flexible — keep as a *future* option once foundational baseline is in |

### Foundational candidates (the actual shortlist)

| Method | What it is | Pros | Cons |
|---|---|---|---|
| **LocalDyGS** (cloned at `~/github/LocalDyGS`) | Multi-view dynamic 3DGS with seed-anchored local spaces, static-vs-dynamic feature decoupling | Designed for exactly our multi-view sync setup. Handles large motion (basketball-court scale). Repo already present. ICCV 2025. **Env now built** at `~/envs/localdygs/` (see SETUP.md). | Output is Gaussians + per-frame state, not a mesh and not a skeleton. |
| **Dynamic 3D Gaussians** (Luiten 3DV 2024) | Per-Gaussian SE(3) trajectory + physics-based local rigidity | Foundational, designed for synchronised multi-view. Per-point trajectories enable later articulation discovery. | Smaller, simpler than LocalDyGS; performance on large-motion sequences unproven on cattle. |
| **4D Gaussian Splatting** (Wu CVPR 2024) | HexPlane-based decomposition over 3DGS | Active community, fast inference. | Designed for monocular dynamic; multi-view + cow scale less proven than LocalDyGS. |

### Recommendation

**Try LocalDyGS first.** Repo is cloned, paper-data layout matches our
multi-view sync rig, Stage 1's per-frame COLMAP outputs slot into its
seed initialization. Output is a deformable Gaussian representation
that *can* render any frame and *can* be queried for per-point
trajectories — enough to call "Stage 2 done" for a foundational
baseline.

Hold **Dynamic 3D Gaussians** as a fallback if LocalDyGS env build
proves intractable on Narval (smaller method, easier env).

**Articulation is deferred to research follow-on** (§9, "Stage 3
candidate"): cluster the per-point trajectories from Stage 2 into
rigid parts and fit a skeleton graph — no priors needed, all
discovered from observed motion.

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
├── stage_b_localdygs/               ← Stage 2 driver (data-prep + training launcher)
│   ├── SETUP.md                     ← reproducible env-build recipe on Narval
│   └── patches/                     ← upstream patches we apply at install time
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
- [x] **Build LocalDyGS env on Narval** — done at `~/envs/localdygs/`.
  Two non-obvious patches needed; see Lesson #8 in §5 and the full
  recipe at `stage_b_localdygs/SETUP.md`.
- [ ] Evaluate whether per-frame stride 5 is enough vs stride 1 for Stage 2.
  Smaller stride = less per-frame deformation = easier optimization.

### Stage 3 (research follow-on, post-Stage 2)

Foundational articulation discovery — turn Stage 2's deformable Gaussians
into an explicit skeleton + skinning, with no category prior:

- [ ] Track per-point trajectories across frames (Stage 2 output)
- [ ] Cluster trajectories into rigid parts (e.g., affinity over relative
  motion; no fixed `# bones` — let the data say)
- [ ] Fit a skeleton graph over the discovered parts (RANSAC over part
  pairs that maintain constant offset → joint locations)
- [ ] Validate by re-rendering with the discovered articulation and
  comparing photometric error to the unconstrained Stage 2 output

This is research, not engineering. Deliberately deferred until Stage 2
is producing something to discover from.

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
| 2026-04-27 | Replace "canonical deformable model" with "**canonical articulated model**" throughout: more accurate for cow's skeleton-driven motion. Update Stage 2 recommendation: BANMo/RAC over LocalDyGS since LocalDyGS isn't natively articulated. | Claude |
| 2026-04-27 | **Reverse the §6 recommendation**: drop "articulated" from the immediate Stage 2 goal and re-add the *foundational* constraint (no category priors). All articulated methods (BANMo / RAC / SMAL / Lab4D) use category-specific priors (CSE features etc.) in their shipped code, conflicting with foundational. Pick **LocalDyGS** as Stage 2; defer articulation discovery to Stage 3 (cluster trajectories into rigid parts → fit skeleton, no priors). | Claude |
| 2026-04-27 | LocalDyGS env built at `~/envs/localdygs/`. Add two env-build lessons (§5): `setuptools<81` pin for tinycudann; `<cfloat>` patch for LocalDyGS's simple_knn fork. Add `stage_b_localdygs/SETUP.md` with full recipe and `stage_b_localdygs/patches/` with the simple_knn patch. | Claude |

> When you change the pipeline, the layout, or a decision: add a row here
> with the date and a one-line description of what changed.
