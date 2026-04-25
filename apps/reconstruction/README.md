# Classical + NeuS multi-view reconstruction

A self-contained, optimization-only 3D reconstruction pipeline that takes
posed, calibrated multi-view images of a static scene and produces a dense
point cloud, a Poisson mesh, and a NeuS (neural SDF) mesh. No pretrained
priors, no single-image-to-3D, no SfM — only classical MVS and per-scene
neural optimization.

The pipeline has two stages:

* **Stage A — classical MVS.** Sparse triangulation from SIFT tracks, dense
  plane-sweep MVS with per-view depth maps, depth fusion, voxel cleanup, and
  screened Poisson meshing.
* **Stage B — scene-specific NeuS.** A randomly-initialized SDF + color MLP
  trained from scratch on the observed images; the zero-level set is
  extracted with marching cubes.

Stage B uses Stage A only for a scene-bounding sphere; the networks see no
reconstructed geometry as supervision.

---

## Stage A explained — from pixels to point cloud

This section is a step-by-step walkthrough of how Stage A turns
`N` calibrated RGB images into a dense colored point cloud. Mesh
generation (Poisson) is described separately and is optional.

### What Stage A is given, what it produces

| Given | Produced |
|---|---|
| `intri.yml`, `extri.yml` — known camera intrinsics `K`, distortion `dist`, world-to-camera rotation `R`, translation `T` for every camera | `sparse.ply` — high-precision seed points (~thousands) |
| `images/<cam>/000000.jpg` — one RGB image per camera | `depth/<cam>.npz` — per-view depth + confidence maps |
| | `fused.ply` — dense oriented colored point cloud (millions of points) |

**Crucially: no depth sensor.** Depth is *computed* from the multi-view
RGB images using geometry. This is the central problem Stage A solves.

### The pipeline at a glance

```
RGB images + known camera poses
        │
        ▼
┌────────────────────────────────────────────┐
│ STEP 1 — SPARSE                            │
│   detect SIFT keypoints in each view       │
│   match features across image pairs        │
│   build multi-view tracks                  │
│   triangulate -> 3D points (sparse.ply)    │
└────────────────────────────────────────────┘
        │     (used to estimate depth ranges)
        ▼
┌────────────────────────────────────────────┐
│ STEP 2 — DENSE MVS (plane-sweep)           │
│   for every reference view:                │
│     pick neighboring source views          │
│     for each candidate depth plane:        │
│       warp source views onto reference     │
│       measure photometric similarity (NCC) │
│     pick the depth that matches best       │
│   -> per-view depth + confidence maps      │
└────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────┐
│ STEP 3 — FUSION                            │
│   back-project every depth pixel to 3D     │
│   keep only points confirmed by ≥K views   │
│   -> raw colored point cloud               │
└────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────┐
│ STEP 4 — CLEANUP                           │
│   voxel downsample (uniform spacing)       │
│   statistical outlier removal              │
│   -> fused.ply                             │
└────────────────────────────────────────────┘
```

---

### Step 1 — Sparse reconstruction (SIFT tracks → 3D points)

**Goal:** produce a small but very reliable set of 3D points. These are
needed because Step 2 has to know roughly *where* the object is in depth
in order to know which depth values to test.

1. **SIFT detection.** Each image is fed at native resolution into
   OpenCV's SIFT detector with `contrastThreshold=0.02`. **SIFT** stands
   for *Scale-Invariant Feature Transform* (Lowe, IJCV 2004) — for each
   detected keypoint it produces a 128-dimensional descriptor that is
   invariant to image scale and in-plane rotation, and robust to moderate
   changes in viewpoint and illumination. That invariance is exactly why
   the same physical point on the cow can be matched between cameras
   that see it from very different angles.

   On `cow_1/10465` at native resolution this yields **130k–260k
   keypoints per image** (e.g. cam 02 = 132k, cam 09 = 261k). Cameras
   near the middle of the half-circle arc (07–10) get the most because
   they look at the cow head-on and see the largest projected area;
   cameras at the ends (02, 11) see it more obliquely and pick up
   fewer features.

2. **Pairwise matching with Lowe's ratio test.** For every pair of cameras
   `(i, j)`, find each keypoint in `i`'s nearest two descriptors in `j`.
   Keep the match only if the best is significantly better than the
   second-best (`ratio = 0.75`). This eliminates ambiguous matches in
   repetitive textures (cow fur in particular has many self-similar
   patches).

3. **Epipolar filter (uses known calibration).** Because we already know
   the camera poses, we know that any true match must lie on a specific
   line in the other image (the *epipolar line*). We compute the
   fundamental matrix `F_ij` between each pair from `K`, `R`, `T`, and
   reject any match whose distance from its epipolar line exceeds 2 px.

   On `cow_1/10465` this kept **44 733 matches in total** across all 55
   pairs. Looking at the per-pair counts is very informative for
   diagnosing the rig:

   | pair | raw → kept | what it tells you |
   |---|---|---|
   | `07-08` | 9 734 → **8 694** | adjacent cams near the middle of the half-circle: huge overlap |
   | `03-04` | 8 154 → 4 768 | adjacent on the left side: also rich |
   | `01-07` | 6 184 → 5 457 | cam 01 sees the same surface as 07 |
   | `02-07` | 110 → **0** | opposite ends of the half-circle: no shared content |
   | `02-11` | 105 → **1** | same — confirms the rig is an arc, not a ring |
   | `01-06` | 732 → **347** | cam 01 should be *between* 6 and 7, but it matches 07 ~16× more strongly than 06 — likely a residual calibration error on cam 01 or 06 |

4. **Track building.** A keypoint that appears in views 1, 4, 7 should be
   merged into one *track*. Using union-find, we connect all pairwise
   matches into multi-view tracks. Tracks shorter than 3 views are
   discarded. Run produced **6 636 candidate tracks** with ≥3 views.

5. **Multi-view triangulation (DLT).** Each track of `m` 2D observations
   is triangulated to one 3D point by stacking projection equations and
   solving via SVD. Points that reproject with error > 2.5 px in any
   view, or that lie behind any camera, are dropped. **5 665 / 6 636**
   tracks survived (≈85% keep rate).

**Output:** `sparse.ply`, 5 665 points with median reprojection error
**1.67 px** (max 2.50 px). End-to-end runtime on a CPU: **~15 min** for
11 native-resolution images. The bulk of that is SIFT extraction +
all-pairs matching — both scale roughly with `n_pixels` and `N²`.

---

### Step 2 — Dense MVS via plane-sweep stereo

This is the heart of the pipeline and where depth is actually estimated
from RGB. The algorithm dates back to **Collins, CVPR 1996** (*"A
Space-Sweep Approach to True Multi-Image Matching"*).

**Setup.** For each *reference* view we want a depth map. We pick a small
set of *source* views (other cameras) that look at roughly the same part
of the scene. The depth map records, for each pixel, the distance from
the camera at which the scene surface lies.

#### 2a. Pick source views

For each reference camera, we score every other camera by:

* **Baseline angle** between the optical axes — must be between **5°** and
  **45°**. Too small ⇒ no depth signal. Too large ⇒ surfaces self-occlude
  and the photometric match fails.
* **Forward-axis similarity** — cameras pointing the same way see the
  same surfaces.

Up to `--max_sources` (default 4) best-scoring sources are kept. In your
half-circle rig, end-of-arc cameras (cam 11) might only get 3 sources
instead of 4 because there are no cameras "to the right" of them.

#### 2b. Estimate the depth search range

For the reference view, we project the sparse cloud (Step 1) into it and
take the 2nd–98th percentile of those depths, then pad slightly. This
gives e.g. depth range `[3.60 m, 6.60 m]` for cam 01 in your run. We
discretize this range into `--n_depths` (default 128) candidate depth
*planes*, **uniformly in inverse depth `1/d`**. (Inverse depth is the
right parameterization because errors in pixel triangulation grow with
depth, so we want finer sampling near the camera and coarser sampling
far away.)

#### 2c. The plane-sweep loop (this is the key step)

For each candidate depth `d`:

1. **Hypothesize a fronto-parallel plane** at distance `d` in front of
   the reference camera.
2. **Compute the homography** `H_ref→src` that maps the reference image
   to each source image *under the assumption that all reference pixels
   sit on this plane*.
   
   $$H = K_{src} \big( R_{src} R_{ref}^T - \tfrac{1}{d} (R_{src} R_{ref}^T \, c_{ref} - c_{src}) \, n^T \big) K_{ref}^{-1}$$
   
   where `c` is the camera center and `n` is the plane normal.
3. **Warp each source image** through `H` so it is now in the reference
   coordinate frame *as if the world were the plane at depth `d`*.
4. **Compare warped source patches to the reference patch** using
   **Zero-mean Normalized Cross-Correlation (ZNCC)** over a 7×7 window
   (`--ncc_ksize 7`). ZNCC is invariant to additive brightness and
   multiplicative contrast changes, so it tolerates exposure
   differences between cameras.
5. **Aggregate across source views** — `--aggregate top2` averages the 2
   best ZNCC scores per pixel (robust to occluded sources). `mean`
   averages all sources.
6. **Cost** for this depth = `1 − ZNCC`.

After sweeping all 128 planes, each pixel has a 1×128 cost curve. The
**best depth** is the plane with the lowest cost.

#### 2d. Confidence and rejection

A depth is only trustworthy if its cost curve is **peaky** (one clear
winner). We compute confidence as

```
confidence = (mean_cost − best_cost) / mean_cost
```

Pixels are rejected if:

* Confidence < `--min_conf` (default 0.25) — the cost curve is flat,
  the algorithm couldn't decide.
* Reference patch variance < `--texture_var_thr` (default 2.0) — a
  blank wall has no information to match. The variance is measured on a
  **CLAHE-equalised** grayscale (LAB-L, clip 3.0, 8×8 tiles), not the raw
  gray, so very dark regions (black fur, deep shadow) get the same chance
  as bright ones — without CLAHE, dark Holstein patches were below the
  variance floor and produced empty holes in the depth map. Disable with
  `use_clahe=False` in `plane_sweep` if matching a controlled-lighting
  scene where raw intensity is more informative.

Surviving depths are smoothed with a **median filter** (kills isolated
spikes) and a **joint bilateral filter** that uses the RGB image as a
guide, so depth edges align with image edges.

**Why this works at all.** If your camera poses and intrinsics are
correct, then a 3D point on the true surface reprojects to consistent
locations in all source views — so warped patches *agree*. A wrong depth
hypothesis warps source pixels from non-corresponding scene points, which
look different, so ZNCC is low.

**Output:** for each camera, `depth/<cam>.npz` containing a depth map
(meters) and a confidence map ([0,1]). E.g. cam 03 in your run produced
85.6% valid depth pixels with mean confidence 0.565.

---

### Step 3 — Fusion via cross-view consistency

A single depth map has plenty of errors (textureless regions, occluded
patches, soft shadows). Fusion uses **multiple depth maps to vote**.

For each pixel `(u, v)` in each camera with valid depth `d`:

1. **Back-project** `(u, v, d)` to a 3D world point `X`.
2. **For every other camera `j`**, project `X` into its image. If the
   projection lands inside camera `j`'s depth map and the depth there
   agrees with the depth implied by `X` (within `--rel_tol 0.02`, i.e.
   2% relative), this is a *consistent observation*.
3. **Keep the point** only if it has at least `--min_consistent` (default
   2) consistent observations in other cameras.

The 3D point is colored from the reference image and gets a normal
estimated from local depth gradients.

This is the single most important filter in Stage A. A real surface point
agrees across views; an MVS hallucination does not.

**Output:** raw fused cloud, e.g. 2.66 M points in your run.

---

### Step 4 — Cleanup

Two cheap operations remove redundancy and noise:

1. **Voxel downsample** at `--voxel_size 0.01` m (1 cm). Multiple points
   inside one voxel are merged into a single averaged point. This keeps
   density uniform and the file size manageable. (2.66 M → 2.08 M in
   your run.)
2. **Statistical outlier removal.** For each point, find the 16 nearest
   neighbors and compute the mean distance. Points whose mean distance is
   more than 2 standard deviations above the population mean are
   isolated outliers and get dropped. (2.08 M → 2.00 M in your run.)

**Final output:** `fused.ply`, an oriented colored point cloud in the
world frame defined by `extri.yml`.

---

## Is depth-from-pixels reliable? (classical vs learning-based)

You asked whether estimating depth from pixels alone is trustworthy, and
whether learning-based depth would be more accurate. Honest answer:

### What classical MVS (our pipeline) is good at

* **Pixel-accurate when textures and viewpoints are favorable.** On
  textured surfaces seen by 3+ cameras with good baselines, ZNCC plane-
  sweep delivers depth accurate to a fraction of a pixel — often better
  than learning-based methods because it uses *exact geometry* (the
  homography), not a learned prior.
* **No training data needed.** Works on any scene out of the box.
* **Failure modes are predictable:** textureless regions, repetitive
  patterns, specular highlights, occluded boundaries. The confidence
  map tells you *where* it failed.
* **Self-consistent with calibration.** If your poses are good, the
  geometry just works. If your poses are bad, no algorithm rescues you.

### Where learning-based depth wins

* **Textureless surfaces.** A painted white wall has *zero* photometric
  signal across views, so plane-sweep can't tell 2 m from 5 m. A
  learned monocular depth network like **Depth Anything**, **MiDaS**,
  or **Marigold** uses *semantic* cues (perspective, shading, object
  shape priors) and produces something plausible.
* **Single-view robustness.** Monocular depth needs only one image; MVS
  needs ≥ 2 with sufficient baseline.
* **Speed at scale.** A trained network is one forward pass per image;
  plane-sweep is `O(N · D · H · W)` per reference.

### Where learning-based depth loses

* **Scale ambiguity.** Monocular depth is up to an unknown scale + shift
  per image. To use it for 3D reconstruction across multiple cameras,
  you must align depths to your *known* metric scale, which itself
  needs MVS or sparse triangulation as anchors.
* **Geometric inconsistency.** Different views of the same surface can
  predict slightly different depths because the network has no
  cross-view constraint. Fusing them naively produces fuzzy clouds.
* **Texture biases.** Networks trained on indoor/outdoor benchmarks may
  not generalize to your domain (e.g. cattle in a barn).
* **Black-box failure.** When it's wrong, you don't know why.

### State-of-the-art combines both

Modern systems like **MVSFormer**, **GeoMVS**, or **DUSt3R** use a
neural network to *initialize and regularize* depth, then enforce
multi-view photometric and geometric constraints (essentially a
learned plane-sweep). They are more accurate than either pure approach
on hard data — but require GPU, training data, and become opaque.

### Practical guidance for your cow scene

* Your subject (cow + barn floor) is **moderately textured** — fur,
  hay, equipment. Classical plane-sweep should work *for the cow*.
* Background and uniform surfaces will be noisy. The cleanup is best
  done by **masking the object** rather than by depth refinement.
* If you want to try learning-based depth as a comparison, the cleanest
  experiment is to run **Depth Anything V2** per-view, scale-align to
  the sparse cloud, and fuse with the same Step 3/4 logic. Then visually
  diff against `fused.ply`.

In summary: **pixel-based depth is reliable on textured surfaces with
known accurate poses**, which is exactly your setup. The current
limitations of `fused.ply` are not the depth algorithm itself — they are
(a) lack of object-vs-background segmentation, and (b) cameras 02/05/11
having too few good source views due to the half-circle geometry.

## GPU requirements

| Stage | GPU required? | Notes |
|---|---|---|
| Stage A (classical MVS) | No | Pure NumPy/OpenCV, runs on CPU only. A 12-camera scene at 0.25× scale takes ~5–15 min on a modern CPU. |
| Stage B (NeuS training) | Strongly recommended | 100k iterations with MLP forward+backward is feasible on CPU but takes ~10× longer than GPU. On a single NVIDIA GPU (e.g. A100, V100, RTX 3090) expect 1–3 hours. `--device auto` picks CUDA automatically. |

On Linux servers, `--device auto` resolves to `cuda` if a GPU is available,
otherwise falls back to `cpu`.

## Inputs

Expected layout at `<data_root>`:

```
<data_root>/
  intri.yml           # OpenCV FileStorage: K_<cam>, dist_<cam>, H_<cam>, W_<cam>
  extri.yml           # OpenCV FileStorage: R_<cam>, T_<cam>  (world -> camera)
  images/
    <cam>/000000.jpg
    ...
```

Camera names in `intri.yml`'s `names` list must match the `images/<cam>/`
subfolders. Distortion may be zero (already-rectified inputs) or a standard
OpenCV `(k1,k2,p1,p2,k3)` vector.

## Environment setup

Stage B requires PyTorch. Activate the environment that has it:

```bash
conda activate cv          # or whichever env has torch installed
```

**macOS only** — OpenMP initialises twice when PyTorch and OpenCV are loaded
together. Add this prefix to every Stage B / pipeline command on macOS:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

This workaround is not needed on Linux.

## Running

All commands are run from the repo root. By default, outputs are written
to **`<data_root>_output/`** — i.e. next to the input data, never inside
the codebase. Override with `--output` if you need a different location.

All examples below use `<data_root>` as a placeholder; for the cow scene
on Narval that's `/scratch/yubo/cow_1/10465`, which produces
`/scratch/yubo/cow_1/10465_output/`.

> **Resolution policy:** images are always processed at **native
> resolution**. Downscaling has been removed — for fur/hair subjects it
> destroys the matching signal that Stage A and Stage B both rely on.

### Recommended: write to a script, then run it

Long commands break when pasted into a terminal because visual line wraps
become real newlines. Writing to a script sidesteps this:

```bash
cat > /tmp/run_neus.sh << 'EOF'
cd /path/to/Motion-Capture
python -m apps.reconstruction.run_stage_b \
  <data_root> \
  --n_iters 100000 \
  --batch_rays 512 \
  --mesh_resolution 256 \
  --device auto
EOF
bash /tmp/run_neus.sh
```

### Stage B smoke test

For end-to-end validation at low cost, reduce iterations / samples /
mesh resolution rather than image resolution:

```bash
python -m apps.reconstruction.run_stage_b <data_root> \
  --n_iters 2000 --batch_rays 256 \
  --n_samples 32 --n_importance 32 \
  --val_every 500 --ckpt_every 1000 \
  --mesh_resolution 128 --device auto
```

Check `<data_root>_output/neus/train.log` (PSNR should climb past ~15 dB
within 1k iters) and `neus/val/` for re-rendered views.

### Full Stage B run (~1–3 hours on GPU)

```bash
python -m apps.reconstruction.run_stage_b <data_root> \
  --n_iters 100000 --batch_rays 512 \
  --mesh_resolution 256 --device auto
```

On a multi-GPU machine, point to a specific card with `--device cuda:1`.
On A100/H100, raise `--batch_rays 1024` or `2048` for faster convergence.

Resume a stopped run:

```bash
python -m apps.reconstruction.run_stage_b <data_root> \
  --resume_from <data_root>_output/neus/ckpt/final.pt
```

Re-extract the mesh from a finished checkpoint without retraining:

```bash
python -m apps.reconstruction.run_stage_b <data_root> \
  --only_mesh --mesh_resolution 256 --device auto
```

### Stage A only (classical MVS, CPU)

No GPU needed. ~35–40 min at native 4K (11 cameras, 128 depth planes).

```bash
python -m apps.reconstruction.run_stage_a <data_root>
```

#### Narval / SLURM template (CPU-only)

Measured on `cow_1/10465` (11 cameras, 4K native, 128 depth planes,
4 sources/view): wall **38 min**, peak RSS **4 GB**, 8 CPUs (OpenCV
boxFilter/warpPerspective thread internally — diminishing returns past 8).

```bash
#!/bin/bash
#SBATCH --account=def-vislearn          # rrg-vislearn is GPU-only; CPU jobs use def
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G                         # 4 GB peak observed; 8 G gives 2× headroom
#SBATCH --time=1:00:00                   # ~40 min observed; 1 h covers slower nodes
#SBATCH --job-name=stageA

module load python/3.11 opencv/4.9.0
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index numpy scipy tqdm pyyaml open3d

python -m apps.reconstruction.run_stage_a /scratch/yubo/cow_1/10465
# outputs to /scratch/yubo/cow_1/10465_output/
```

### Stage A — COLMAP backend (alternative)

Uses Narval's prebuilt CUDA-enabled COLMAP module (3.12.6). PatchMatch
stereo runs on GPU; expect a single-GPU job.

```bash
#!/bin/bash
#SBATCH --account=rrg-vislearn          # GPU job
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --job-name=stageA_colmap

module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.6 \
            python/3.11 opencv/4.9.0 colmap/3.12.6
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index numpy scipy tqdm pyyaml tabulate termcolor

python -m apps.reconstruction.stage_a_colmap.run_stage_a_colmap \
  /scratch/yubo/cow_1/10465 --neighbor 6
# outputs to /scratch/yubo/cow_1/10465_output/colmap_ws/
# dense cloud at /scratch/yubo/cow_1/10465_output/colmap_ws/dense/fused.ply
```

The non-CUDA `colmap` module is also available (`module load colmap/3.12.6`
without `cuda/12.6`), but it cannot run `patch_match_stereo` — use it only
for sparse triangulation (`--skip_dense`).

### Stage B — 3DGS backend (alternative)

Trains a 3D Gaussian Splatting model against a COLMAP workspace produced
by `stage_a_colmap` (or any other COLMAP-formatted directory). Output is a
`.ply` of Gaussians + spherical-harmonic colors, **not** a triangle mesh —
use it for novel-view rendering rather than geometry.

#### One-time env build (~5 min on a MIG slice)

The CUDA submodules `simple-knn`, `diff-gaussian-rasterization`, and
`fused-ssim` need to be compiled against the installed PyTorch + CUDA. Build
once into a persistent venv at `~/envs/3dgs/`:

```bash
#!/bin/bash
#SBATCH --account=rrg-vislearn --gres=gpu:a100_1g.5gb:1
#SBATCH --cpus-per-task=8 --mem=16G --time=2:00:00 --job-name=build_3dgs

ENV_DIR=$HOME/envs/3dgs
GS_DIR=$HOME/github/gaussian-splatting   # cloned with --recursive

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11 opencv/4.9.0
[ -d "$ENV_DIR" ] || virtualenv --no-download "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install --no-index --upgrade pip wheel setuptools
pip install --no-index torch torchvision numpy plyfile tqdm opencv-python joblib

export TORCH_CUDA_ARCH_LIST="8.0"   # A100
export MAX_JOBS=4
cd "$GS_DIR"
# --no-build-isolation: setup.py imports torch; PEP 517 isolation hides it
pip install --no-build-isolation ./submodules/simple-knn
pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
pip install --no-build-isolation ./submodules/fused-ssim

python -c "import diff_gaussian_rasterization, simple_knn, fused_ssim; print('OK')"
```

If the gaussian-splatting submodule dirs are empty, run
`git submodule update --init --recursive` from the login node first.

#### Training a single frame (~30–60 min on a full A100)

```bash
#!/bin/bash
#SBATCH --account=rrg-vislearn --gres=gpu:a100:1
#SBATCH --cpus-per-task=8 --mem=32G --time=2:00:00 --job-name=cow1_3dgs

WS=/scratch/yubo/cow_1/10465_output/colmap_ws        # from stage_a_colmap
OUT=/scratch/yubo/cow_1/10465_output/3dgs

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11 opencv/4.9.0
source $HOME/envs/3dgs/bin/activate

cd $HOME/github/gaussian-splatting
python train.py -s "$WS" -m "$OUT" --iterations 7000

# Output: <OUT>/point_cloud/iteration_7000/point_cloud.ply
```

Vanilla `train.py` does not honor `colmap_ws/masks/` — Gaussians cover the
full scene including background. Filter by opacity in
`viz/vis_gaussians.py --opacity 0.5` if you only want high-confidence ones.

### Full pipeline from scratch (Stage A → Stage B)

Skips Stage A automatically if `sparse.ply` and `fused.ply` already exist.

```bash
python -m apps.reconstruction.run_pipeline <data_root> \
  --neus_iters 100000 --device auto
```

## Outputs

Defaults land under `<data_root>_output/` next to the input data, never
inside the codebase. Each backend gets its own subdirectory so multiple
implementations of the same stage can coexist.

```
<data_root>_output/
├── stage_a/
│   ├── plane_sweep/         # default for run_stage_a (hand-rolled)
│   │   ├── sparse.ply       # SIFT tracks → DLT triangulation
│   │   ├── depth/<cam>.npz  # per-view plane-sweep depth + confidence
│   │   ├── depth/<cam>.png  # turbo colormap preview
│   │   ├── fused.ply        # cross-view consistency + outlier removal
│   │   ├── mesh.ply         # screened Poisson (requires open3d)
│   │   └── config.json      # run params + summary
│   └── colmap/              # default for run_stage_a_colmap (PatchMatch)
│       ├── sparse/0/        # COLMAP sparse model
│       ├── images/  masks/  database.db
│       └── dense/
│           └── fused.ply    # stereo_fusion output
└── stage_b/
    ├── neus/                # default for run_stage_b (NeuS)
    │   ├── ckpt/iter_*.pt, final.pt
    │   ├── val/<cam>_<iter>.png
    │   ├── mesh_neus.ply
    │   └── config.json, train.log
    └── 3dgs/                # default for stage_b_3dgs.run_3dgs
        └── point_cloud/iteration_<N>/point_cloud.ply
```

All `.ply` files are in the same world coordinate frame defined by
`intri.yml` / `extri.yml`, so any artifact can be overlaid against any
other (e.g. NeuS mesh on top of plane-sweep `fused.ply`).

## Key tunables

**Stage A (`run_stage_a`):**

| Flag | Default | Notes |
|---|---|---|
| `--n_depths` | `128` | number of depth planes in the plane sweep |
| `--max_sources` | `4` | neighboring views per reference |
| `--ncc_ksize` | `7` | ZNCC window (odd, pixels) |
| `--aggregate` | `top2` | ZNCC aggregation across sources: `mean` or `top2` |
| `--min_conf` | `0.25` | peak-vs-mean confidence threshold |
| `--texture_var_thr` | `2.0` | mask low-texture reference pixels |
| `--bilateral_d` | `5` | joint bilateral depth smoothing diameter (0=off) |
| `--min_consistent` | `2` | fusion: required agreeing views |
| `--voxel_size` | `0.01` | voxel downsample size (meters) |
| `--crop_center`, `--crop_extent` | `None` | optional axis-aligned bbox in meters |

**Stage B (`run_stage_b`):**

| Flag | Default | Notes |
|---|---|---|
| `--n_iters` | `100000` | total optimization iterations |
| `--batch_rays` | `512` | rays per iter; raise to 1024–2048 on a large GPU |
| `--lr` | `5e-4` | Adam learning rate (warmup + cosine decay) |
| `--weight_eikonal` | `0.1` | gradient-magnitude regularizer |
| `--n_samples`, `--n_importance` | `64`, `64` | coarse / fine samples per ray |
| `--bound_padding` | `1.1` | radius padding around sparse cloud |
| `--mesh_resolution` | `256` | marching-cubes grid resolution |
| `--device` | `auto` | `auto`, `cuda`, `cuda:N`, `mps`, or `cpu` |
| `--only_mesh` | off | skip training, re-extract mesh from `final.pt` |
| `--resume_from` | `None` | checkpoint path to continue a stopped run |

## Dependencies

- `numpy`, `opencv-python`, `opencv-contrib-python` (for `cv2.ximgproc.jointBilateralFilter`)
- `torch` (CUDA build recommended for Stage B on Linux; MPS on Apple Silicon)
- `scipy` (kd-tree for outlier removal)
- `open3d` for Poisson meshing (Stage A) — optional; without it, Stage A still writes `fused.ply`
- `scikit-image` (preferred) or `PyMCubes` for marching cubes (Stage B)

## Package layout

```
apps/reconstruction/
  common/
    cameras.py            # Camera dataclass, load_cameras, geometry helpers
    images.py             # imread, load_views, undistort_view
    io_utils.py           # dependency-free PLY writers + timing context
  stage_a_plane_sweep/
    sparse.py             # SIFT + tracks + DLT triangulation
    mvs_plane_sweep.py    # plane-sweep ZNCC depth maps
    fuse.py               # cross-view consistency + oriented cloud
    mesh.py               # Poisson meshing (Open3D)
  stage_a_colmap/         # alternate Stage A backend — runs COLMAP MVS
    export_colmap.py      # calibration -> COLMAP workspace + sparse model
    dense_reconstruct.py  # PatchMatch stereo + stereo_fusion -> dense/fused.ply
    run_stage_a_colmap.py # one-shot driver wrapping the two scripts
  stage_b_neus/
    models.py             # SDFNetwork + ColorNetwork + learnable variance
    renderer.py           # NeuS volume renderer (unbiased alpha)
    dataset.py            # ray sampler in normalized object frame
    train.py              # training loop (photometric + Eikonal)
    extract_mesh.py       # marching cubes + vertex coloring
  stage_b_3dgs/           # alternate Stage B backend — 3D Gaussian Splatting
    run_3dgs.py           # wraps export_colmap + 3DGS train.py
  tools/                  # PLY post-processing utilities (work on any dense cloud)
    clean_pointcloud.py   # statistical/DBSCAN/radius outlier + camera-oriented normals
    pointcloud_to_mesh.py # Poisson or ball-pivoting + smoothing/decimation/color
  viz/                    # visualization helpers (Open3D windows)
    vis_colmap_sparse.py  # sparse points + camera frustums
    vis_gaussians.py      # 3DGS .ply with opacity filtering
    render_mesh_turntable.py  # turntable MP4 of a mesh.ply
  preprocess_segment_sam3.py # foreground masks via SAM 3
  run_stage_a.py
  run_stage_b.py
  run_pipeline.py         # Stage A then Stage B
```

### Choosing a backend

For Stage A you have **two backends** that produce comparable dense point
clouds; for Stage B you have **two backends** with different output types
(implicit surface vs. radiance field).

| Stage | Backend | Output | Use when |
|---|---|---|---|
| A | `stage_a_plane_sweep` | `fused.ply` (CPU plane-sweep) | Default — no extra deps; reproducible. |
| A | `stage_a_colmap` | `colmap_ws/dense/fused.ply` | Reference baseline; needs `colmap` binary + GPU. |
| B | `stage_b_neus` | `mesh_neus.ply` (clean surface) | When you want a triangle mesh. |
| B | `stage_b_3dgs` | `point_cloud.ply` (Gaussians) | When you want photorealistic novel-view rendering, not geometry. |

## Notes on the design

Scene normalization maps world coordinates into the unit sphere used by the
NeuS renderer: `x_obj = (x_world - center) / radius`. `center` / `radius`
come from the Stage A sparse cloud when available, else from the camera
centers. Stage B's mesh is transformed back to world units before writing so
it aligns with Stage A artifacts.

The SDF network uses geometric initialization (Atzmon & Lipman, SIGGRAPH '20)
so the initial field is approximately a sphere of radius 0.5 — a strong
inductive bias that makes from-scratch optimization stable even without
masks. Masks can be added later via the `weight_mask` slot in
`TrainConfig`.
