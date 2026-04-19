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

All commands are run from the repo root. Replace `<data_root>` and `<output>`
with your actual paths. The cow scene example uses
`/path/to/data/cow_2/7813` and `output/reconstruction_classical/cow_2_7813`.

### Recommended: write to a script, then run it

Long commands break when pasted into a terminal because visual line wraps
become real newlines. Writing to a script sidesteps this:

```bash
cat > /tmp/run_neus.sh << 'EOF'
cd /path/to/Motion-Capture
python -m apps.reconstruction_classical.run_stage_b \
  /path/to/data/cow_2/7813 \
  --stage_a_output output/reconstruction_classical/cow_2_7813 \
  --output output/reconstruction_classical/cow_2_7813/neus \
  --downscale 0.25 \
  --n_iters 100000 \
  --batch_rays 512 \
  --mesh_resolution 256 \
  --device auto
EOF
bash /tmp/run_neus.sh
```

### Quick smoke test (~5–10 min on GPU)

Validates the full Stage B pipeline end-to-end at very low resolution before
committing to a long run:

```bash
cat > /tmp/run_neus_smoke.sh << 'EOF'
cd /path/to/Motion-Capture
python -m apps.reconstruction_classical.run_stage_b \
  /path/to/data/cow_2/7813 \
  --stage_a_output output/reconstruction_classical/cow_2_7813 \
  --output output/reconstruction_classical/cow_2_7813/neus_smoke \
  --downscale 0.125 \
  --n_iters 2000 \
  --batch_rays 256 \
  --n_samples 32 \
  --n_importance 32 \
  --val_every 500 \
  --ckpt_every 1000 \
  --mesh_resolution 128 \
  --device auto
EOF
bash /tmp/run_neus_smoke.sh
```

Check `neus_smoke/train.log` (PSNR should climb past ~15 dB within 1k iters)
and `neus_smoke/val/` for re-rendered views.

### Full Stage B run (~1–3 hours on GPU)

```bash
cat > /tmp/run_neus_full.sh << 'EOF'
cd /path/to/Motion-Capture
python -m apps.reconstruction_classical.run_stage_b \
  /path/to/data/cow_2/7813 \
  --stage_a_output output/reconstruction_classical/cow_2_7813 \
  --output output/reconstruction_classical/cow_2_7813/neus \
  --downscale 0.25 \
  --n_iters 100000 \
  --batch_rays 512 \
  --mesh_resolution 256 \
  --device auto
EOF
bash /tmp/run_neus_full.sh
```

On a multi-GPU machine you can point to a specific card:
`--device cuda:1`

On a powerful GPU (A100, H100) raise `--batch_rays 1024` or `2048` for faster
convergence.

Resume a stopped run:

```bash
python -m apps.reconstruction_classical.run_stage_b ... \
  --resume_from output/reconstruction_classical/cow_2_7813/neus/ckpt/final.pt
```

Re-extract the mesh from a finished checkpoint without retraining:

```bash
python -m apps.reconstruction_classical.run_stage_b \
  /path/to/data/cow_2/7813 \
  --stage_a_output output/reconstruction_classical/cow_2_7813 \
  --output output/reconstruction_classical/cow_2_7813/neus \
  --only_mesh --mesh_resolution 256 --device auto
```

### Stage A only (classical MVS, CPU)

No GPU needed. Typical runtime ~5–15 min at 0.25× scale.

```bash
python -m apps.reconstruction_classical.run_stage_a \
  /path/to/data/cow_2/7813 \
  --output output/reconstruction_classical/cow_2_7813 \
  --downscale 0.25
```

### Full pipeline from scratch (Stage A → Stage B)

Skips Stage A automatically if `sparse.ply` and `fused.ply` already exist.

```bash
cat > /tmp/run_pipeline.sh << 'EOF'
cd /path/to/Motion-Capture
python -m apps.reconstruction_classical.run_pipeline \
  /path/to/data/cow_2/7813 \
  --output output/reconstruction_classical/cow_2_7813 \
  --downscale 0.25 \
  --neus_iters 100000 \
  --device auto
EOF
bash /tmp/run_pipeline.sh
```

## Outputs

```
<output>/
  sparse.ply               # Stage A: triangulated SIFT tracks
  depth/<cam>.npz          # Stage A: per-view depth + confidence
  depth/<cam>.png          # turbo-colormapped preview
  fused.ply                # Stage A: dense oriented + colored point cloud
  mesh.ply                 # Stage A: Poisson mesh (requires open3d)
  config.json              # Stage A run parameters + summary
  neus/
    ckpt/iter_*.pt, final.pt
    val/<cam>_<iter>.png   # periodic re-renderings of training views
    mesh_neus.ply          # Stage B final marching-cubes mesh (world coords)
    config.json, train.log
```

All `.ply` files are in the same world coordinate frame defined by
`intri.yml` / `extri.yml`, so the NeuS mesh can be overlaid directly on the
Stage A fused cloud or mesh.

## Key tunables

**Stage A (`run_stage_a`):**

| Flag | Default | Notes |
|---|---|---|
| `--downscale` | `0.25` | MVS image scale; smaller = faster, less detail |
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
apps/reconstruction_classical/
  common/
    cameras.py            # Camera dataclass, load_cameras, geometry helpers
    images.py             # imread, load_views, undistort_view
    io_utils.py           # dependency-free PLY writers + timing context
  stage_a_classical/
    sparse.py             # SIFT + tracks + DLT triangulation
    mvs_plane_sweep.py    # plane-sweep ZNCC depth maps
    fuse.py               # cross-view consistency + oriented cloud
    mesh.py               # Poisson meshing (Open3D)
  stage_b_neus/
    models.py             # SDFNetwork + ColorNetwork + learnable variance
    renderer.py           # NeuS volume renderer (unbiased alpha)
    dataset.py            # ray sampler in normalized object frame
    train.py              # training loop (photometric + Eikonal)
    extract_mesh.py       # marching cubes + vertex coloring
  run_stage_a.py
  run_stage_b.py
  run_pipeline.py         # Stage A then Stage B
```

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
