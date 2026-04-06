# Multi-view object reconstruction pipeline

This document describes the **foundation motion capture** object-reconstruction path under `apps/reconstruction/`. It targets **calibrated multi-camera RGB** rigs (e.g. EasyMocap-style `intri.yml` / `extri.yml`), **foreground masks** (required before COLMAP export for reliable object-focused reconstruction), **COLMAP** export and dense MVS, and **3D Gaussian Splatting** training.

---

## Goals and assumptions

- **Input:** Synchronized images per camera (`data/images/<cam>/<frame>.jpg`), camera intrinsics and extrinsics (YAML), background-only frames for `generate_masks.py`, and per-view masks under `data/masks/<cam>/` before export.
- **Not assumed:** Depth sensors. Dense geometry comes from COLMAP MVS and/or neural methods (3DGS).
- **Output:** A COLMAP workspace consumable by 3DGS / Nerfstudio-style tools; optional dense `fused.ply`; trained Gaussian splat model.

---

## Directory layout (scripts)

| Script | Role |
|--------|------|
| `generate_masks.py` | Foreground masks (background subtraction or SAM). |
| `export_colmap.py` | Calibration → COLMAP (defaults: `<data>/colmap_ws`, undistort, `masks/`, triangulate, GPU, Open3D viewer; `--no_*` to disable). |
| `dense_reconstruct.py` | COLMAP dense MVS with **spatial-neighbor** `patch-match.cfg`. |
| `run_3dgs.py` | Wrapper: calls `export_colmap.py` then external `gaussian-splatting/train.py`. |
| `vis_colmap_sparse.py` | Open3D: sparse points + camera frustums. |
| `vis_gaussians.py` | Open3D: 3DGS `point_cloud.ply` (optional opacity filter). |

---

## End-to-end workflow

### 1. Masks (do this first)

Required for a clean object-only COLMAP / 3DGS path: reduces background features and wrong matches.

```bash
python apps/reconstruction/generate_masks.py /path/to/data \
  --frame 0 --bg_frame 0 --bg_data /path/to/background \
  --threshold 50 --center_only --vis
```

**Outputs:** `data/masks/<cam>/000000.png` (and by default `mask_vis/` and `foreground_images/`).

**Notes:** Background subtraction has limits when lighting or scene content changes; SAM is an alternative (`--mode sam`).

---

### 2. COLMAP workspace + sparse points (`export_colmap.py`)

Reads calibration paths (can be absolute or under `data/`):

```bash
# Defaults: <data>/colmap_ws, frame 0, undistort, masks, triangulate, GPU, sparse viewer
python apps/reconstruction/export_colmap.py /path/to/data

# Custom intrinsics/extrinsics paths and workspace (still uses other defaults):
python apps/reconstruction/export_colmap.py /path/to/data \
  --intri /path/to/intri_colmap_ba.yml \
  --extri /path/to/extri_colmap_ba.yml \
  -o /path/to/colmap_ws
```

**What it does:**

1. Builds **PINHOLE** cameras (by default: undistorted images and intrinsics; `--no_undistort` keeps OpenCV distortion model).
2. Copies images as `01.jpg`, `02.jpg`, … and masks as `masks/01.jpg.png`, … (skip masks with `--no_mask`).
3. Writes `sparse/0/` with **your** poses and intrinsics (COLMAP IDs aligned after feature extraction).
4. By default runs triangulation (`--no_triangulate` to skip):
   - `colmap feature_extractor` (optional `--ImageReader.mask_path`).
   - `colmap exhaustive_matcher`.
   - **Prune matches** to **K-nearest cameras in 3D** (default K = 6): camera center \(C = -R^\top t\) from extrinsics; only neighbor image pairs are kept in the SQLite DB before `point_triangulator`.

**Why neighbor pruning:** On a **ring** of cameras, opposite views can see **visually similar** object sides (e.g. repeated packaging). Matching every pair encourages wrong correspondences. Restricting pairs to **geometrically nearby** cameras uses extrinsics as a prior consistent with “cameras 01…N arranged around the object.”

By default, `export_colmap.py` opens the Open3D sparse viewer after export (same as `vis_colmap_sparse.py`). Use **`--no_vis`** on headless runs. Requires `open3d` and a display. Optional: `--vis_frustum_scale`, `--vis_point_size`.

**Outputs:**

```
colmap_ws/
  images/
  masks/
  sparse/0/   cameras.*, images.*, points3D.*
  database.db
```

---

### 3. Dense COLMAP (optional) (`dense_reconstruct.py`)

Produces a dense colored point cloud (`dense/fused.ply`). Stereo pairs are **not** hard-coded by filename order: **K nearest camera centers** from `sparse/0/images.bin` define `patch-match.cfg`.

```bash
python apps/reconstruction/dense_reconstruct.py /path/to/colmap_ws \
  --neighbor 6 --mask --min_num_pixels 2
```

**Steps:** `image_undistorter` → optional mask black-out in `dense/images/` → write `patch-match.cfg` → `patch_match_stereo` → `stereo_fusion`.

**Caveats:** MVS favors **many overlapping views** with favorable baselines. A **small ring of elevated** cameras often reconstructs **top + partial sides** well; **occluded** faces (e.g. bottom on the table) remain invisible. Dense quality is separate from 3DGS.

---

### 4. 3D Gaussian Splatting (`run_3dgs.py` or manual `train.py`)

```bash
python apps/reconstruction/run_3dgs.py /path/to/data \
  --intri ... --extri ... \
  --frame 0 --output /path/to/out \
  --undistort --mask masks --colmap_gpu \
  --gs_repo /path/to/gaussian-splatting \
  --gpu 0
```

**What 3DGS uses from the COLMAP folder:**

- **Cameras:** `sparse/0/cameras.bin`, `images.bin`.
- **Images:** `images/` (training RGB).
- **Initialization:** `sparse/0/points3D.*` (sparse by default; you may copy `dense/fused.ply` → `sparse/0/points3D.ply` for a denser init—verify compatibility and re-train).

**Masks:** Not applied inside vanilla `train.py` unless you **black out** backgrounds in `images/` or use a fork that supports mask loss. Ghost / low-opacity Gaussians can be filtered when visualizing (`vis_gaussians.py --opacity`).

---

### 5. Visualization

**Sparse COLMAP model (cameras + points):**

```bash
python apps/reconstruction/vis_colmap_sparse.py /path/to/colmap_ws/sparse/0
```

**3DGS Gaussians (headful machine):**

```bash
python apps/reconstruction/vis_gaussians.py /path/to/point_cloud.ply --opacity 0.5
```

**Training renders** (`train/ours_*/renders/`) mostly match training viewpoints; for **novel-view** verification use `render.py` in the 3DGS repo or rotate the point cloud in Open3D.

---

## External dependencies

- **COLMAP** (`colmap` on PATH).
- **OpenCV**, **NumPy**; repo **EasyMocap** modules for COLMAP I/O (`easymocap.mytools.colmap_structure`, etc.).
- **Open3D** + **plyfile** for visualization scripts.
- **3DGS:** [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) (CUDA build, separate env often easiest).

---

## Configuration knobs (summary)

| Knob | Location | Effect |
|------|----------|--------|
| Neighbor count K (sparse) | `export_colmap.py` → `_camera_neighbor_pairs(..., k=6)` | How many nearest cameras per view for feature matching after exhaustive match. |
| Neighbor count K (dense) | `dense_reconstruct.py --neighbor` | Sources per reference image in `patch-match.cfg`. |
| Masks | `export_colmap.py --mask` / `--no_mask`, `dense_reconstruct.py --mask` | COLMAP feature extraction + dense image black-out. |
| Fusion density | `dense_reconstruct.py --min_num_pixels`, `--max_reproj_error` | More points vs. stricter consistency. |

---

## Known limitations

1. **Invisible geometry:** No multi-view method recovers surfaces **never** seen (e.g. box bottom on table).
2. **Repeated texture:** Neighbor matching **reduces** front/back confusion on a ring; it does not remove ambiguity if neighbors still see aliased patterns.
3. **COLMAP MVS vs. 3DGS:** MVS is patch-based stereo; 3DGS optimizes a radiance field—often **better** for sparse rings but can look “messy” as a raw point cloud without opacity filtering.
4. **`_inject_neighbor_matches`:** Depends on COLMAP’s SQLite `pair_id` encoding; if you use a very different COLMAP version, verify match pruning still behaves as expected.

---

## Quick reference: minimal server flow

```bash
# 1) Masks
python apps/reconstruction/generate_masks.py $DATA --frame 0 --bg_frame 0 \
  --bg_data $BG --threshold 50 --center_only

# 2) COLMAP + triangulation (spatial neighbors)
python apps/reconstruction/export_colmap.py $DATA \
  --intri $INTRI --extri $EXTRI -o $OUT/colmap --no_vis

# 3) Dense (optional)
python apps/reconstruction/dense_reconstruct.py $OUT/colmap --neighbor 6 --mask

# 4) 3DGS
cd $GS_REPO && python train.py -s $OUT/colmap -m $OUT/model --iterations 7000
```

---

*Last updated to match `apps/reconstruction` as of the spatial-neighbor matching and `dense_reconstruct.py` behavior.*
