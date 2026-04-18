# Foundation Motion Capture Pipeline — Technical Report & Workflow Summary

This document serves as a **technical report**, **code review overview**, and **onboarding guide** for the multi-view markerless motion capture pipeline. It describes the end-to-end workflow from camera calibration through 3D human body reconstruction and **arbitrary object reconstruction**.

---

## Audience

- **Mentors**: High-level code review and pipeline architecture overview
- **Collaborators**: Project introduction and integration points
- **New interns**: Step-by-step guidance to run and understand the pipeline

---

## Pipeline Overview

The pipeline consists of four main stages: (1) Camera calibration → (2) 2D pose estimation → (3) 3D human reconstruction → (4) 3D object reconstruction. Outputs flow as intri.yml/extri.yml → annots/*.json → keypoints3d, smpl/ (human) or COLMAP workspace → Gaussians/mesh (object).

**Input**: Multi-view synchronized images  
**Output**: SMPL-X body model parameters (human) or 3D Gaussians / mesh (arbitrary objects)

---

## Multi-camera capture session layout (`raw`)

Phone (or other) captures are organized as one **session folder** (often named after the subject or shoot), with a **`raw`** subfolder that holds **one directory per camera**. Folders use **two-digit ids**: `01`, `02`, …, `11`.

Example (eleven cameras):

```
cow_1_board/                          # session / dataset name (arbitrary)
└── raw/                              # session root passed to sync.py, move_unmatched.py, etc.
    ├── 01/
    │   ├── VID/
    │   │   └── VID_20260414_141256.mp4
    │   ├── 20260414_141256.csv       # per-frame timestamps for this camera’s clip
    │   └── images/                   # extracted frames (filled by sync --extract)
    ├── 02/
    │   ├── VID/
    │   ├── <stem>.csv
    │   └── images/
    └── ...
```

**Per camera (`NN/`, e.g. `01/`):**

| Path | Role |
|------|------|
| **`VID/`** | Recorded video(s). The main clip is typically `VID_<dateparts>.mp4`. Backups such as `*_ori.mp4` are ignored by discovery. |
| **`<stem>.csv`** | Text file with **one timestamp per line**, in order, **one line per frame** in the paired `VID_*.mp4`. The filename stem matches the video: e.g. `VID_20260414_141256.mp4` ↔ `20260414_141256.csv` in the **parent** of `VID/` (same folder as `images/`). Used when extracting frames and renaming them to timestamp-based filenames. |
| **`images/`** | Extracted frames. After `scripts/preprocess/sync.py --extract`, each file is usually **renamed to its timestamp** from the CSV (so **filenames are not identical across cameras** for the same instant). |

**Cross-camera alignment** for tools like **`move_unmatched.py`** is **not** “same filename in every `images/`”. Sync uses the per-phone timestamp lists to build a **pairing table** (saved as **`matched.csv`**) under this repo: **`output/exp/<session_slug>_<threshold>/matched.csv`** (see `scripts/preprocess/sync.py`). Each row is one synchronized moment; column `i` is the image **stem** for camera `i` in order `01`, `02`, ….

Preview utilities such as **`scripts/postprocess/multiview_grid_video.py`** only need the folder layout above; they pair frames by **sorted order** inside each `images/` (no sync files).

**Session slug** for `output/exp/…` is usually the folder **above** `raw` (e.g. `.../cow_1_board/raw` → `cow_1_board`), unless overridden when running sync.

### Copy a multi-view clip (`scripts/preprocess/copy_multiview_clip.py`)

Copies a **contiguous** frame range from every `raw/NN/images/` folder into a new dataset root. Output frames are **renumbered** from `000000` under `<output>/images/NN/`. Indices follow **natural sort** of filenames in `01/images/` (same idea as `multiview_grid_video.py`).

**Required:** `--input` (session `raw` root), `--output` (destination root), `--frame-start`, `--frame-end`. Both frame bounds are **1-based and inclusive**.

```bash
python scripts/preprocess/copy_multiview_clip.py \
  --input /home/yubo/scratch/cow_1_board/raw/ \
  --output /home/yubo/scratch/cow_1_board/board \
  --frame-start 686 --frame-end 2653
```

Short flags:

```bash
python scripts/preprocess/copy_multiview_clip.py \
  -i ~/scratch/cow_2_board/raw \
  -o ~/scratch/cow_2_board/board \
  --frame-start 1525 --frame-end 3361
```

**Optional:** `--camera-workers N` (default `0` = auto: prefer 8 / 4 / 2 / 1 parallel cameras), `--copy-mode copyfile` (default, faster) or `copy2`, `--progress auto|camera|overall|none`.

---

## Stage 1: Camera Calibration

Camera calibration produces intrinsic (`intri.yml`) and extrinsic (`extri.yml`) parameters. These define each camera’s lens model and pose in a shared world coordinate system.

### 1.1 Data Preparation

Per-camera images with a chessboard visible from various angles.

Expected layout:

```
<board_data>/
├── images/
│   ├── 01/           # camera 01
│   │   ├── 000000.jpg
│   │   └── ...
│   ├── 02/
│   └── ...
└── ...
```

### 1.2 Chessboard Detection

Detect and store chessboard corners for all calibration images:

```bash
python apps/calibration/detect_calibration_board.py --mode chessboard <path> --pattern 9,6 --grid 0.111
```

(`detect_chessboard.py` is a thin shim that sets `--mode chessboard`.)

- `--pattern 9,6`: Inner corners (columns × rows)
- `--grid 0.111`: Grid size in meters
- Output: `chessboard/<cam>/*.json` and visualization in `output/calibration/`

### 1.3 Intrinsic Calibration

Calibrate each camera’s focal length and distortion:

```bash
python apps/calibration/calib_intri.py <intri_data>
```

- Uses OpenCV `calibrateCamera` with chessboard 3D–2D correspondences
- Output: `output/intri.yml`

### 1.4 Extrinsic Calibration (Stereo)

Estimate camera poses in a common world frame using stereo calibration:

```bash
python apps/calibration/calib_extri.py <extri_data> --stereo --intri <intri_data>/output/intri.yml
```

- `--stereo`: Uses stereo calibration between adjacent cameras for more robust extrinsics
- Output: `intri.yml`, `extri.yml` in the extrinsic data directory

### 1.5 Bundle Adjustment (COLMAP)

Refine intrinsics and extrinsics with COLMAP’s bundle adjuster:

```bash
python apps/calibration/chessboard_ba_colmap.py <path> \
  --intri intri.yml --extri extri.yml \
  --out_intri intri_colmap_ba.yml --out_extri extri_colmap_ba.yml
```

- Injects chessboard corners into a COLMAP sparse model and runs `colmap bundle_adjuster`
- Output: `intri_colmap_ba.yml`, `extri_colmap_ba.yml`, and optimized 3D points

**Note**: Use the BA-refined yml files as the final calibration for the motion capture stage. Copy `intri_colmap_ba.yml` → `intri.yml` and `extri_colmap_ba.yml` → `extri.yml` into the mocap data path (or symlink them).

---

## Stage 2: 2D Pose Estimation

Extract 2D keypoints from each view for multi-view triangulation.

### 2.1 Extract Keypoints

```bash
python apps/preprocess/extract_keypoints.py <path> --mode mmpose
```

- `--mode mmpose`: Uses MMPose (RTMPose whole-body) for body + hand + face keypoints
- Output: `annots/<cam>/*.json` with 2D keypoints and confidence scores

Other modes: `openpose`, `hrnet`, `mp-holistic`, etc. See `extract_keypoints.py` config.

---

## Stage 3: 3D Human Reconstruction

Triangulate 2D keypoints and fit an SMPL-X body model.

### 3.1 Run the Pipeline

```bash
python apps/demo/mv1p.py <path> \
  --body bodyhandface \
  --model smplx \
  --gender male \
  --vis_det \
  --vis_repro \
  --vis_smpl
```

**Key arguments**:

| Argument      | Description                                      |
|---------------|--------------------------------------------------|
| `--body`      | Keypoint set: `bodyhandface`, `body25`, `total`  |
| `--model`     | Body model: `smplx`, `smpl`, `smplh`             |
| `--gender`    | `male`, `female`, or `neutral`                    |
| `--vis_det`   | Visualize 2D detections                          |
| `--vis_repro` | Visualize 3D reprojection                        |
| `--vis_smpl`  | Visualize fitted mesh overlay                    |

**Inputs expected**:

- `images/<cam>/*.jpg` — synchronized frames
- `annots/<cam>/*.json` — 2D keypoints
- `intri.yml`, `extri.yml` — camera parameters (or BA-refined versions)

**Outputs**:

- `keypoints3d/` — triangulated 3D skeleton
- `output/smpl/` — SMPL-X parameters (pose, shape) per frame
- Optional: `output/vertices/` if `--write_vertices` is set

### 3.2 Pipeline Internals

1. **Triangulation**: Multi-view DLT from 2D keypoints to 3D
2. **Reprojection check**: Outlier rejection based on reprojection error
3. **SMPL fitting**: Optimization of pose and shape to match 3D keypoints and 2D reprojection

---

## Stage 4: 3D Object Reconstruction

Reconstruct arbitrary objects (no predefined body model) from calibrated multi-view images using neural surface reconstruction or 3D Gaussian Splatting.

### 4.1 Export Calibration to COLMAP Format

Convert `intri.yml` / `extri.yml` to a COLMAP sparse model that standard reconstruction frameworks accept:

```bash
python apps/reconstruction/export_colmap.py <data_path> \
  --frame 0 --output <colmap_workspace> --undistort --mask masks
```

- `--undistort`: undistorts images and exports PINHOLE cameras (recommended for neural methods)
- `--mask masks`: copies foreground masks from `<data_path>/masks/` (run `generate_masks.py` in section 4.2 first)
- Output: `<colmap_workspace>/images/`, `<colmap_workspace>/sparse/0/{cameras,images,points3D}.{bin,txt}`

### 4.2 Foreground masks (required for object reconstruction)

Generate masks before COLMAP export and any downstream step that expects masked images. They exclude background features and stabilize multi-view geometry.

**Typical command** (run from the Motion-Capture repo root). Defaults: **hybrid** mode (background subtraction → SAM box prompt → post-SAM single blob), `frame=0`, `bg_frame=0`, `threshold=30`, SAM checkpoint at `data/sam/sam_vit_h_4b8939.pth`. Also writes `mask_vis/` and `foreground_images/` unless disabled.

```bash
python apps/reconstruction/generate_masks.py /mnt/yubo/obj/cube \
  --bg_data /mnt/yubo/obj/background
```

Replace `/mnt/yubo/obj/cube` with your object data root (`images/<cam>/`, `intri.yml`, `extri.yml`) and `--bg_data` with the folder that holds the same per-camera layout for **background-only** captures. Masks are written to `<data_path>/masks/<cam>/000000.png`.

Other modes (override `--mode` as needed):

```bash
# Background subtraction only (no SAM)
python apps/reconstruction/generate_masks.py <data_path> \
  --mode bg_sub --bg_data <background_root> \
  --frame 0 --bg_frame 0 --threshold 30

# Full-image SAM only (requires segment-anything)
python apps/reconstruction/generate_masks.py <data_path> \
  --mode sam --sam_checkpoint /path/to/sam_vit_h.pth
```

See `python apps/reconstruction/generate_masks.py --help` for hybrid options (`--hybrid_combine intersect`, `--no_post_sam_center_only`, SAM2, etc.).

### 4.3 3D Gaussian Splatting (Real-time Rendering)

Fast reconstruction producing a real-time renderable Gaussian splat:

```bash
# Install (one-time)
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
pip install submodules/diff-gaussian-rasterization submodules/simple-knn

# Train
python train.py -s <colmap_workspace> --iterations 7000
```

Or use the integrated wrapper for single/batch reconstruction:

```bash
python apps/reconstruction/run_3dgs.py <data_path> \
  --frame 0 --output <recon_path> --undistort \
  --gs_repo /path/to/gaussian-splatting
```

### 4.4 Mesh Extraction via Nerfstudio NeuS-facto

High-quality watertight mesh from multi-view images:

```bash
pip install nerfstudio

ns-train neus-facto --data <colmap_workspace> colmap

ns-export poisson --load-config outputs/.../config.yml --output-dir meshes/
```

---

## Complete Workflow Summary

```bash
# 1. Camera calibration
python apps/calibration/detect_calibration_board.py --mode chessboard <intri_path> --pattern 9,6 --grid 0.111
python apps/calibration/calib_intri.py <intri_path>
python apps/calibration/detect_calibration_board.py --mode chessboard <extri_path> --pattern 9,6 --grid 0.111
python apps/calibration/calib_extri.py <extri_path> --stereo --intri <intri_path>/output/intri.yml
python apps/calibration/chessboard_ba_colmap.py <extri_path>

# 2. 2D pose estimation (human pipeline)
python apps/preprocess/extract_keypoints.py <mocap_path> --mode mmpose

# 3. 3D human reconstruction
python apps/demo/mv1p.py <mocap_path> --body bodyhandface --model smplx --gender male --vis_det --vis_repro --vis_smpl

# 4. 3D object reconstruction (arbitrary objects)
python apps/reconstruction/export_colmap.py <data_path> --frame 0 --output <colmap_ws> --undistort
python apps/reconstruction/generate_masks.py <data_path> --bg_data <background_root>  # required; see section 4.2
python apps/reconstruction/run_3dgs.py <data_path> --frame 0 --output <recon> --gs_repo <gs_path> --undistort
```

---

## Project Structure (Key Paths)

```
Motion-Capture/
├── apps/
│   ├── calibration/           # Stage 1
│   │   ├── detect_calibration_board.py
│   │   ├── detect_chessboard.py   # shim → detect_calibration_board --mode chessboard
│   │   ├── detect_charuco.py      # shim → detect_calibration_board --mode charuco
│   │   ├── calib_intri.py
│   │   ├── calib_extri.py
│   │   ├── chessboard_ba_colmap.py
│   │   └── vis_chess_sfm_ba.py
│   ├── preprocess/            # Stage 2
│   │   └── extract_keypoints.py
│   ├── demo/                  # Stage 3
│   │   └── mv1p.py
│   └── reconstruction/        # Stage 4
│       ├── export_colmap.py
│       ├── generate_masks.py
│       └── run_3dgs.py
├── easymocap/
│   ├── dataset/               # Data loaders
│   ├── pyfitting/             # SMPL optimization
│   ├── smplmodel/              # Body model loading
│   └── mytools/                # Camera utils, COLMAP I/O
└── data/models/               # Pretrained weights (mmpose, SMPL-X, etc.)
```

---

## Installation

See the [EasyMocap documentation](https://chingswy.github.io/easymocap-public-doc/install/install.html) for setup. Key dependencies:

- Python 3.8+
- PyTorch, OpenCV
- MMPose (for `--mode mmpose`)
- COLMAP (for `chessboard_ba_colmap.py`)
- SMPL-X model files (from [SMPL-X](https://github.com/vchoutas/smplx))

For Stage 4 (object reconstruction):

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) (clone + install submodules)
- [Nerfstudio](https://docs.nerf.studio/) (`pip install nerfstudio`) for NeuS-facto mesh extraction
- [Segment Anything](https://github.com/facebookresearch/segment-anything) for SAM-based masking (default hybrid path in `generate_masks.py`)

---

## Acknowledgements

This project builds on [EasyMocap](https://github.com/zju3dv/EasyMocap) by the 3D Vision Group at Zhejiang University. It integrates:

- SMPL-X body model ([vchoutas/smplx](https://github.com/vchoutas/smplx))
- MMPose / RTMPose for 2D pose estimation
- COLMAP for bundle adjustment
