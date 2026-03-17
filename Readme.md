# Motion Capture Pipeline — Technical Report & Workflow Summary

This document serves as a **technical report**, **code review overview**, and **onboarding guide** for the multi-view markerless human motion capture pipeline. It describes the end-to-end workflow from camera calibration through 3D body reconstruction.

---

## Audience

- **Mentors**: High-level code review and pipeline architecture overview
- **Collaborators**: Project introduction and integration points
- **New interns**: Step-by-step guidance to run and understand the pipeline

---

## Pipeline Overview

The pipeline consists of three main stages: (1) Camera calibration → (2) 2D pose estimation → (3) 3D reconstruction. Outputs flow as intri.yml/extri.yml → annots/*.json → keypoints3d, smpl/.

**Input**: Multi-view synchronized images  
**Output**: SMPL-X body model parameters (pose, shape) per frame

---

## Stage 1: Camera Calibration

Camera calibration produces intrinsic (`intri.yml`) and extrinsic (`extri.yml`) parameters. These define each camera’s lens model and pose in a shared world coordinate system.

### 1.1 Data Preparation

- **Intrinsic data**: Per-camera images with a chessboard visible from various angles.
- **Extrinsic data**: Synchronized frames from all cameras with the chessboard visible in a shared pose (e.g., on the floor).

Expected layout:

```
<intri_data>/
├── images/
│   ├── 01/           # camera 01
│   │   ├── 000000.jpg
│   │   └── ...
│   ├── 02/
│   └── ...
└── ...

<extri_data>/
├── images/
│   ├── 01/
│   ├── 02/
│   └── ...
└── ...
```

### 1.2 Chessboard Detection

Detect and store chessboard corners for all calibration images:

```bash
python apps/calibration/detect_chessboard.py <path> --pattern 9,6 --grid 0.111
```

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

## Stage 3: 3D Reconstruction

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

## Complete Workflow Summary

```bash
# 1. Camera calibration
python apps/calibration/detect_chessboard.py <intri_path> --pattern 9,6 --grid 0.111
python apps/calibration/calib_intri.py <intri_path>
python apps/calibration/detect_chessboard.py <extri_path> --pattern 9,6 --grid 0.111
python apps/calibration/calib_extri.py <extri_path> --stereo --intri <intri_path>/output/intri.yml
python apps/calibration/chessboard_ba_colmap.py <extri_path>

# 2. 2D pose estimation
python apps/preprocess/extract_keypoints.py <mocap_path> --mode mmpose

# 3. 3D reconstruction
python apps/demo/mv1p.py <mocap_path> --body bodyhandface --model smplx --gender male --vis_det --vis_repro --vis_smpl
```

---

## Project Structure (Key Paths)

```
Motion-Capture/
├── apps/
│   ├── calibration/           # Stage 1
│   │   ├── detect_chessboard.py
│   │   ├── calib_intri.py
│   │   ├── calib_extri.py
│   │   ├── chessboard_ba_colmap.py
│   │   └── vis_chess_sfm_ba.py
│   ├── preprocess/            # Stage 2
│   │   └── extract_keypoints.py
│   └── demo/                  # Stage 3
│       └── mv1p.py
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

---

## Acknowledgements

This project builds on [EasyMocap](https://github.com/zju3dv/EasyMocap) by the 3D Vision Group at Zhejiang University. It integrates:

- SMPL-X body model ([vchoutas/smplx](https://github.com/vchoutas/smplx))
- MMPose / RTMPose for 2D pose estimation
- COLMAP for bundle adjustment
