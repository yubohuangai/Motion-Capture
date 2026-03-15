# Why the 3D View Appears Upside-Down

## Root Cause: OpenCV vs Open3D Coordinate Conventions

The pipeline (calib_extri, chessboard_ba_colmap, triangulation) uses **OpenCV's coordinate system**, while the visualization (vis_chess_sfm_ba) uses **Open3D**, which follows **OpenGL's convention**. These differ in the Y axis:

| System | X | Y | Z (camera looks along) |
|--------|---|---|------------------------|
| **OpenCV** | right | **down** | into scene (+Z) |
| **OpenGL / Open3D** | right | **up** | into/out of scene |

When Open3D renders coordinates defined in OpenCV (Y-down), the scene appears upside-down because "up" in your data is "down" on screen.

---

## Coordinate Flow Through the Pipeline

### 1. Chessboard 3D Definition (detect_chessboard, annotator/chessboard)

- `getChessboard3d(axis='yx')` creates points in the board's object frame.
- The comment "标定板z轴朝上" (calibration board z-axis up) indicates the board lies in XY with Z as the normal.
- This is consistent with OpenCV’s pinhole model and image conventions.

### 2. Calibration (calib_extri.py)

- `cv2.solvePnP(k3d, k2d, K, dist)` returns R, t in OpenCV convention:
  - **Xc = R @ Xw + T** (world to camera)
  - Camera X right, Y down, Z into scene
- `relative2world` propagates extrinsics in the same convention.

### 3. Triangulation (chessboard_ba_colmap.py)

```python
# triangulate_pair, line 184-185
P0 = np.hstack([cam0["R"], cam0["T"]])  # world-to-camera
Xh = cv2.triangulatePoints(P0, P1, uv0_u, uv1_u)
```

- `cv2.triangulatePoints` expects projection matrices in OpenCV convention.
- 3D points X are in the **same world frame** as the cameras: OpenCV (Y-down).

### 4. COLMAP (chessboard_ba_colmap.py)

- Uses model `OPENCV` and stores R, T from the same world frame.
- `read_back_colmap_model` converts qvec/tvec back to R, T without changing conventions.
- The whole BA pipeline stays in OpenCV coordinates.

### 5. Origin Alignment (align_world_to_camera)

- Sets camera 01 as the origin: its frame becomes the world frame.
- Camera 01’s frame in OpenCV has **Y down**, so the world frame still has **Y down**.
- This does not change the upside-down behavior.

### 6. Visualization (vis_chess_sfm_ba.py)

- Uses Open3D’s GUI and rendering (OpenGL-style, **Y up**).
- Points and cameras are passed directly in OpenCV world coordinates.
- Open3D treats the data as if it were in its own frame, so the scene looks upside-down.

---

## Summary

| Component | Convention | Correct? |
|-----------|------------|----------|
| calib_extri | OpenCV (Y down) | ✓ |
| chessboard_ba_colmap | OpenCV (Y down) | ✓ |
| Triangulation | OpenCV (Y down) | ✓ |
| Pinhole model (Xc = R@Xw + T) | OpenCV | ✓ |
| **vis_chess_sfm_ba (Open3D)** | Assumes OpenGL (Y up) | ✗ Mismatch |

The triangulation and pinhole model are correct. The issue is only in the visualization step: Open3D expects Y-up data, but receives Y-down.

---

## Fix

Apply a coordinate transform in the visualization to convert OpenCV → OpenGL-style before passing to Open3D:

- Flip Y: `y' = -y`
- Optionally flip Z if the view direction feels wrong: `z' = -z`

The minimal fix is a Y-flip for all 3D points and camera poses before visualization.
