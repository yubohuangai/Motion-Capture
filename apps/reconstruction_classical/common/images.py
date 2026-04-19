"""Image loading helpers shared across reconstruction stages.

All pipeline code operates on BGR uint8 (cv2 convention) unless it explicitly
converts to float. Downsampling returns a resized image plus a matching
:class:`Camera` with scaled intrinsics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .cameras import Camera, scale_camera


def imread(path: str | Path) -> np.ndarray:
    """Read an image as BGR uint8. Raises FileNotFoundError if missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cv2 failed to decode: {p}")
    return img


def load_views(images_root: str | Path,
               cams: Dict[str, Camera],
               frame: int = 0,
               downscale: float = 1.0,
               ) -> Tuple[Dict[str, np.ndarray], Dict[str, Camera]]:
    """Load one frame per camera and optionally downscale.

    Parameters
    ----------
    images_root : directory containing per-camera subfolders (``images/<cam>``).
        If the final path segment is not ``images``, we append it automatically.
    cams : dict of loaded Cameras (only these names will be read).
    frame : frame index; filenames are zero-padded to 6 digits (``000000.jpg``).
    downscale : factor <=1. 1 keeps native resolution; 0.5 halves each side.

    Returns
    -------
    views : cam_name -> BGR uint8 image.
    cams_scaled : cam_name -> Camera with updated (K, W, H).
    """
    images_root = Path(images_root)
    if images_root.name != "images" and (images_root / "images").is_dir():
        images_root = images_root / "images"
    views: Dict[str, np.ndarray] = {}
    cams_out: Dict[str, Camera] = {}
    for name, cam in cams.items():
        cam_dir = images_root / name
        # accept either zero-padded frame index or any sorted file if the
        # caller points us at a single-frame directory.
        cand = cam_dir / f"{frame:06d}.jpg"
        if not cand.exists():
            files = sorted(p for p in cam_dir.glob("*.jpg"))
            if not files:
                raise FileNotFoundError(f"no jpg found under {cam_dir}")
            cand = files[frame] if frame < len(files) else files[0]
        img = imread(cand)
        if downscale != 1.0:
            H0, W0 = img.shape[:2]
            new_w = max(1, int(round(W0 * downscale)))
            new_h = max(1, int(round(H0 * downscale)))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cam_s = scale_camera(cam.__class__(name=cam.name, K=cam.K, dist=cam.dist,
                                               R=cam.R, T=cam.T,
                                               width=W0, height=H0),
                                 sx=new_w / W0, sy=new_h / H0)
        else:
            H0, W0 = img.shape[:2]
            cam_s = Camera(name=cam.name, K=cam.K, dist=cam.dist,
                           R=cam.R, T=cam.T, width=W0, height=H0)
        views[name] = img
        cams_out[name] = cam_s
    return views, cams_out


def undistort_view(img: np.ndarray, cam: Camera) -> Tuple[np.ndarray, Camera]:
    """Undistort an image and return it alongside a zero-distortion Camera."""
    if np.allclose(cam.dist, 0.0):
        return img, cam
    new_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(cam.K, cam.dist,
                                                     (cam.width, cam.height), 0)
    und = cv2.undistort(img, cam.K, cam.dist, None, new_cam_mtx)
    new_cam = Camera(name=cam.name, K=new_cam_mtx,
                     dist=np.zeros(5), R=cam.R, T=cam.T,
                     width=cam.width, height=cam.height)
    return und, new_cam


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
