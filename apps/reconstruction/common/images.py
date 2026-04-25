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

from .cameras import Camera


def imread(path: str | Path) -> np.ndarray:
    """Read an image as BGR uint8. Raises FileNotFoundError if missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cv2 failed to decode: {p}")
    return img


def load_masks(data_root: str | Path,
               cam_names: Iterable[str],
               frame: int = 0,
               target_hw: Optional[Dict[str, Tuple[int, int]]] = None,
               ) -> Dict[str, np.ndarray]:
    """Load binary foreground masks (uint8, 0/255) if they exist.

    Looks for ``<data_root>/masks/<cam>/<frame:06d>.png``. If a mask is
    missing for a given camera, that key is omitted from the result (the
    caller should treat "no mask" as "use the whole image").

    If ``target_hw[cam]`` is given, the mask is resized (NEAREST) to match.
    """
    data_root = Path(data_root)
    if data_root.name == "images":
        data_root = data_root.parent
    masks_root = data_root / "masks"
    out: Dict[str, np.ndarray] = {}
    if not masks_root.is_dir():
        return out
    for name in cam_names:
        cand = masks_root / name / f"{frame:06d}.png"
        if not cand.exists():
            files = sorted((masks_root / name).glob("*.png"))
            files = [f for f in files if not f.name.endswith("_overlay.jpg")]
            if not files:
                continue
            cand = files[frame] if frame < len(files) else files[0]
        m = cv2.imread(str(cand), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if target_hw is not None and name in target_hw:
            H, W = target_hw[name]
            if m.shape[:2] != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        # ensure strict 0/255
        m = ((m > 127).astype(np.uint8)) * 255
        out[name] = m
    return out


def load_views(images_root: str | Path,
               cams: Dict[str, Camera],
               frame: int = 0,
               ) -> Tuple[Dict[str, np.ndarray], Dict[str, Camera]]:
    """Load one frame per camera at native resolution.

    Parameters
    ----------
    images_root : directory containing per-camera subfolders (``images/<cam>``).
        If the final path segment is not ``images``, we append it automatically.
    cams : dict of loaded Cameras (only these names will be read).
    frame : frame index; filenames are zero-padded to 6 digits (``000000.jpg``).

    Returns
    -------
    views : cam_name -> BGR uint8 image.
    cams_out : cam_name -> Camera with width/height set from the loaded image.
    """
    images_root = Path(images_root)
    if images_root.name != "images" and (images_root / "images").is_dir():
        images_root = images_root / "images"
    views: Dict[str, np.ndarray] = {}
    cams_out: Dict[str, Camera] = {}
    for name, cam in cams.items():
        cam_dir = images_root / name
        cand = cam_dir / f"{frame:06d}.jpg"
        if not cand.exists():
            files = sorted(p for p in cam_dir.glob("*.jpg"))
            if not files:
                raise FileNotFoundError(f"no jpg found under {cam_dir}")
            cand = files[frame] if frame < len(files) else files[0]
        img = imread(cand)
        H0, W0 = img.shape[:2]
        views[name] = img
        cams_out[name] = Camera(name=cam.name, K=cam.K, dist=cam.dist,
                                R=cam.R, T=cam.T, width=W0, height=H0)
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
