"""Posed multi-view image dataset for NeuS optimisation.

The dataset owns:

* The 11 RGB images (as float tensors in [0, 1], BGR->RGB).
* Per-pixel rays in the *normalised object frame* (unit-sphere bounded).
* Uniform random sampling of (rays_o, rays_d, rgb_gt) batches for training.

World -> object transform
-------------------------
We need to map the scene into the unit sphere that the NeuS renderer assumes.
Given a scene ``center`` (3,) and ``radius`` (scalar) in world units,

    x_obj = (x_world - center) / radius

so any world point within ``radius`` of ``center`` lands inside the unit sphere.
A ray's direction scales the same way but is renormalised to keep it a unit
vector — the t-values in object space therefore differ from world units, which
is fine because the renderer only uses ``rays_o`` and ``rays_d`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from ..common.cameras import Camera
from ..common.images import load_views, undistort_view


# ---------------------------------------------------------------------------
# Scene normalisation helpers
# ---------------------------------------------------------------------------


def scene_bounds_from_points(points: np.ndarray,
                             padding: float = 1.1,
                             robust_pct: float = 1.0) -> Tuple[np.ndarray, float]:
    """Derive (center, radius) from a sparse point cloud.

    The robust_pct percentile clips obvious outliers (e.g. lonely points from
    bad triangulation) before taking the bounding sphere.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    lo = np.percentile(pts, robust_pct, axis=0)
    hi = np.percentile(pts, 100.0 - robust_pct, axis=0)
    center = 0.5 * (lo + hi)
    # choose radius as distance from center to the furthest kept point
    keep = np.all((pts >= lo) & (pts <= hi), axis=1)
    kept = pts[keep] if keep.any() else pts
    radius = float(np.linalg.norm(kept - center, axis=1).max()) * padding
    return center.astype(np.float32), radius


def scene_bounds_from_cameras(cams: Dict[str, Camera],
                              padding: float = 1.2) -> Tuple[np.ndarray, float]:
    """Fallback normalisation when no sparse cloud is available."""
    C = np.stack([c.center for c in cams.values()], axis=0)
    center = C.mean(axis=0)
    radius = float(np.linalg.norm(C - center, axis=1).max()) * padding
    return center.astype(np.float32), radius


# ---------------------------------------------------------------------------
# Ray generation
# ---------------------------------------------------------------------------


def _pixel_grid(H: int, W: int) -> np.ndarray:
    """Return an (H, W, 2) array of (u, v) pixel-centre coordinates."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float32) + 0.5,
                       np.arange(H, dtype=np.float32) + 0.5, indexing="xy")
    return np.stack([u, v], axis=-1)


def rays_for_camera(cam: Camera) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rays_o_world, rays_d_world) for every pixel of the camera.

    The caller is expected to have undistorted the image first; we only use
    ``cam.K`` here.
    """
    H, W = cam.height, cam.width
    uv = _pixel_grid(H, W)                           # (H, W, 2)
    uvh = np.concatenate([uv, np.ones_like(uv[..., :1])], axis=-1)   # (H, W, 3)
    K_inv = np.linalg.inv(cam.K).astype(np.float32)
    dirs_cam = uvh @ K_inv.T                         # (H, W, 3), z=1
    # rotate into world: world = R^T @ dir_cam
    dirs_world = dirs_cam @ cam.R                    # (uses R^T with this convention)
    dirs_world = dirs_world / (np.linalg.norm(dirs_world, axis=-1, keepdims=True) + 1e-12)
    origin = cam.center.astype(np.float32)
    rays_o = np.broadcast_to(origin, dirs_world.shape).copy()
    return rays_o, dirs_world.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class ViewData:
    name: str
    cam: Camera
    rgb: np.ndarray            # (H, W, 3) float32 in [0, 1], RGB order
    rays_o: np.ndarray         # (H, W, 3) origins in object frame
    rays_d: np.ndarray         # (H, W, 3) unit directions in object frame


class NeuSDataset:
    """Multi-view image dataset with pre-computed rays in object coordinates.

    Parameters
    ----------
    data_root : path containing ``images/<cam>/<frame>.jpg`` and
        ``intri.yml`` / ``extri.yml`` (same layout Stage A expects).
    cams      : the calibrated cameras.
    frame     : frame index to load (one frame per camera).
    downscale : factor <=1 applied to both images and intrinsics.
    scene_center, scene_radius : world -> object transform (see module doc).
    device    : torch device for the cached tensors (CPU is fine, GPU optional).
    """

    def __init__(self,
                 data_root: str | Path,
                 cams: Dict[str, Camera],
                 scene_center: np.ndarray,
                 scene_radius: float,
                 frame: int = 0,
                 downscale: float = 1.0,
                 device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.scene_center = np.asarray(scene_center, dtype=np.float32).reshape(3)
        self.scene_radius = float(scene_radius)

        views_imgs, cams_scaled = load_views(data_root, cams, frame=frame, downscale=downscale)
        self.views: List[ViewData] = []
        for name, img_bgr in views_imgs.items():
            cam = cams_scaled[name]
            # Undistort so a single pinhole model suffices for ray generation
            img_u, cam_u = undistort_view(img_bgr, cam)
            rgb = cv2.cvtColor(img_u, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rays_o_w, rays_d_w = rays_for_camera(cam_u)
            rays_o_obj = (rays_o_w - self.scene_center) / self.scene_radius
            # Isotropic scaling doesn't change unit ray directions.
            rays_d_obj = rays_d_w
            self.views.append(ViewData(
                name=name, cam=cam_u, rgb=rgb,
                rays_o=rays_o_obj.astype(np.float32),
                rays_d=rays_d_obj.astype(np.float32),
            ))

        # Stack into flat tensors for random indexing
        self._stack_tensors()

    def _stack_tensors(self) -> None:
        rgbs = np.concatenate([v.rgb.reshape(-1, 3) for v in self.views], axis=0)
        ros = np.concatenate([v.rays_o.reshape(-1, 3) for v in self.views], axis=0)
        rds = np.concatenate([v.rays_d.reshape(-1, 3) for v in self.views], axis=0)
        self._rgb = torch.from_numpy(rgbs).to(self.device)
        self._ro = torch.from_numpy(ros).to(self.device)
        self._rd = torch.from_numpy(rds).to(self.device)
        self._n_pixels = self._rgb.shape[0]
        # Per-view slice offsets (used by :meth:`rays_for_view`)
        self._view_offsets: List[Tuple[int, int, int, int]] = []
        off = 0
        for v in self.views:
            h, w = v.rgb.shape[:2]
            self._view_offsets.append((off, off + h * w, h, w))
            off += h * w

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @property
    def n_views(self) -> int:
        return len(self.views)

    @property
    def n_pixels(self) -> int:
        return self._n_pixels

    def sample_rays(self, n_rays: int, generator: torch.Generator | None = None
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw ``n_rays`` uniformly from the full image pool.

        Returns (rays_o, rays_d, rgb_gt) on ``self.device``.
        """
        idx = torch.randint(0, self._n_pixels, (n_rays,),
                            generator=generator, device=self.device)
        return self._ro[idx], self._rd[idx], self._rgb[idx]

    def rays_for_view(self, view_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """Return all rays for a single view (for full-image evaluation)."""
        s, e, H, W = self._view_offsets[view_idx]
        return self._ro[s:e], self._rd[s:e], self._rgb[s:e], (H, W)

    def view_name(self, view_idx: int) -> str:
        return self.views[view_idx].name

    # ------------------------------------------------------------------
    # Utility: world <-> object transforms (for export)
    # ------------------------------------------------------------------

    def world_to_object(self, pts_world: np.ndarray) -> np.ndarray:
        return (np.asarray(pts_world) - self.scene_center) / self.scene_radius

    def object_to_world(self, pts_obj: np.ndarray) -> np.ndarray:
        return np.asarray(pts_obj) * self.scene_radius + self.scene_center
