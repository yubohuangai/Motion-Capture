"""Depth-map fusion and Poisson meshing.

The fusion step filters each per-view depth map by cross-view consistency:
a pixel's 3D point is accepted only if, when projected into at least
``min_consistent`` other views, their depth maps agree (within a relative
tolerance). Accepted points are colored from the reference view and given
an oriented normal estimated from the depth's local gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..common.cameras import Camera
from .mvs_plane_sweep import DepthMap


# ---------------------------------------------------------------------------
# Back-projection and normals from a single depth map
# ---------------------------------------------------------------------------


def _pixel_grid(H: int, W: int) -> np.ndarray:
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    return np.stack([uu, vv], axis=-1)              # (H, W, 2)


def backproject_depth(dm: DepthMap, cam: Camera) -> Tuple[np.ndarray, np.ndarray]:
    """Return (points_world (N,3), pixel_indices (N,2)) for valid pixels."""
    H, W = dm.depth.shape
    uv = _pixel_grid(H, W)
    mask = dm.depth > 0
    uv_sel = uv[mask]
    d_sel = dm.depth[mask]
    X = cam.backproject(uv_sel.reshape(-1, 2), d_sel.reshape(-1))
    return X.astype(np.float32), uv_sel.astype(np.int32)


def estimate_normals(dm: DepthMap, cam: Camera,
                     min_depth_ratio: float = 0.02,
                     ) -> np.ndarray:
    """Estimate per-pixel surface normals from central depth differences.

    Returns (H, W, 3) float32 in the world frame. Invalid pixels get (0,0,0).
    """
    H, W = dm.depth.shape
    depth = dm.depth
    # central-difference neighbours, with validity from depth>0
    d_x = np.zeros_like(depth); d_y = np.zeros_like(depth)
    d_x[:, 1:-1] = 0.5 * (depth[:, 2:] - depth[:, :-2])
    d_y[1:-1, :] = 0.5 * (depth[2:, :] - depth[:-2, :])

    uv = _pixel_grid(H, W)
    K = cam.K
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # tangent vectors in camera frame
    X = (uv[..., 0] - cx) * depth / fx
    Y = (uv[..., 1] - cy) * depth / fy
    Z = depth
    # derivatives w.r.t. u, v
    dX_du = (d_x + depth) / fx                       # approx d(X)/d(u)
    dZ_du = d_x
    dY_du = (uv[..., 1] - cy) * d_x / fy
    dX_dv = (uv[..., 0] - cx) * d_y / fx
    dY_dv = (d_y + depth) / fy
    dZ_dv = d_y

    n_cam = np.stack([
        dY_du * dZ_dv - dZ_du * dY_dv,
        dZ_du * dX_dv - dX_du * dZ_dv,
        dX_du * dY_dv - dY_du * dX_dv,
    ], axis=-1)
    norm = np.linalg.norm(n_cam, axis=-1, keepdims=True) + 1e-12
    n_cam = n_cam / norm
    # orient normals toward the camera (so they face "out of the surface")
    sign = -np.sign(np.sum(n_cam * np.stack([X, Y, Z], axis=-1), axis=-1, keepdims=True))
    n_cam = n_cam * np.where(sign == 0, 1.0, sign)
    # transform to world frame: n_world = R^T n_cam
    n_world = n_cam @ cam.R                          # (H, W, 3): row-vec @ R gives (R^T n)^T
    # invalidate borders and tiny-depth gradients
    valid = depth > 0
    valid[:2] = False; valid[-2:] = False; valid[:, :2] = False; valid[:, -2:] = False
    small = (np.abs(d_x) < min_depth_ratio * depth) & (np.abs(d_y) < min_depth_ratio * depth)
    valid &= np.isfinite(n_world).all(axis=-1)
    n_world = np.where(valid[..., None], n_world, 0.0).astype(np.float32)
    _ = small  # kept for future experiments; not currently used to reject pixels
    return n_world


def sample_colors(view_bgr: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear sample (BGR -> RGB uint8) at (N,2) float pixel coords."""
    H, W = view_bgr.shape[:2]
    x = np.clip(uv[:, 0].astype(np.int32), 0, W - 1)
    y = np.clip(uv[:, 1].astype(np.int32), 0, H - 1)
    bgr = view_bgr[y, x]
    rgb = bgr[:, ::-1]
    return rgb.astype(np.uint8)


# ---------------------------------------------------------------------------
# Geometric cross-view consistency
# ---------------------------------------------------------------------------


def consistency_check(points: np.ndarray,
                      point_normals: np.ndarray,
                      cams: Dict[str, Camera],
                      depth_maps: Dict[str, DepthMap],
                      normal_maps: Dict[str, np.ndarray],
                      ref_name: str,
                      rel_tol: float = 0.02,
                      min_consistent: int = 2,
                      max_normal_deg: float = 60.0,
                      ) -> np.ndarray:
    """For each candidate point, count how many other views agree on both
    depth (within ``rel_tol``) and surface normal (within ``max_normal_deg``).

    ``point_normals`` are the ref-view normals at each candidate point (world
    frame, unit length). ``normal_maps`` is a dict of per-view normal images
    (H, W, 3) in the world frame.

    Returns a boolean mask (N,) of accepted points.
    """
    N = points.shape[0]
    count = np.zeros(N, dtype=np.int32)
    cos_thr = float(np.cos(np.deg2rad(max_normal_deg)))
    ref_n_valid = np.linalg.norm(point_normals, axis=1) > 1e-6
    for other_name, other_dm in depth_maps.items():
        if other_name == ref_name:
            continue
        other_cam = cams[other_name]
        uv, z = other_cam.project(points)
        W, H = other_cam.width, other_cam.height
        u = np.round(uv[:, 0]).astype(np.int32)
        v = np.round(uv[:, 1]).astype(np.int32)
        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
        other_d = np.zeros(N, dtype=np.float32)
        other_n = np.zeros((N, 3), dtype=np.float32)
        idx = np.where(in_img)[0]
        other_d[idx] = other_dm.depth[v[idx], u[idx]]
        other_n[idx] = normal_maps[other_name][v[idx], u[idx]]
        valid = (other_d > 0) & in_img
        # relative depth agreement
        rel = np.where(valid, np.abs(z - other_d) / np.maximum(z, 1e-3), 1.0)
        depth_ok = valid & (rel < rel_tol)
        # normal agreement (skip this filter where either side has no normal)
        other_n_valid = np.linalg.norm(other_n, axis=1) > 1e-6
        cos_sim = np.sum(point_normals * other_n, axis=1)
        normal_ok = ~(ref_n_valid & other_n_valid) | (cos_sim > cos_thr)
        count += (depth_ok & normal_ok).astype(np.int32)
    return count >= min_consistent


# ---------------------------------------------------------------------------
# Top-level fusion
# ---------------------------------------------------------------------------


@dataclass
class FusedCloud:
    points: np.ndarray     # (N, 3) float32
    normals: np.ndarray    # (N, 3) float32
    colors: np.ndarray     # (N, 3) uint8 RGB
    confidence: np.ndarray # (N,)   float32


def fuse_depth_maps(views: Dict[str, np.ndarray],
                    cams: Dict[str, Camera],
                    depth_maps: Dict[str, DepthMap],
                    rel_tol: float = 0.02,
                    min_consistent: int = 2,
                    max_normal_deg: float = 60.0,
                    masks: Optional[Dict[str, np.ndarray]] = None,
                    verbose: bool = True,
                    ) -> FusedCloud:
    """Fuse per-view depth maps into a single oriented + colored point cloud.

    Cross-view consistency keeps only pixels whose 3D location is confirmed
    by at least ``min_consistent`` other views' depth maps (within ``rel_tol``
    of the predicted depth, and with normal angle below ``max_normal_deg``).

    ``masks`` (optional): cam_name -> uint8 (H, W) foreground mask at the same
    resolution as ``views`` / depth maps. Where a mask is present, only
    foreground pixels are back-projected (background depth is discarded even
    if it survived MVS, e.g. from an earlier unmasked run).
    """
    # Precompute normals for every view once (reused inside the inner loop
    # for cross-view agreement).
    normal_maps: Dict[str, np.ndarray] = {
        name: estimate_normals(dm, cams[name]) for name, dm in depth_maps.items()
    }

    all_pts: List[np.ndarray] = []
    all_nrm: List[np.ndarray] = []
    all_col: List[np.ndarray] = []
    all_cnf: List[np.ndarray] = []
    for ref_name, dm in depth_maps.items():
        cam = cams[ref_name]
        normals_map = normal_maps[ref_name]
        # Optional foreground mask zeroes out background depth so it never
        # enters fusion.
        if masks and ref_name in masks:
            fg = masks[ref_name]
            if fg.shape[:2] != dm.depth.shape:
                fg = cv2.resize(fg, (dm.depth.shape[1], dm.depth.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            dm = DepthMap(cam_name=dm.cam_name,
                          depth=np.where(fg > 0, dm.depth, 0.0).astype(np.float32),
                          confidence=np.where(fg > 0, dm.confidence, 0.0).astype(np.float32),
                          normal=dm.normal)
        pts, pix = backproject_depth(dm, cam)
        if pts.shape[0] == 0:
            continue
        n = normals_map[pix[:, 1], pix[:, 0]]
        conf = dm.confidence[pix[:, 1], pix[:, 0]]
        # consistency check (depth + normal)
        keep = consistency_check(pts, n, cams, depth_maps, normal_maps,
                                 ref_name=ref_name, rel_tol=rel_tol,
                                 min_consistent=min_consistent,
                                 max_normal_deg=max_normal_deg)
        pts = pts[keep]; n = n[keep]; conf = conf[keep]; pix = pix[keep]
        col = sample_colors(views[ref_name], pix.astype(np.float32))

        # drop points with invalid normals
        nok = np.linalg.norm(n, axis=1) > 1e-6
        pts = pts[nok]; n = n[nok]; conf = conf[nok]; col = col[nok]

        all_pts.append(pts); all_nrm.append(n); all_col.append(col); all_cnf.append(conf)
        if verbose:
            print(f"[fuse] {ref_name}: {pts.shape[0]:6d} pts after consistency")

    if not all_pts:
        return FusedCloud(
            points=np.zeros((0, 3), np.float32),
            normals=np.zeros((0, 3), np.float32),
            colors=np.zeros((0, 3), np.uint8),
            confidence=np.zeros((0,), np.float32),
        )
    return FusedCloud(
        points=np.concatenate(all_pts, axis=0),
        normals=np.concatenate(all_nrm, axis=0),
        colors=np.concatenate(all_col, axis=0),
        confidence=np.concatenate(all_cnf, axis=0),
    )


# ---------------------------------------------------------------------------
# Voxel down-sampling and statistical outlier removal (numpy-only, lightweight)
# ---------------------------------------------------------------------------


def voxel_downsample(cloud: FusedCloud, voxel_size: float) -> FusedCloud:
    """Voxel-grid downsample by averaging points that fall in the same cell."""
    if cloud.points.shape[0] == 0:
        return cloud
    v = voxel_size
    idx = np.floor(cloud.points / v).astype(np.int64)
    key = idx[:, 0] * 73856093 ^ idx[:, 1] * 19349663 ^ idx[:, 2] * 83492791
    order = np.argsort(key)
    key_s = key[order]
    uniq, start = np.unique(key_s, return_index=True)
    end = np.concatenate([start[1:], [len(key_s)]])
    pts = cloud.points[order]
    nrm = cloud.normals[order]
    col = cloud.colors[order].astype(np.float32)
    cnf = cloud.confidence[order]

    new_pts = np.empty((uniq.size, 3), np.float32)
    new_nrm = np.empty_like(new_pts)
    new_col = np.empty((uniq.size, 3), np.uint8)
    new_cnf = np.empty((uniq.size,), np.float32)
    for i, (s, e) in enumerate(zip(start, end)):
        new_pts[i] = pts[s:e].mean(0)
        n = nrm[s:e].mean(0)
        n_norm = np.linalg.norm(n)
        new_nrm[i] = n / n_norm if n_norm > 1e-6 else np.array([0, 0, 1], np.float32)
        new_col[i] = np.clip(col[s:e].mean(0), 0, 255).astype(np.uint8)
        new_cnf[i] = cnf[s:e].mean()
    return FusedCloud(new_pts, new_nrm, new_col, new_cnf)


def statistical_outlier_removal(cloud: FusedCloud, k: int = 16,
                                std_ratio: float = 2.0) -> FusedCloud:
    """Keep points whose mean-k-NN distance is within std_ratio of the mean.

    Implemented with scipy.spatial.cKDTree to stay lightweight.
    """
    if cloud.points.shape[0] < k + 1:
        return cloud
    from scipy.spatial import cKDTree
    tree = cKDTree(cloud.points)
    d, _ = tree.query(cloud.points, k=k + 1)          # self is first
    md = d[:, 1:].mean(axis=1)
    thr = md.mean() + std_ratio * md.std()
    keep = md < thr
    return FusedCloud(cloud.points[keep], cloud.normals[keep],
                      cloud.colors[keep], cloud.confidence[keep])
