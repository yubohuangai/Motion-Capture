"""Sparse point cloud from known-pose multi-view images.

Because the 11 cameras are already calibrated, we skip Structure-from-Motion's
pose-estimation step entirely and go directly to

    detect SIFT features ->
    pairwise match (ratio test) ->
    epipolar filter using the known fundamental matrices ->
    build tracks (connected components over pairwise matches) ->
    multi-view DLT triangulation ->
    reject by reprojection error + forward-visibility

The result is a sparse but high-precision oriented point cloud that seeds the
downstream dense MVS fusion step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..common.cameras import Camera
from ..common.images import to_gray


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ViewFeatures:
    """SIFT keypoints + descriptors for one image."""
    name: str
    keypoints: np.ndarray      # (K,2) float pixel coords (x, y)
    descriptors: np.ndarray    # (K, 128) float32 SIFT descriptor

    def __len__(self) -> int:
        return int(self.keypoints.shape[0])


@dataclass
class Track:
    """Multi-view observation of a single world point."""
    obs: Dict[str, Tuple[int, np.ndarray]]   # cam_name -> (kp index, (u,v) pixel)
    point3d: Optional[np.ndarray] = None     # (3,) when triangulated
    color: Optional[np.ndarray] = None       # (3,) uint8
    reproj_err: float = 0.0                  # max per-view error (px)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def detect_sift(views: Dict[str, np.ndarray],
                max_features: int = 0,
                contrast_threshold: float = 0.02,
                masks: Optional[Dict[str, np.ndarray]] = None,
                ) -> Dict[str, ViewFeatures]:
    """Detect SIFT features in every view.

    Parameters
    ----------
    max_features : 0 for "no cap" (OpenCV SIFT default).
    masks : optional ``cam_name -> uint8 (H,W)`` foreground mask. Where
        present, SIFT keypoints are detected only on ``mask>0`` pixels.
        Cameras missing from the dict use the whole image (no masking).
    """
    sift = cv2.SIFT_create(nfeatures=max_features,
                           contrastThreshold=contrast_threshold)
    out: Dict[str, ViewFeatures] = {}
    for name, img in views.items():
        gray = to_gray(img)
        m = None
        if masks is not None and name in masks:
            m = masks[name]
            if m.shape[:2] != gray.shape[:2]:
                m = cv2.resize(m, (gray.shape[1], gray.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            # OpenCV expects uint8 with non-zero meaning "detect here"
            m = (m > 0).astype(np.uint8) * 255
        kps, desc = sift.detectAndCompute(gray, m)
        if desc is None:
            kps = []; desc = np.zeros((0, 128), dtype=np.float32)
        pts = np.array([kp.pt for kp in kps], dtype=np.float32) if len(kps) else np.zeros((0, 2), np.float32)
        out[name] = ViewFeatures(name=name, keypoints=pts, descriptors=desc.astype(np.float32))
    return out


# ---------------------------------------------------------------------------
# Pairwise matching with epipolar guidance
# ---------------------------------------------------------------------------


def match_pair(f_a: ViewFeatures,
               f_b: ViewFeatures,
               ratio: float = 0.75,
               ) -> np.ndarray:
    """Ratio-test mutual-nearest SIFT matches.

    Returns (M, 2) int array of (idx_a, idx_b).
    """
    if len(f_a) == 0 or len(f_b) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    # FLANN for SIFT: KDTree. Use k=2 for ratio test in both directions.
    idx_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(idx_params, search_params)

    # forward matches: for each a -> closest b
    m_ab = matcher.knnMatch(f_a.descriptors, f_b.descriptors, k=2)
    good_ab: Dict[int, int] = {}
    for pair in m_ab:
        if len(pair) < 2: continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_ab[m.queryIdx] = m.trainIdx
    # reverse matches for mutual-NN check
    m_ba = matcher.knnMatch(f_b.descriptors, f_a.descriptors, k=2)
    good_ba: Dict[int, int] = {}
    for pair in m_ba:
        if len(pair) < 2: continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_ba[m.queryIdx] = m.trainIdx
    matches: List[Tuple[int, int]] = []
    for ia, ib in good_ab.items():
        if good_ba.get(ib, -1) == ia:
            matches.append((ia, ib))
    if not matches:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(matches, dtype=np.int64)


def _epipolar_distance(F: np.ndarray, uv_a: np.ndarray, uv_b: np.ndarray) -> np.ndarray:
    """Symmetric epipolar distance (in pixels).

    ``uv_a`` / ``uv_b`` are (N,2). F is such that ``uv_b^T F uv_a = 0``.
    """
    N = uv_a.shape[0]
    a = np.concatenate([uv_a, np.ones((N, 1))], axis=1)           # (N,3)
    b = np.concatenate([uv_b, np.ones((N, 1))], axis=1)           # (N,3)
    lb = a @ F.T                                                   # epipolar lines in b
    la = b @ F                                                     # epipolar lines in a
    # distance from point to its epipolar line
    d_b = np.abs(np.einsum("ij,ij->i", b, lb)) / (np.sqrt(lb[:, 0] ** 2 + lb[:, 1] ** 2) + 1e-12)
    d_a = np.abs(np.einsum("ij,ij->i", a, la)) / (np.sqrt(la[:, 0] ** 2 + la[:, 1] ** 2) + 1e-12)
    return 0.5 * (d_a + d_b)


def filter_matches_epipolar(matches: np.ndarray,
                            f_a: ViewFeatures, f_b: ViewFeatures,
                            cam_a: Camera, cam_b: Camera,
                            max_epi_px: float = 2.0) -> np.ndarray:
    """Drop putative matches that violate the known epipolar geometry."""
    if matches.size == 0:
        return matches
    uv_a = f_a.keypoints[matches[:, 0]]
    uv_b = f_b.keypoints[matches[:, 1]]
    F = cam_a.fundamental_to(cam_b)            # x_b^T F x_a = 0
    d = _epipolar_distance(F, uv_a, uv_b)
    keep = d < max_epi_px
    return matches[keep]


# ---------------------------------------------------------------------------
# Track construction (connected components over pairwise matches)
# ---------------------------------------------------------------------------


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[Tuple[str, int], Tuple[str, int]] = {}

    def find(self, x):
        while self.parent.get(x, x) != x:
            self.parent[x] = self.parent.get(self.parent[x], self.parent[x])
            x = self.parent[x]
        self.parent.setdefault(x, x)
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_tracks(features: Dict[str, ViewFeatures],
                 pair_matches: Dict[Tuple[str, str], np.ndarray],
                 min_views: int = 3,
                 ) -> List[Dict[str, int]]:
    """Union-find tracks from pairwise matches.

    Returns a list of dicts: cam_name -> feature index. Tracks that contain
    multiple features from the same view (conflicting merges) are dropped
    since they indicate an ambiguous match.
    """
    uf = _UnionFind()
    for (a, b), matches in pair_matches.items():
        for ia, ib in matches:
            uf.union((a, int(ia)), (b, int(ib)))

    groups: Dict[Tuple[str, int], Dict[str, int]] = {}
    for (cam, idx) in list(uf.parent.keys()):
        root = uf.find((cam, idx))
        g = groups.setdefault(root, {})
        if cam in g and g[cam] != idx:
            g["__conflict__"] = 1            # mark conflict (dropped below)
        g[cam] = idx

    tracks: List[Dict[str, int]] = []
    for g in groups.values():
        if g.get("__conflict__"):
            continue
        g = {k: v for k, v in g.items() if k != "__conflict__"}
        if len(g) < min_views:
            continue
        tracks.append(g)
    return tracks


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------


def triangulate_multiview(uvs: Sequence[np.ndarray],
                          Ps: Sequence[np.ndarray]) -> np.ndarray:
    """Linear DLT triangulation from >=2 views.

    uvs : list of (2,) pixel coordinates (undistorted).
    Ps  : list of (3,4) projection matrices K[R|T].

    Returns (3,) world-space point.
    """
    A = []
    for uv, P in zip(uvs, Ps):
        u, v = float(uv[0]), float(uv[1])
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.stack(A, axis=0)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X[:3] / X[3]
    return X


def triangulate_tracks(tracks: List[Dict[str, int]],
                       features: Dict[str, ViewFeatures],
                       cameras: Dict[str, Camera],
                       views: Dict[str, np.ndarray],
                       max_reproj_err: float = 2.0,
                       ) -> List[Track]:
    """Triangulate every track and apply a reprojection-error filter."""
    out: List[Track] = []
    for t in tracks:
        uvs_u: Dict[str, np.ndarray] = {}
        uvs, Ps, obs = [], [], {}
        for cam_name, kp_idx in t.items():
            uv = features[cam_name].keypoints[kp_idx]
            uv_u = cameras[cam_name].undistort_points(uv.reshape(1, 2))[0]
            uvs_u[cam_name] = uv_u
            uvs.append(uv_u)
            Ps.append(cameras[cam_name].P)
            obs[cam_name] = (int(kp_idx), np.asarray(uv, dtype=np.float32))
        X = triangulate_multiview(uvs, Ps)

        # verify: in front of all cameras + small reprojection error
        max_err = 0.0
        behind = False
        for cam_name, (kp_idx, _) in obs.items():
            cam = cameras[cam_name]
            uv_pred, z = cam.project(X.reshape(1, 3))
            if z[0] <= 0:
                behind = True; break
            err = float(np.linalg.norm(uv_pred[0] - uvs_u[cam_name]))
            if err > max_err:
                max_err = err
        if behind or max_err > max_reproj_err:
            continue

        # sample color from the view closest to the track mean image location
        # (use the first view in the dict: Python dict preserves insertion order)
        first_cam = next(iter(obs))
        u, v = obs[first_cam][1]
        img = views[first_cam]
        H, W = img.shape[:2]
        cu = int(np.clip(round(u), 0, W - 1))
        cv_ = int(np.clip(round(v), 0, H - 1))
        bgr = img[cv_, cu]
        color = np.array([bgr[2], bgr[1], bgr[0]], dtype=np.uint8)

        out.append(Track(obs=obs, point3d=X.astype(np.float32),
                         color=color, reproj_err=max_err))
    return out


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------


def sparse_reconstruct(views: Dict[str, np.ndarray],
                       cameras: Dict[str, Camera],
                       ratio: float = 0.75,
                       max_epi_px: float = 2.0,
                       max_reproj_err: float = 2.0,
                       min_views: int = 3,
                       masks: Optional[Dict[str, np.ndarray]] = None,
                       verbose: bool = True,
                       ) -> Tuple[List[Track], Dict[str, ViewFeatures],
                                  Dict[Tuple[str, str], np.ndarray]]:
    """Run the whole sparse pipeline end to end.

    ``masks`` (optional): cam_name -> uint8 (H,W) foreground mask. Keypoints
    are only detected within the masked region for cameras present in this
    dict; cameras missing from the dict use the whole image.

    Returns (tracks, features, pairwise_matches).
    """
    if verbose:
        n_masked = 0 if masks is None else len(masks)
        print(f"[sparse] detecting SIFT on {len(views)} views "
              f"({n_masked} with foreground masks) ...")
    features = detect_sift(views, masks=masks)
    if verbose:
        for n, f in features.items():
            print(f"  [{n}] {len(f)} keypoints")

    names = list(views.keys())
    pair_matches: Dict[Tuple[str, str], np.ndarray] = {}
    total = 0
    for a, b in combinations(names, 2):
        raw = match_pair(features[a], features[b], ratio=ratio)
        flt = filter_matches_epipolar(raw, features[a], features[b],
                                      cameras[a], cameras[b], max_epi_px=max_epi_px)
        pair_matches[(a, b)] = flt
        total += len(flt)
        if verbose:
            print(f"  match {a}-{b}: raw={len(raw):5d}  kept={len(flt):5d}")
    if verbose:
        print(f"[sparse] total kept matches: {total}")

    tracks_idx = build_tracks(features, pair_matches, min_views=min_views)
    if verbose:
        print(f"[sparse] {len(tracks_idx)} candidate tracks (>= {min_views} views)")

    tracks = triangulate_tracks(tracks_idx, features, cameras, views,
                                max_reproj_err=max_reproj_err)
    if verbose:
        errs = [t.reproj_err for t in tracks]
        if errs:
            print(f"[sparse] kept {len(tracks)} 3D points "
                  f"(median reproj {np.median(errs):.2f}px, max {max(errs):.2f}px)")
        else:
            print("[sparse] no tracks survived filtering")
    return tracks, features, pair_matches


def tracks_to_point_cloud(tracks: Sequence[Track]) -> Tuple[np.ndarray, np.ndarray]:
    """Stack triangulated points and RGB colors for output."""
    pts = np.stack([t.point3d for t in tracks], axis=0)
    col = np.stack([t.color if t.color is not None else np.array([200, 200, 200], np.uint8)
                    for t in tracks], axis=0)
    return pts.astype(np.float32), col.astype(np.uint8)
