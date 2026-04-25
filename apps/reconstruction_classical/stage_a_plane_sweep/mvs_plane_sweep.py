"""Plane-sweep Multi-View Stereo for calibrated cameras (CPU / numpy + cv2).

Per reference view, we sweep N_D candidate depths uniformly in inverse depth,
warp the chosen source views onto the reference through the plane-induced
homography, and pick the depth that maximises the local patch similarity
(masked NCC, aggregated over sources).

The implementation is deliberately dependency-light: only ``numpy`` and ``cv2``
are required. For a ~1000x500 reference image and ~128 depth planes with 4
source views it runs in well under a minute per view.

References
----------
- Collins (CVPR '96), "A Space-Sweep Approach to True Multi-Image Matching"
- Galliani et al. (ICCV '15), "Massively Parallel Multiview Stereopsis by
  Surface Normal Diffusion" — we adopt the neighbour selection heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..common.cameras import Camera
from ..common.images import to_gray


@dataclass
class DepthMap:
    """Per-pixel depth + confidence for one reference view."""
    cam_name: str
    depth: np.ndarray       # (H,W) float32, 0 where invalid
    confidence: np.ndarray  # (H,W) float32 in [0,1]
    normal: Optional[np.ndarray] = None   # (H,W,3), optional


# ---------------------------------------------------------------------------
# Depth-range estimation
# ---------------------------------------------------------------------------


def depth_range_from_points(cam: Camera,
                            world_points: np.ndarray,
                            pad_ratio: float = 0.25,
                            min_percentile: float = 2.0,
                            max_percentile: float = 98.0,
                            ) -> Tuple[float, float]:
    """Choose a depth range for ``cam`` from a batch of 3D points.

    Uses percentiles of the in-frustum points to be robust to outliers and
    adds a symmetric log-padding.
    """
    if world_points.shape[0] == 0:
        return 0.5, 10.0
    uv, z = cam.project(world_points)
    W, H = cam.width, cam.height
    in_img = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (z > 0)
    z_valid = z[in_img]
    if z_valid.size < 10:
        z_valid = z[z > 0]
    if z_valid.size < 2:
        return 0.5, 10.0
    d_lo = float(np.percentile(z_valid, min_percentile))
    d_hi = float(np.percentile(z_valid, max_percentile))
    # log-pad
    lo_log = np.log(max(d_lo, 1e-3))
    hi_log = np.log(max(d_hi, d_lo + 1e-3))
    span = hi_log - lo_log
    lo = float(np.exp(lo_log - pad_ratio * span))
    hi = float(np.exp(hi_log + pad_ratio * span))
    return max(lo, 1e-2), hi


def inverse_depth_planes(d_min: float, d_max: float, n: int) -> np.ndarray:
    """Return ``n`` depth values uniformly spaced in inverse depth."""
    inv = np.linspace(1.0 / d_max, 1.0 / d_min, n, dtype=np.float32)
    return 1.0 / inv


# ---------------------------------------------------------------------------
# Source view selection
# ---------------------------------------------------------------------------


def pick_source_views(ref_name: str,
                      cams: Dict[str, Camera],
                      max_sources: int = 4,
                      min_angle_deg: float = 5.0,
                      max_angle_deg: float = 45.0,
                      ) -> List[str]:
    """Pick the best N source cameras around the reference.

    The heuristic scores candidates by forward-axis similarity but excludes
    those whose baseline is too small (weak triangulation) or whose viewing
    angle differs too much (low co-visibility).
    """
    ref = cams[ref_name]
    scores: List[Tuple[float, str]] = []
    for name, cam in cams.items():
        if name == ref_name:
            continue
        cos = float(np.clip(np.dot(ref.forward, cam.forward), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos)))
        if angle_deg < min_angle_deg or angle_deg > max_angle_deg:
            continue
        baseline = float(np.linalg.norm(ref.center - cam.center))
        score = cos / (1.0 + 0.1 * abs(30.0 - angle_deg))   # prefer ~30-deg offset
        score += 0.05 * min(baseline, 3.0)                  # reward some baseline
        scores.append((score, name))
    scores.sort(reverse=True)
    picked = [name for _, name in scores[:max_sources]]
    return picked


# ---------------------------------------------------------------------------
# NCC helpers
# ---------------------------------------------------------------------------


def _box_mean(img: np.ndarray, ksize: int) -> np.ndarray:
    """Mean of ``img`` over a ksize x ksize box with ``BORDER_REFLECT``."""
    return cv2.boxFilter(img, ddepth=-1, ksize=(ksize, ksize),
                         normalize=True, borderType=cv2.BORDER_REFLECT)


def zncc_map(ref: np.ndarray, src: np.ndarray, ksize: int = 7,
             eps: float = 1e-3) -> np.ndarray:
    """Per-pixel ZNCC between two grayscale float images of matching shape.

    Returns values in [-1, 1]; invalid (constant) regions return 0.
    """
    ref = ref.astype(np.float32)
    src = src.astype(np.float32)
    mr = _box_mean(ref, ksize)
    ms = _box_mean(src, ksize)
    mrr = _box_mean(ref * ref, ksize)
    mss = _box_mean(src * src, ksize)
    mrs = _box_mean(ref * src, ksize)
    cov = mrs - mr * ms
    var_r = np.clip(mrr - mr * mr, 0.0, None)
    var_s = np.clip(mss - ms * ms, 0.0, None)
    denom = np.sqrt(var_r * var_s) + eps
    ncc = cov / denom
    # kill plain regions (below a minimal variance threshold)
    low_var = (var_r < eps) | (var_s < eps)
    ncc[low_var] = 0.0
    return ncc


# ---------------------------------------------------------------------------
# Plane sweep core
# ---------------------------------------------------------------------------


def clahe_gray(img_bgr: np.ndarray, clip_limit: float = 3.0,
               tile: int = 8) -> np.ndarray:
    """Grayscale with CLAHE applied on the LAB L channel.

    Equalises local contrast so very dark regions (black fur, shadowed cloth)
    carry the same matchable signal as bright ones. Without this, ZNCC on
    high-dynamic-range subjects silently drops the dark side because its
    patch variance falls below sensor noise.
    """
    if img_bgr.ndim == 2:
        L = img_bgr
    else:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L = lab[..., 0]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(L)


def texture_mask_from_gray(gray: np.ndarray, ksize: int = 7,
                           min_grad_var: float = 2.0) -> np.ndarray:
    """Boolean mask of pixels with enough local texture to trust for NCC.

    Operates on a precomputed grayscale (typically CLAHE-equalised) so the
    threshold has consistent meaning across bright and dark regions.
    """
    g = gray.astype(np.float32)
    mean = _box_mean(g, ksize)
    mean_sq = _box_mean(g * g, ksize)
    var = np.clip(mean_sq - mean * mean, 0.0, None)
    return var > min_grad_var


def plane_sweep(ref_view: np.ndarray,
                src_views: Sequence[np.ndarray],
                ref_cam: Camera,
                src_cams: Sequence[Camera],
                depths: np.ndarray,
                ncc_ksize: int = 7,
                aggregate: str = "mean",
                min_sources_per_pixel: int = 2,
                use_texture_mask: bool = True,
                texture_var_thr: float = 2.0,
                ref_fg_mask: Optional[np.ndarray] = None,
                use_clahe: bool = True,
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Winner-take-all plane-sweep depth estimation for one reference view.

    Parameters
    ----------
    ref_view : (H,W,3) BGR uint8 reference image.
    src_views : list of (H_s, W_s, 3) BGR uint8 source images (resolutions
        can differ; their intrinsics in ``src_cams`` must match).
    ref_cam, src_cams : associated calibrated cameras.
    depths : (D,) candidate depths (camera-frame Z distances).
    aggregate : how to aggregate per-source NCC into a single cost. One of
        "mean" (average) or "top2" (average of the top 2 sources per pixel).
    use_texture_mask : when True, pixels whose local gray-variance is below
        ``texture_var_thr`` are invalidated (no dense depth assigned).

    Returns
    -------
    depth : (H, W) float32. 0 where no source supports the pixel.
    confidence : (H, W) float32 in [0,1]. Combines two signals:
        (i) peak-vs-mean cost gap (how distinctive the best depth is
            relative to the average cost across all planes)
        (ii) peak-vs-second cost gap (how close the runner-up is)
        and multiplies them together for a stable confidence.
    """
    H_r, W_r = ref_view.shape[:2]
    if use_clahe:
        ref_g_u8 = clahe_gray(ref_view)
    else:
        ref_g_u8 = to_gray(ref_view)
    ref_g = ref_g_u8.astype(np.float32)
    # Precompute source grays once so we can warp the gray (instead of BGR)
    # each plane — avoids re-applying CLAHE on warped output, which would
    # spread border pixels into the equalisation histogram.
    src_grays_u8 = [
        clahe_gray(s) if use_clahe else to_gray(s) for s in src_views
    ]
    D = int(depths.shape[0])

    # Best cost at each pixel so far (we invert NCC into cost = 1 - ncc).
    best_cost = np.full((H_r, W_r), np.inf, dtype=np.float32)
    second_cost = np.full_like(best_cost, np.inf)
    best_depth = np.zeros((H_r, W_r), dtype=np.float32)
    any_valid = np.zeros((H_r, W_r), dtype=bool)
    # Running mean cost across planes (over all valid plane sweeps)
    cost_sum = np.zeros_like(best_cost, dtype=np.float32)
    cost_cnt = np.zeros_like(best_cost, dtype=np.int32)

    # (Optional) texture mask: low-variance pixels are unlikely to match
    # reliably; we invalidate them upfront.
    tex_valid = np.ones((H_r, W_r), dtype=bool)
    if use_texture_mask:
        tex_valid = texture_mask_from_gray(ref_g_u8, ksize=ncc_ksize,
                                           min_grad_var=texture_var_thr)
    # Foreground mask: restrict depth estimation to the object. Background
    # pixels get cost=inf so the confidence filter drops them — cleaner and
    # faster because most images are >90% background after segmentation.
    if ref_fg_mask is not None:
        fg = ref_fg_mask
        if fg.shape[:2] != (H_r, W_r):
            fg = cv2.resize(fg, (W_r, H_r), interpolation=cv2.INTER_NEAREST)
        tex_valid = tex_valid & (fg > 0)

    for di, d in enumerate(depths):
        per_src_ncc = np.full((len(src_views), H_r, W_r), -1.0, dtype=np.float32)
        per_src_valid = np.zeros_like(per_src_ncc, dtype=bool)
        for si, (src_gray_u8, src_cam) in enumerate(zip(src_grays_u8, src_cams)):
            H = ref_cam.plane_induced_homography(src_cam, depth=float(d))
            # cv2.warpPerspective's default flag treats M as the FORWARD
            # src->dst map and inverts it internally to sample src per dst
            # pixel. Since plane_induced_homography already returns H: src->ref,
            # we pass it directly — inverting it first produces the wrong warp.
            warped_g_u8 = cv2.warpPerspective(src_gray_u8, H, (W_r, H_r),
                                              flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=0)
            mask_src = cv2.warpPerspective(np.ones(src_gray_u8.shape[:2], np.uint8) * 255,
                                           H, (W_r, H_r),
                                           flags=cv2.INTER_NEAREST,
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=0)
            warped_g = warped_g_u8.astype(np.float32)
            ncc = zncc_map(ref_g, warped_g, ksize=ncc_ksize)
            valid = mask_src > 0
            # invalidate border pixels where the NCC window would reach outside
            half = ncc_ksize // 2
            border = np.zeros_like(valid, dtype=bool)
            border[half:-half, half:-half] = True
            valid = valid & border & tex_valid
            per_src_ncc[si] = ncc
            per_src_valid[si] = valid

        # aggregate sources
        ncc_agg = _aggregate_ncc(per_src_ncc, per_src_valid, aggregate,
                                 min_sources=min_sources_per_pixel)
        cost = 1.0 - ncc_agg                         # lower is better
        # pixels with no valid sources get +inf cost
        support = per_src_valid.sum(axis=0) >= min_sources_per_pixel
        cost = np.where(support, cost, np.inf)

        # update running mean (valid planes only)
        cost_sum = np.where(support, cost_sum + cost, cost_sum)
        cost_cnt = cost_cnt + support.astype(np.int32)

        # update winner
        better = cost < best_cost
        second_cost = np.where(better, best_cost, np.minimum(second_cost, cost))
        best_cost = np.where(better, cost, best_cost)
        best_depth = np.where(better, np.float32(d), best_depth)
        any_valid |= support

    # Peak-vs-mean cost confidence: measures how distinctive the best depth
    # is relative to the average cost across the sweep. This is much more
    # stable than the best-vs-second-best ratio when adjacent depth planes
    # have similar costs (a common case at dense sampling).
    have_data = any_valid & (cost_cnt >= max(2, D // 4))
    mean_cost = np.where(cost_cnt > 0, cost_sum / np.maximum(cost_cnt, 1), 0.0)
    best_cost_safe = np.where(have_data, best_cost, 0.0)
    peak_vs_mean = np.where(
        have_data,
        (mean_cost - best_cost_safe) / np.maximum(mean_cost, 1e-3),
        0.0,
    )
    conf = np.clip(peak_vs_mean, 0.0, 1.0).astype(np.float32)

    depth_out = np.where(have_data, best_depth, 0.0).astype(np.float32)
    return depth_out, conf


def _aggregate_ncc(per_src_ncc: np.ndarray,
                   per_src_valid: np.ndarray,
                   mode: str,
                   min_sources: int) -> np.ndarray:
    S, H, W = per_src_ncc.shape
    if mode == "mean":
        ncc_sum = np.where(per_src_valid, per_src_ncc, 0.0).sum(axis=0)
        cnt = per_src_valid.sum(axis=0)
        out = np.where(cnt >= min_sources, ncc_sum / np.maximum(cnt, 1), -1.0)
        return out
    if mode == "top2":
        # replace invalid with very negative so they lose the top-k race
        tmp = np.where(per_src_valid, per_src_ncc, -2.0)
        k = min(2, S)
        part = -np.partition(-tmp, kth=k - 1, axis=0)[:k]      # top-k along S
        out = part.mean(axis=0)
        cnt = per_src_valid.sum(axis=0)
        out = np.where(cnt >= min_sources, out, -1.0)
        return out
    raise ValueError(f"unknown aggregate mode: {mode}")


# ---------------------------------------------------------------------------
# Simple post-processing
# ---------------------------------------------------------------------------


def filter_depth_by_confidence(dm: DepthMap, min_confidence: float = 0.1) -> DepthMap:
    mask = dm.confidence >= min_confidence
    depth = np.where(mask, dm.depth, 0.0).astype(np.float32)
    conf = np.where(mask, dm.confidence, 0.0).astype(np.float32)
    return DepthMap(cam_name=dm.cam_name, depth=depth, confidence=conf, normal=dm.normal)


def median_filter_depth(dm: DepthMap, ksize: int = 5) -> DepthMap:
    if ksize <= 1:
        return dm
    depth = cv2.medianBlur(dm.depth, ksize)
    return DepthMap(cam_name=dm.cam_name, depth=depth.astype(np.float32),
                    confidence=dm.confidence, normal=dm.normal)


def joint_bilateral_smooth_depth(dm: DepthMap,
                                 guide_bgr: np.ndarray,
                                 d: int = 5,
                                 sigma_color: float = 20.0,
                                 sigma_space: float = 5.0) -> DepthMap:
    """Edge-preserving smoothing of the depth map guided by the color image.

    Invalid (zero) pixels are preserved; valid pixels are smoothed within
    regions of similar color, which tames speckle without bridging object
    silhouettes.
    """
    mask = (dm.depth > 0).astype(np.float32)
    if mask.sum() == 0:
        return dm
    guide = guide_bgr if guide_bgr.ndim == 3 else cv2.cvtColor(guide_bgr, cv2.COLOR_GRAY2BGR)
    depth_pad = dm.depth.astype(np.float32)
    # Fill zeros with the mean of valid neighbours to avoid pulling the
    # bilateral filter toward 0 at object boundaries.
    if (dm.depth == 0).any():
        dilated = cv2.dilate(dm.depth, np.ones((3, 3), np.uint8), iterations=2)
        depth_pad = np.where(dm.depth > 0, dm.depth, dilated)
    try:
        # cv2.ximgproc provides a true joint bilateral; fall back to
        # per-channel bilateralFilter if ximgproc isn't installed.
        import cv2.ximgproc as ximgproc  # type: ignore
        smoothed = ximgproc.jointBilateralFilter(
            guide.astype(np.uint8), depth_pad, d=d,
            sigmaColor=sigma_color, sigmaSpace=sigma_space)
    except Exception:
        smoothed = cv2.bilateralFilter(depth_pad, d=d,
                                        sigmaColor=sigma_color,
                                        sigmaSpace=sigma_space)
    out = np.where(dm.depth > 0, smoothed, 0.0).astype(np.float32)
    return DepthMap(cam_name=dm.cam_name, depth=out,
                    confidence=dm.confidence, normal=dm.normal)


# ---------------------------------------------------------------------------
# End-to-end MVS driver for all views
# ---------------------------------------------------------------------------


def compute_all_depth_maps(views: Dict[str, np.ndarray],
                           cams: Dict[str, Camera],
                           sparse_points: np.ndarray,
                           n_depths: int = 128,
                           max_sources: int = 4,
                           ncc_ksize: int = 7,
                           aggregate: str = "top2",
                           min_conf: float = 0.05,
                           median_ksize: int = 5,
                           bilateral_d: int = 5,
                           bilateral_sigma_color: float = 20.0,
                           bilateral_sigma_space: float = 5.0,
                           texture_var_thr: float = 6.0,
                           masks: Optional[Dict[str, np.ndarray]] = None,
                           verbose: bool = True,
                           ) -> Dict[str, DepthMap]:
    """Compute a :class:`DepthMap` for each view using the shared sparse points
    to set per-view depth ranges."""
    results: Dict[str, DepthMap] = {}
    for ref_name, ref_view in views.items():
        ref_cam = cams[ref_name]
        src_names = pick_source_views(ref_name, cams, max_sources=max_sources)
        if len(src_names) < 2:
            if verbose:
                print(f"[mvs] {ref_name}: not enough sources ({src_names}); skipping")
            continue
        d_min, d_max = depth_range_from_points(ref_cam, sparse_points)
        depths = inverse_depth_planes(d_min, d_max, n_depths)
        if verbose:
            print(f"[mvs] ref={ref_name} srcs={src_names} "
                  f"depth=[{d_min:.2f}, {d_max:.2f}] ({n_depths} planes)")
        src_views = [views[s] for s in src_names]
        src_cams = [cams[s] for s in src_names]
        ref_fg = masks.get(ref_name) if masks else None
        depth, conf = plane_sweep(ref_view, src_views, ref_cam, src_cams,
                                  depths, ncc_ksize=ncc_ksize,
                                  aggregate=aggregate,
                                  texture_var_thr=texture_var_thr,
                                  ref_fg_mask=ref_fg)
        dm = DepthMap(cam_name=ref_name, depth=depth, confidence=conf)
        dm = filter_depth_by_confidence(dm, min_confidence=min_conf)
        if median_ksize > 1:
            dm = median_filter_depth(dm, ksize=median_ksize)
        if bilateral_d > 0:
            dm = joint_bilateral_smooth_depth(
                dm, ref_view, d=bilateral_d,
                sigma_color=bilateral_sigma_color,
                sigma_space=bilateral_sigma_space)
        if verbose:
            v = dm.depth[dm.depth > 0]
            if v.size:
                print(f"       {100 * (v.size / dm.depth.size):5.1f}% valid  "
                      f"median d={np.median(v):.2f}m  mean conf={dm.confidence[dm.depth > 0].mean():.3f}")
            else:
                print("       no valid pixels after filtering")
        results[ref_name] = dm
    return results
