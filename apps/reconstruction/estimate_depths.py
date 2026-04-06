"""
Plane-sweep multi-view stereo depth estimation.

Uses photometric consistency (pixel variance across warped source views)
rather than feature matching — handles repeated textures better than COLMAP.

Core algorithm adapted from ReconViaGen/classical_mvs/plane_sweep.py.

Usage:
    python apps/reconstruction/estimate_depths.py /path/to/data
    # Reads images/<cam>/, masks/<cam>/, intri.yml, extri.yml.
    # Optionally uses colmap_ws/sparse/0/ for depth range.
    # Writes depth maps + colormapped PNGs to <data>/depths/.

Output per camera:
    depths/<cam>.npy          raw float32 depth (camera Z)
    depths/<cam>_color.png    colormapped depth for visual inspection
    depths/<cam>_conf.npy     confidence map [0, 1]
"""

import argparse
import os
import sys
import time
from os.path import join
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, join(os.path.dirname(__file__), '..', '..'))

from easymocap.mytools.camera_utils import read_camera


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _camera_center(R, T):
    return (-R.T @ T).ravel()


def _select_source_views(ref_idx, centers, num_sources):
    """Pick views with good triangulation angle (~10-60 deg from reference)."""
    scene_center = centers.mean(axis=0)
    ref_dir = centers[ref_idx] - scene_center
    ref_dir /= np.linalg.norm(ref_dir) + 1e-12

    scores = []
    for i, c in enumerate(centers):
        if i == ref_idx:
            scores.append(-1.0)
            continue
        d = c - scene_center
        d /= np.linalg.norm(d) + 1e-12
        cos_a = np.clip(np.dot(ref_dir, d), -1, 1)
        angle = np.degrees(np.arccos(cos_a))
        score = np.exp(-((angle - 30) ** 2) / (2 * 25 ** 2))
        scores.append(score)

    order = np.argsort(scores)[::-1]
    return [int(i) for i in order[:num_sources]]


def _depth_range_from_cameras(centers, margin=0.5):
    """Heuristic depth range from camera geometry."""
    scene_center = centers.mean(axis=0)
    dists = np.linalg.norm(centers - scene_center, axis=1)
    d_min = max(dists.min() * margin, 1e-4)
    d_max = dists.max() * (1.0 + margin)
    return float(d_min), float(d_max)


def _depth_range_from_sparse(sparse_dir, cams, cam_names, percentile=2):
    """Compute per-view depth range from COLMAP sparse 3D points."""
    from easymocap.mytools.colmap_structure import read_points3d_binary, read_points3D_text
    pts_bin = join(sparse_dir, 'points3D.bin')
    pts_txt = join(sparse_dir, 'points3D.txt')
    if os.path.exists(pts_bin):
        pts3d = read_points3d_binary(pts_bin)
    elif os.path.exists(pts_txt):
        pts3d = read_points3D_text(pts_txt)
    else:
        return None, None

    if len(pts3d) < 10:
        return None, None

    xyz = np.array([p.xyz for p in pts3d.values()])
    print(f'[depths] Sparse points: {len(xyz)} points')

    all_depths = []
    for name in cam_names:
        R = cams[name]['R']
        T = cams[name]['T']
        pts_cam = (R @ xyz.T + T).T
        z = pts_cam[:, 2]
        z = z[z > 0]
        all_depths.extend(z.tolist())

    all_depths = np.array(all_depths)
    d_min = float(np.percentile(all_depths, percentile))
    d_max = float(np.percentile(all_depths, 100 - percentile))
    margin = (d_max - d_min) * 0.3
    d_min = max(d_min - margin, 1e-4)
    d_max = d_max + margin
    return d_min, d_max


# ---------------------------------------------------------------------------
# Core plane-sweep
# ---------------------------------------------------------------------------

def _build_homography(K_ref, K_src, R_ref, T_ref, R_src, T_src, depth):
    """Homography warping src to ref at fronto-parallel plane at given depth."""
    R_rel = R_ref @ R_src.T
    t_rel = T_ref - R_rel @ T_src
    n = torch.tensor([[0.0], [0.0], [1.0]], device=K_ref.device, dtype=K_ref.dtype)
    H = K_ref @ (R_rel - t_rel @ n.T / depth) @ torch.inverse(K_src)
    return H


def _warp_image(src, H_matrix, height, width):
    """Warp src image to reference view via homography."""
    ys, xs = torch.meshgrid(
        torch.arange(height, device=src.device, dtype=src.dtype),
        torch.arange(width, device=src.device, dtype=src.dtype),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    coords = torch.stack([xs, ys, ones], dim=-1).reshape(-1, 3)

    H_inv = torch.inverse(H_matrix)
    src_coords = (H_inv @ coords.T).T
    src_coords = src_coords[:, :2] / (src_coords[:, 2:3] + 1e-8)

    src_coords[..., 0] = 2.0 * src_coords[..., 0] / (width - 1) - 1.0
    src_coords[..., 1] = 2.0 * src_coords[..., 1] / (height - 1) - 1.0
    grid = src_coords.reshape(1, height, width, 2)

    warped = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros",
                           align_corners=True)
    return warped


def _compute_cost_volume(ref_img, src_imgs, K_ref, Ks_src, R_ref, T_ref,
                         Rs_src, Ts_src, depths, window_size=7, mask_ref=None):
    """Cost volume (D, H, W) — lower = more inconsistent."""
    D = depths.shape[0]
    _, _, H, W = ref_img.shape
    device = ref_img.device

    pad = window_size // 2
    avg_kernel = torch.ones(1, 1, window_size, window_size, device=device) / (window_size ** 2)

    cost_volume = torch.zeros(D, H, W, device=device)

    for di, d in enumerate(depths):
        sum_c = torch.zeros(1, 3, H, W, device=device)
        sum_c2 = torch.zeros(1, 3, H, W, device=device)
        count = torch.zeros(1, 1, H, W, device=device)

        # Include reference image in the variance computation
        sum_c += ref_img
        sum_c2 += ref_img ** 2
        count += 1.0

        for si in range(len(src_imgs)):
            Hm = _build_homography(
                K_ref, Ks_src[si], R_ref, T_ref,
                Rs_src[si], Ts_src[si], float(d),
            )
            warped = _warp_image(src_imgs[si], Hm, H, W)
            valid = (warped.abs().sum(dim=1, keepdim=True) > 1e-6).float()
            sum_c += warped * valid
            sum_c2 += (warped ** 2) * valid
            count += valid

        count = count.clamp(min=1)
        mean = sum_c / count
        variance = (sum_c2 / count - mean ** 2).mean(dim=1, keepdim=True)
        variance = variance.clamp(min=0)

        var_padded = F.pad(variance, [pad] * 4, mode="reflect")
        var_smooth = F.conv2d(var_padded, avg_kernel)
        cost_volume[di] = var_smooth.squeeze()

    if mask_ref is not None:
        cost_volume[:, ~mask_ref.squeeze().bool()] = 1e6

    return cost_volume


# ---------------------------------------------------------------------------
# Geometric consistency filter
# ---------------------------------------------------------------------------

def _geometric_consistency(depth_maps, Ks, Rs, Ts, source_indices,
                           thresh_px=1.5, thresh_depth_rel=0.02):
    """Cross-view reprojection check. Returns confidence maps."""
    confidence = {}
    for ref_idx, depth_ref in depth_maps.items():
        H, W = depth_ref.shape
        K_r, R_r, T_r = Ks[ref_idx], Rs[ref_idx], Ts[ref_idx]
        invK = np.linalg.inv(K_r)

        ys, xs = np.mgrid[:H, :W].astype(np.float64)
        ones = np.ones_like(xs)
        pixels = np.stack([xs, ys, ones], axis=-1)

        rays = (invK @ pixels.reshape(-1, 3).T).T.reshape(H, W, 3)
        pts_cam = rays * depth_ref[..., None]
        pts_world = (R_r.T @ (pts_cam.reshape(-1, 3).T - T_r)).T.reshape(H, W, 3)

        votes = np.zeros((H, W), dtype=np.float32)
        for src_idx in source_indices.get(ref_idx, []):
            if src_idx not in depth_maps:
                continue
            K_s, R_s, T_s = Ks[src_idx], Rs[src_idx], Ts[src_idx]
            depth_src = depth_maps[src_idx]

            pts_src_cam = (R_s @ pts_world.reshape(-1, 3).T + T_s).T.reshape(H, W, 3)
            z_src = pts_src_cam[..., 2]

            proj = (K_s @ pts_src_cam.reshape(-1, 3).T).T.reshape(H, W, 3)
            u = proj[..., 0] / (proj[..., 2] + 1e-8)
            v = proj[..., 1] / (proj[..., 2] + 1e-8)

            H_s, W_s = depth_src.shape
            in_bounds = (u >= 0) & (u < W_s - 1) & (v >= 0) & (v < H_s - 1) & (z_src > 0)
            ui = np.clip(np.round(u).astype(int), 0, W_s - 1)
            vi = np.clip(np.round(v).astype(int), 0, H_s - 1)
            depth_sampled = depth_src[vi, ui]

            reproj_ok = np.sqrt((u - np.round(u)) ** 2 + (v - np.round(v)) ** 2) < thresh_px
            depth_ok = np.abs(z_src - depth_sampled) / (depth_sampled + 1e-8) < thresh_depth_rel
            votes += (in_bounds & reproj_ok & depth_ok).astype(np.float32)

        confidence[ref_idx] = votes / max(1, len(source_indices.get(ref_idx, [])))

    return confidence


# ---------------------------------------------------------------------------
# Depth colormap
# ---------------------------------------------------------------------------

def colorize_depth(depth, mask=None, cmap=cv2.COLORMAP_TURBO):
    """Colormap a depth map for visualization. Invalid (0) pixels are black."""
    valid = depth > 0
    if mask is not None:
        valid &= mask > 0
    if valid.sum() == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    d_min = depth[valid].min()
    d_max = depth[valid].max()
    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = (depth[valid] - d_min) / (d_max - d_min + 1e-8)
    norm_u8 = (norm * 255).clip(0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm_u8, cmap)
    colored[~valid] = 0
    return colored


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Plane-sweep MVS depth estimation (photometric, no feature matching)',
    )
    parser.add_argument('data', help='Data root with images/<cam>/, intri.yml, extri.yml')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: <data>/depths)')
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--ext', default='.jpg')
    parser.add_argument('--intri', default='intri.yml')
    parser.add_argument('--extri', default='extri.yml')
    parser.add_argument('--masks', default='masks',
                        help='Mask subdirectory (default: masks)')

    g = parser.add_argument_group('plane-sweep parameters')
    g.add_argument('--num_depths', type=int, default=128,
                   help='Depth hypotheses (default: 128)')
    g.add_argument('--num_sources', type=int, default=4,
                   help='Source views per reference (default: 4)')
    g.add_argument('--window_size', type=int, default=7,
                   help='Cost aggregation window (default: 7)')
    g.add_argument('--depth_min', type=float, default=None,
                   help='Override min depth')
    g.add_argument('--depth_max', type=float, default=None,
                   help='Override max depth')

    g = parser.add_argument_group('filtering')
    g.add_argument('--confidence_threshold', type=float, default=0.0,
                   help='Discard depths below this confidence (default: 0.0 = keep all)')
    g.add_argument(
        '--geo_consistency', action=argparse.BooleanOptionalAction, default=True,
        help='Cross-view geometric consistency filter (default: on)',
    )
    g.add_argument(
        '--undistort', action=argparse.BooleanOptionalAction, default=True,
        help='Undistort images before processing (default: on)',
    )
    g.add_argument('--device', default='cuda:0')

    args = parser.parse_args()

    out_dir = args.output if args.output else join(args.data, 'depths')
    os.makedirs(out_dir, exist_ok=True)

    # --- Load cameras ---
    intri_path = join(args.data, args.intri)
    extri_path = join(args.data, args.extri)
    print(f'[depths] Reading cameras from {intri_path}, {extri_path}')
    all_cams = read_camera(intri_path, extri_path)
    cam_names = all_cams.pop('basenames')
    N = len(cam_names)
    print(f'[depths] {N} cameras: {cam_names}')

    # --- Load images ---
    print('[depths] Loading images ...')
    images = {}
    for cam in cam_names:
        cam_dir = join(args.data, 'images', cam)
        path = join(cam_dir, f'{args.frame:06d}{args.ext}')
        if not os.path.exists(path):
            from glob import glob
            candidates = sorted(glob(join(cam_dir, f'*{args.ext}')))
            if not candidates:
                print(f'  ERROR: no images for {cam}', file=sys.stderr)
                sys.exit(1)
            path = candidates[0]
        img = cv2.imread(path)
        if img is None:
            print(f'  ERROR: cannot read {path}', file=sys.stderr)
            sys.exit(1)

        if args.undistort:
            K = all_cams[cam]['K']
            dist = all_cams[cam]['dist']
            if dist is not None and not np.allclose(dist, 0):
                img = cv2.undistort(img, K, dist, None)

        images[cam] = img
        h, w = img.shape[:2]
        print(f'  {cam}: {w}x{h}')

    # --- Load masks ---
    masks = {}
    mask_dir = join(args.data, args.masks)
    if os.path.isdir(mask_dir):
        print(f'[depths] Loading masks from {mask_dir}/')
        for cam in cam_names:
            mp = join(mask_dir, cam, f'{args.frame:06d}.png')
            if os.path.exists(mp):
                masks[cam] = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            else:
                masks[cam] = None
    else:
        print('[depths] No masks directory found — running without masks')
        for cam in cam_names:
            masks[cam] = None

    # --- Depth range ---
    Ks = [all_cams[c]['K'].astype(np.float64) for c in cam_names]
    Rs = [all_cams[c]['R'].astype(np.float64) for c in cam_names]
    Ts = [all_cams[c]['T'].astype(np.float64) for c in cam_names]
    centers = np.array([_camera_center(R, T) for R, T in zip(Rs, Ts)])

    d_min, d_max = None, None

    if args.depth_min is not None and args.depth_max is not None:
        d_min, d_max = args.depth_min, args.depth_max
        print(f'[depths] Using manual depth range: [{d_min:.4f}, {d_max:.4f}]')
    else:
        sparse_dir = join(args.data, 'colmap_ws', 'sparse', '0')
        if os.path.isdir(sparse_dir):
            print(f'[depths] Estimating depth range from COLMAP sparse points ...')
            d_min, d_max = _depth_range_from_sparse(sparse_dir, all_cams, cam_names)
            if d_min is not None:
                print(f'[depths] Depth range from sparse: [{d_min:.4f}, {d_max:.4f}]')

    if d_min is None:
        d_min, d_max = _depth_range_from_cameras(centers)
        print(f'[depths] Depth range from camera geometry: [{d_min:.4f}, {d_max:.4f}]')

    # --- Prepare torch tensors ---
    dev = torch.device(args.device)
    print(f'[depths] Device: {dev}')

    img_tensors = []
    mask_tensors = []
    for cam in cam_names:
        t = torch.from_numpy(images[cam]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensors.append(t.to(dev))
        m = masks[cam]
        if m is not None:
            mt = torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0) / 255.0
            mask_tensors.append((mt > 0.5).to(dev))
        else:
            mask_tensors.append(None)

    K_ts = [torch.from_numpy(K).float().to(dev) for K in Ks]
    R_ts = [torch.from_numpy(R).float().to(dev) for R in Rs]
    T_ts = [torch.from_numpy(T.astype(np.float32)).to(dev) for T in Ts]

    inv_depths = torch.linspace(1.0 / d_max, 1.0 / d_min, args.num_depths, device=dev)
    depths = 1.0 / inv_depths

    # --- Run plane-sweep per view ---
    t0 = time.time()
    source_map = {}
    raw_depth_maps = {}

    for ref_i in range(N):
        src_indices = _select_source_views(ref_i, centers, args.num_sources)
        source_map[ref_i] = src_indices
        ref_name = cam_names[ref_i]
        H, W = images[ref_name].shape[:2]

        src_names = [cam_names[s] for s in src_indices]
        print(f'\n[depths] View {ref_name} ({ref_i+1}/{N}), '
              f'sources={src_names}')

        with torch.no_grad():
            cost = _compute_cost_volume(
                ref_img=img_tensors[ref_i],
                src_imgs=[img_tensors[si] for si in src_indices],
                K_ref=K_ts[ref_i],
                Ks_src=[K_ts[si] for si in src_indices],
                R_ref=R_ts[ref_i],
                T_ref=T_ts[ref_i],
                Rs_src=[R_ts[si] for si in src_indices],
                Ts_src=[T_ts[si] for si in src_indices],
                depths=depths,
                window_size=args.window_size,
                mask_ref=mask_tensors[ref_i],
            )
            best_idx = cost.argmin(dim=0)
            depth_map = depths[best_idx.reshape(-1)].reshape(H, W).cpu().numpy()

        raw_depth_maps[ref_i] = depth_map.astype(np.float32)

        valid = depth_map > 0
        if masks[ref_name] is not None:
            valid &= masks[ref_name] > 0
        if valid.sum() > 0:
            print(f'  depth stats (foreground): '
                  f'min={depth_map[valid].min():.4f}, '
                  f'max={depth_map[valid].max():.4f}, '
                  f'median={np.median(depth_map[valid]):.4f}, '
                  f'valid={valid.sum():,} px')
        else:
            print('  WARNING: no valid depth pixels')

    # --- Geometric consistency ---
    if args.geo_consistency:
        print(f'\n[depths] Geometric consistency filtering ...')
        conf_maps_idx = _geometric_consistency(
            raw_depth_maps, Ks, Rs, Ts, source_map,
        )
    else:
        conf_maps_idx = {i: np.ones_like(d) for i, d in raw_depth_maps.items()}

    # --- Save results ---
    print(f'\n[depths] Saving to {out_dir}/')
    for i, name in enumerate(cam_names):
        dm = raw_depth_maps[i].copy()
        conf = conf_maps_idx.get(i, np.ones_like(dm))

        if args.confidence_threshold > 0:
            dm[conf < args.confidence_threshold] = 0.0

        mask = masks[name]

        np.save(join(out_dir, f'{name}.npy'), dm)
        np.save(join(out_dir, f'{name}_conf.npy'), conf)

        color = colorize_depth(dm, mask)
        cv2.imwrite(join(out_dir, f'{name}_color.png'), color)

        conf_u8 = (conf * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(join(out_dir, f'{name}_conf.png'), conf_u8)

        valid = dm > 0
        if mask is not None:
            valid &= mask > 0
        pct = 100 * valid.sum() / max(1, (mask > 0).sum() if mask is not None else dm.size)
        print(f'  {name}: {valid.sum():,} valid px ({pct:.1f}% of foreground)')

    elapsed = time.time() - t0
    print(f'\n{"="*60}')
    print(f'[depths] Done in {elapsed:.1f}s')
    print(f'  Output: {out_dir}/')
    print(f'  Files per view: <cam>.npy, <cam>_color.png, <cam>_conf.npy, <cam>_conf.png')
    print(f'{"="*60}')
    print(f'\nInspect depth maps:')
    print(f'  ls {out_dir}/*_color.png   # colormapped depth images')
    print(f'  # Copy to local machine and view in any image viewer')
    print(f'\nIf depths look good, proceed to TSDF fusion or re-run clean + mesh pipeline.')


if __name__ == '__main__':
    main()
