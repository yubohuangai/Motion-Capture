"""Cluster per-Gaussian trajectories into rigid parts (Stage 3 prototype).

Reads `extract_trajectories.py` output, filters to Gaussians that are
active across all sampled timesteps, runs K-means in the chosen feature
space, and writes a single static PLY where each Gaussian is colored by
its cluster label. Drag the PLY into SuperSplat to inspect the partition.

This is a baseline ("does k-means in trajectory space find anything
sensible?") — not the final algorithm. Better approaches to try later:
trajectory-relative-motion variance affinity → spectral clustering;
RANSAC over part-pair offsets that stay constant; HDBSCAN on flattened
trajectories. See PROJECT.md §9 (Stage 3) for the research direction.

Usage (run inside the localdygs venv):
    python -m apps.reconstruction.stage_b_localdygs.cluster_trajectories \\
        --traj-dir   /scratch/.../train_planB_e1/trajectories \\
        --K          8 \\
        --feature    centered \\
        --output-ply /scratch/.../train_planB_e1/exported/cow_clusters_K8.ply

Feature modes:
    flat         — raw (x,y,z) over time, flattened: T*3 dims.
                   Sensitive to absolute position; clusters spatially.
    centered     — subtract per-Gaussian mean position before flattening.
                   Captures motion *pattern* independent of where the
                   Gaussian sits → closer to "rigid co-movement".
    displacement — pos(t) - pos(0): clusters by motion trajectory shape.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.cluster.vq import kmeans2

SH_C0 = 0.28209479177387814
F_REST_DIM = 45

# 8-color qualitative palette (matplotlib tab10 minus two)
_PALETTE = np.array([
    [0.99, 0.39, 0.40],
    [0.31, 0.69, 0.86],
    [0.45, 0.78, 0.39],
    [0.97, 0.81, 0.39],
    [0.73, 0.40, 0.85],
    [0.99, 0.55, 0.21],
    [0.37, 0.73, 0.71],
    [0.86, 0.60, 0.86],
], dtype=np.float32)


def make_palette(K: int, rng_seed: int) -> np.ndarray:
    if K <= len(_PALETTE):
        return _PALETTE[:K]
    rng = np.random.default_rng(rng_seed)
    extra = rng.uniform(size=(K - len(_PALETTE), 3)).astype(np.float32)
    return np.vstack([_PALETTE, extra])


def write_static_3dgs_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    """Write a static "just dots" 3DGS PLY: high opacity, tiny isotropic
    Gaussians. SuperSplat-readable; great for cluster visualization."""
    n = xyz.shape[0]
    normals = np.zeros_like(xyz, dtype=np.float32)
    f_dc = ((rgb - 0.5) / SH_C0).astype(np.float32)
    f_rest = np.zeros((n, F_REST_DIM), dtype=np.float32)
    opacity_logit = np.full((n, 1), 5.0, dtype=np.float32)   # sigmoid(5) ≈ 0.993
    scale_log = np.full((n, 3), -6.0, dtype=np.float32)      # exp(-6) ≈ 2.5e-3
    rot = np.zeros((n, 4), dtype=np.float32)
    rot[:, 0] = 1.0                                          # identity quat (w=1)

    properties = (
        ['x', 'y', 'z', 'nx', 'ny', 'nz']
        + [f'f_dc_{i}' for i in range(3)]
        + [f'f_rest_{i}' for i in range(F_REST_DIM)]
        + ['opacity']
        + [f'scale_{i}' for i in range(3)]
        + [f'rot_{i}' for i in range(4)]
    )
    dtype_full = [(p, 'f4') for p in properties]
    attrs = np.concatenate([
        xyz.astype(np.float32), normals, f_dc, f_rest, opacity_logit, scale_log, rot,
    ], axis=1)
    elements = np.empty(n, dtype=dtype_full)
    elements[:] = list(map(tuple, attrs))
    el = PlyElement.describe(elements, 'vertex')
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([el]).write(str(path))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--traj-dir', required=True, type=Path,
                   help='output dir from extract_trajectories.py')
    p.add_argument('--K', type=int, default=8, help='number of clusters')
    p.add_argument('--feature', choices=['flat', 'centered', 'displacement'],
                   default='centered')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output-ply', required=True, type=Path)
    p.add_argument('--max-gaussians', type=int, default=200_000,
                   help='subsample if active-always Gaussians exceed this (kmeans2 is O(N*K*iters))')
    args = p.parse_args()

    positions = np.load(args.traj_dir / 'positions.npy')          # (N, K, T, 3)
    active = np.load(args.traj_dir / 'active_mask.npy')           # (N, K, T)
    meta = json.loads((args.traj_dir / 'meta.json').read_text())
    N, Kg, T, _ = positions.shape
    print(f'loaded: N={N}  K={Kg}  T={T}  (model={meta.get("model_path")})')

    pos_flat = positions.reshape(N * Kg, T, 3)
    act_flat = active.reshape(N * Kg, T)
    keep = act_flat.all(axis=1)
    M_total = int(keep.sum())
    print(f'active-always Gaussians: {M_total} ({100 * M_total / (N * Kg):.1f}% of N*K)')

    if M_total > args.max_gaussians:
        rng = np.random.default_rng(args.seed)
        candidate_ids = np.where(keep)[0]
        chosen = rng.choice(candidate_ids, size=args.max_gaussians, replace=False)
        keep_idx = np.zeros_like(keep)
        keep_idx[chosen] = True
        keep = keep_idx
        print(f'subsampled to {args.max_gaussians} for clustering')

    pos_kept = pos_flat[keep]                                     # (M, T, 3)
    M = pos_kept.shape[0]

    if args.feature == 'flat':
        feats = pos_kept.reshape(M, T * 3).astype(np.float32)
    elif args.feature == 'centered':
        center = pos_kept.mean(axis=1, keepdims=True)
        feats = (pos_kept - center).reshape(M, T * 3).astype(np.float32)
    elif args.feature == 'displacement':
        feats = (pos_kept - pos_kept[:, 0:1, :]).reshape(M, T * 3).astype(np.float32)
    else:
        raise ValueError(args.feature)

    print(f'kmeans2: M={M}  feature={args.feature}  dim={feats.shape[1]}  K={args.K}')
    _centroids, labels = kmeans2(feats, args.K, seed=args.seed, minit='++')

    palette = make_palette(args.K, rng_seed=args.seed)
    rgb = palette[labels]                                          # (M, 3) in [0,1]

    # Use mean trajectory position as the static viz position (motion-blurred dot)
    xyz_viz = pos_kept.mean(axis=1)

    write_static_3dgs_ply(args.output_ply, xyz_viz, rgb)
    sz = args.output_ply.stat().st_size / (1024 ** 2)
    print(f'wrote {args.output_ply} ({sz:.1f} MB)')

    counts = np.bincount(labels, minlength=args.K)
    print('per-cluster size: ' + ', '.join(f'k{i}={c}' for i, c in enumerate(counts)))


if __name__ == '__main__':
    main()
