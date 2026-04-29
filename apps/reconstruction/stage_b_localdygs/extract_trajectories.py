"""Extract per-Gaussian trajectories from a LocalDyGS checkpoint.

For each (anchor_idx i, offset_idx k) pair, sample the spawned Gaussian's
center position at T uniformly-spaced time samples in [0, 1]. This yields
a stable-identity trajectory tensor of shape (N, K, T, 3) — the input that
Stage 3 articulation discovery needs (cluster trajectories whose pairwise
relative motion stays constant → rigid parts).

Why "stable identity": activeness (the opacity gate at threshold 0.01)
varies per timestep, but the (i, k) index pair is constant across t since
LocalDyGS has fixed anchors (position_lr=0) and fixed n_offsets per
anchor. The downstream clustering needs to compare trajectory i.k vs j.l
across all t — which only works with stable indexing.

Usage (run inside the localdygs venv on a GPU node):
    python -m apps.reconstruction.stage_b_localdygs.extract_trajectories \\
        --model-path  /scratch/yubo/cow_1/9148_10581_output/stage_b/train_planB_e1 \\
        --iteration   30000 \\
        --num-times   30 \\
        --output-dir  /scratch/.../trajectories

Output (in --output-dir):
    anchors.npy        (N, 3)        — fixed anchor positions
    positions.npy      (N, K, T, 3)  — spawned-Gaussian centers over time
    active_mask.npy    (N, K, T)     — opacity > opt_thro at each (i,k,t)
    meta.json          times, opt_thro, model_path, iteration, n_anchors, n_offsets
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

LOCALDYGS_ROOT = Path.home() / 'github' / 'LocalDyGS'
sys.path.insert(0, str(LOCALDYGS_ROOT))

from scene.gaussian_model import GaussianModel  # noqa: E402

# Reuse the loading helpers from the PLY exporter
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from export_3dgs_ply import load_cfg_args, build_and_load  # noqa: E402


@torch.no_grad()
def sample_gaussian_state(pc: GaussianModel, t: float, opt_thro: float = 0.01):
    """Return per-(anchor, offset_idx) Gaussian centers + activeness at time t.

    Returns:
        positions  : (N, K, 3) — anchor[i] + offset_mlp(feat)[i,k] * _scaling[i,:3,k]
        active     : (N, K)    — bool, opacity > opt_thro
    Both are CPU numpy arrays.
    """
    anchor = pc.get_anchor                                         # [N, 3]
    N = anchor.shape[0]
    K = pc.n_offsets

    timestamp = torch.full((N, 1), float(t), device=anchor.device)
    if pc.hash:
        dy_feat, dy_factor = pc.dynamic_module(anchor, timestamp)
    else:
        dy_feat, dy_factor = pc.hexplane(anchor, timestamp)
    sta_feat = pc._anchor_feat
    feat = dy_factor * dy_feat + (1 - dy_factor) * sta_feat        # [N, F]

    # Per-anchor offsets (scaled by anchor's first 3 _scaling dims = "step size")
    raw_offsets = pc.get_offset_mlp(feat).view(N, K, 3)            # [N, K, 3]
    grid_scaling = pc.get_scaling                                  # [N, 6] (post exp)
    step = grid_scaling[:, :3].unsqueeze(1)                        # [N, 1, 3]
    offsets = raw_offsets * step                                   # [N, K, 3]

    positions = anchor.unsqueeze(1) + offsets                      # [N, K, 3]

    # Activeness from opacity MLP
    neural_opacity = pc.get_opacity_mlp(feat)                      # [N, K]
    active = (neural_opacity > opt_thro)                           # [N, K] bool

    return positions.detach().cpu().numpy(), active.detach().cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model-path', required=True, type=Path)
    p.add_argument('--iteration', type=int, default=30000)
    p.add_argument('--num-times', type=int, default=30,
                   help='Number of evenly-spaced time samples in [0, 1].')
    p.add_argument('--opt-thro', type=float, default=0.01)
    p.add_argument('--output-dir', required=True, type=Path)
    args = p.parse_args()

    cfg = load_cfg_args(args.model_path)
    cfg.model_path = str(args.model_path)
    print(f'cfg: hash={cfg.hash}  feat_dim={cfg.feat_dim}  n_offsets={cfg.n_offsets}  bounds={cfg.bounds}')

    pc = build_and_load(cfg, args.iteration)
    N = pc._anchor.shape[0]
    K = pc.n_offsets
    T = args.num_times
    print(f'loaded iter={args.iteration}  anchors N={N}  offsets K={K}  times T={T}')

    times = np.linspace(0.0, 1.0, T, dtype=np.float32)
    positions = np.empty((N, K, T, 3), dtype=np.float32)
    active = np.empty((N, K, T), dtype=bool)

    for ti, t in enumerate(times):
        pos_t, act_t = sample_gaussian_state(pc, float(t), opt_thro=args.opt_thro)
        positions[:, :, ti, :] = pos_t
        active[:, :, ti] = act_t

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / 'anchors.npy', pc.get_anchor.detach().cpu().numpy().astype(np.float32))
    np.save(args.output_dir / 'positions.npy', positions)
    np.save(args.output_dir / 'active_mask.npy', active)
    meta = {
        'model_path': str(args.model_path),
        'iteration': int(args.iteration),
        'n_anchors': int(N),
        'n_offsets': int(K),
        'num_times': int(T),
        'times': times.tolist(),
        'opt_thro': float(args.opt_thro),
        'shapes': {
            'anchors': [N, 3],
            'positions': [N, K, T, 3],
            'active_mask': [N, K, T],
        },
    }
    (args.output_dir / 'meta.json').write_text(json.dumps(meta, indent=2))

    # Quick stats — useful sanity check before clustering
    n_active_per_t = active.sum(axis=(0, 1))           # (T,)
    n_active_always = (active.all(axis=2)).sum()       # scalar — active at every t
    n_active_ever = (active.any(axis=2)).sum()         # scalar — active at any t
    pos_displacement = np.linalg.norm(
        positions.max(axis=2) - positions.min(axis=2), axis=-1
    )                                                   # (N, K) — temporal extent per Gaussian
    print(f'wrote {args.output_dir}')
    print(f'  per-t active counts: min={n_active_per_t.min()}  '
          f'max={n_active_per_t.max()}  mean={n_active_per_t.mean():.0f}  (out of N*K = {N*K})')
    print(f'  active-always: {n_active_always:>8d}  ({100*n_active_always/(N*K):.1f}%)')
    print(f'  active-ever:   {n_active_ever:>8d}  ({100*n_active_ever/(N*K):.1f}%)')
    print(f'  per-Gaussian temporal displacement (over t∈[0,1]):')
    print(f'    p10={np.percentile(pos_displacement, 10):.4f}  '
          f'p50={np.percentile(pos_displacement, 50):.4f}  '
          f'p90={np.percentile(pos_displacement, 90):.4f}  '
          f'max={pos_displacement.max():.4f}')


if __name__ == '__main__':
    main()
