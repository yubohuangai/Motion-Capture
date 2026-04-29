"""Export a LocalDyGS checkpoint as standard 3DGS PLY at given time(s).

Output PLYs are loadable in any 3DGS viewer (SuperSplat
https://playcanvas.com/supersplat/editor in particular). One PLY per `t`
captures the cow's pose at that time; the static cow-only filter (anchor
AABB bbox-clip) is on by default since LocalDyGS uses `position_lr=0`
and anchors stay at their cow-only init-pcd locations.

Why a custom exporter (vs. LocalDyGS's own `save_ply`):
  LocalDyGS persists *anchors* with f_anchor_feat_* and f_offset_*. A 3DGS
  viewer expects the standard schema (f_dc_0..2 SH-DC, f_rest_0..44, opacity
  pre-sigmoid, scale pre-exp, normalized quaternion). So we run the model
  forward at time t to get *active* Gaussians (the things the rasterizer
  would actually draw) and write those.

Usage (run inside the localdygs venv on a GPU node — needs CUDA for the
forward pass):
    python -m apps.reconstruction.stage_b_localdygs.export_3dgs_ply \\
        --model-path  /scratch/yubo/cow_1/9148_10581_output/stage_b/train_planB_e3 \\
        --iteration   30000 \\
        --times       0.0 0.5 1.0 \\
        --output-dir  /scratch/yubo/cow_1/9148_10581_output/stage_b/train_planB_e3/exported

Output: <output-dir>/cow_t<TT>.ply  with TT = int(round(t*100)).
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from einops import repeat
from plyfile import PlyData, PlyElement

# LocalDyGS lives at ~/github/LocalDyGS — needs to be importable
LOCALDYGS_ROOT = Path.home() / 'github' / 'LocalDyGS'
sys.path.insert(0, str(LOCALDYGS_ROOT))

from scene.gaussian_model import GaussianModel  # noqa: E402

# SH order-0 normalization constant: 1 / (2*sqrt(pi)).
# rendered_color ≈ 0.5 + SH_C0 * f_dc → f_dc = (color - 0.5) / SH_C0.
SH_C0 = 0.28209479177387814

# Standard 3DGS PLY uses SH degree 3 → 15 SH bands × 3 colors = 45 f_rest fields.
# LocalDyGS outputs RGB directly (mlp_color ends in Sigmoid), so all f_rest = 0.
F_REST_DIM = 45


def load_cfg_args(model_path: Path) -> Namespace:
    """LocalDyGS saves cfg_args as `repr(Namespace(...))` text; load via eval."""
    cfg_path = model_path / 'cfg_args'
    return eval(cfg_path.read_text())


def build_and_load(cfg: Namespace, iteration: int) -> GaussianModel:
    """Build a GaussianModel matching cfg and load its checkpoint."""
    iter_dir = Path(cfg.model_path) / 'point_cloud' / f'iteration_{iteration}'
    if not iter_dir.exists():
        raise FileNotFoundError(f'No checkpoint at {iter_dir}')

    # The cfg_args file is a flat Namespace that contains both ModelParams
    # and OptimizationParams. Pass it as both `args` and `opt`.
    pc = GaussianModel(
        cfg, cfg,
        feat_dim=cfg.feat_dim,
        n_offsets=cfg.n_offsets,
        voxel_size=cfg.voxel_size,
        update_depth=cfg.update_depth,
        update_init_factor=cfg.update_init_factor,
        update_hierachy_factor=cfg.update_hierachy_factor,
        use_feat_bank=cfg.use_feat_bank,
        appearance_dim=cfg.appearance_dim,
        ratio=cfg.ratio,
        add_opacity_dist=cfg.add_opacity_dist,
        add_cov_dist=cfg.add_cov_dist,
        add_color_dist=cfg.add_color_dist,
    )
    # load_ply must come first — load_model needs get_xyz_bound() which
    # depends on _anchor.
    pc.load_ply_sparse_gaussian(str(iter_dir / 'point_cloud.ply'))
    pc.load_model(str(iter_dir))  # FDHash + MLPs + time_embedding
    pc.eval()
    return pc


@torch.no_grad()
def compute_active_gaussians(pc: GaussianModel, t: float, opt_thro: float = 0.01):
    """Mirror gaussian_renderer.generate_full_temporal_gaussians without the
    camera dependency. Returns (xyz, color, opacity_post, scaling_post, rot)
    where opacity is post-sigmoid in [0,1], scaling is post-exp positive,
    and rot is a unit quaternion. opt_thro=0.01 matches render() at iter≥20K.
    """
    anchor = pc.get_anchor  # [N, 3]
    N = anchor.shape[0]

    timestamp = torch.full((N, 1), float(t), device=anchor.device)
    if pc.hash:
        dy_feat, dy_factor = pc.dynamic_module(anchor, timestamp)
    else:
        dy_feat, dy_factor = pc.hexplane(anchor, timestamp)

    sta_feat = pc._anchor_feat
    feat = dy_factor * dy_feat + (1 - dy_factor) * sta_feat

    neural_opacity = pc.get_opacity_mlp(feat).reshape([-1, 1])  # [N*K, 1]
    mask = (neural_opacity > opt_thro).view(-1)
    opacity = neural_opacity[mask]

    color = pc.get_color_mlp(feat).reshape([N * pc.n_offsets, 3])
    scale_rot = pc.get_cov_mlp(feat).reshape([N * pc.n_offsets, 7])
    offsets = pc.get_offset_mlp(feat).view([-1, 3])
    grid_scaling = pc.get_scaling  # [N, 6] — first 3 = offset step, last 3 = Gaussian base scale

    concatenated = torch.cat([grid_scaling, anchor], dim=-1)  # [N, 9]
    concatenated_repeated = repeat(concatenated, 'n c -> (n k) c', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split(
        [6, 3, 3, 7, 3], dim=-1
    )

    # Post-process exactly as the renderer does
    scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])  # [M, 3]
    rot = pc.rotation_activation(scale_rot[:, 3:7])                    # [M, 4] unit quat
    offsets = offsets * scaling_repeat[:, :3]
    xyz = repeat_anchor + offsets                                      # [M, 3]

    return xyz, color, opacity, scaling, rot


def filter_to_aabb(xyz, *attrs, anchor_pos, padding):
    """Drop Gaussians whose centers fall outside (anchor AABB ± padding)."""
    aabb_min = anchor_pos.min(dim=0).values - padding
    aabb_max = anchor_pos.max(dim=0).values + padding
    inside = ((xyz >= aabb_min).all(dim=1) & (xyz <= aabb_max).all(dim=1))
    return tuple(t[inside] for t in (xyz,) + attrs)


def write_3dgs_ply(path: str, xyz, color, opacity_post, scaling_post, rot):
    """Write a binary little-endian PLY in the standard 3DGS schema.

    Inputs are post-activation tensors:
      color         : RGB in [0,1]               → f_dc = (color - 0.5)/SH_C0,  f_rest = 0
      opacity_post  : in [0,1] (post-sigmoid)    → opacity = logit(opacity_post)
      scaling_post  : positive (post-exp)        → scale_i = log(scaling_post_i)
      rot           : unit quaternion (w,x,y,z)  → stored as-is
    """
    xyz = xyz.detach().cpu().numpy().astype(np.float32)
    color = color.detach().cpu().numpy().astype(np.float32)
    opacity_post = opacity_post.detach().cpu().numpy().astype(np.float32).reshape(-1)
    scaling_post = scaling_post.detach().cpu().numpy().astype(np.float32)
    rot = rot.detach().cpu().numpy().astype(np.float32)

    n = xyz.shape[0]

    normals = np.zeros_like(xyz)
    f_dc = ((color - 0.5) / SH_C0).astype(np.float32)
    f_rest = np.zeros((n, F_REST_DIM), dtype=np.float32)

    # logit(p) is undefined at 0/1; clamp to a numerically safe range
    op_clip = np.clip(opacity_post, 1e-7, 1.0 - 1e-7).astype(np.float32)
    opacity_logit = np.log(op_clip / (1.0 - op_clip)).astype(np.float32).reshape(n, 1)

    scaling_log = np.log(np.clip(scaling_post, 1e-30, None)).astype(np.float32)

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
        xyz, normals, f_dc, f_rest, opacity_logit, scaling_log, rot,
    ], axis=1).astype(np.float32)

    elements = np.empty(n, dtype=dtype_full)
    elements[:] = list(map(tuple, attrs))
    el = PlyElement.describe(elements, 'vertex')

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([el]).write(path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model-path', required=True, type=Path,
                   help='LocalDyGS train output dir (containing cfg_args + point_cloud/iteration_*).')
    p.add_argument('--iteration', type=int, default=30000)
    p.add_argument('--times', type=float, nargs='+', default=[0.0, 0.5, 1.0])
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--name-prefix', default='cow_t',
                   help='Filename prefix; output is <prefix><TT>.ply where TT=int(round(t*100)).')
    p.add_argument('--no-aabb-filter', action='store_true',
                   help='Skip the cow-only AABB clipping (useful for debugging).')
    p.add_argument('--aabb-padding', type=float, default=0.05,
                   help='AABB padding in scene units (default 0.05; cfg.bounds is the scene radius).')
    p.add_argument('--opt-thro', type=float, default=0.01,
                   help='Per-Gaussian neural-opacity threshold; matches render() default at iter>=20K.')
    args = p.parse_args()

    cfg = load_cfg_args(args.model_path)
    # Force model_path to whatever the user passed (may differ from the trained-on path)
    cfg.model_path = str(args.model_path)
    print(f'cfg: hash={cfg.hash}  feat_dim={cfg.feat_dim}  n_offsets={cfg.n_offsets}  '
          f'bounds={cfg.bounds}  appearance_dim={cfg.appearance_dim}')

    pc = build_and_load(cfg, args.iteration)
    n_anchors = pc._anchor.shape[0]
    print(f'loaded iter={args.iteration}, anchors={n_anchors}')

    anchor_pos = pc.get_anchor.detach()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for t in args.times:
        xyz, color, opacity, scaling, rot = compute_active_gaussians(pc, t, opt_thro=args.opt_thro)
        n_active = xyz.shape[0]
        if not args.no_aabb_filter:
            xyz, color, opacity, scaling, rot = filter_to_aabb(
                xyz, color, opacity, scaling, rot,
                anchor_pos=anchor_pos, padding=args.aabb_padding)
        n_kept = xyz.shape[0]
        out_name = f'{args.name_prefix}{int(round(t * 100)):03d}.ply'
        out_path = args.output_dir / out_name
        write_3dgs_ply(str(out_path), xyz, color, opacity, scaling, rot)
        sz_mb = out_path.stat().st_size / (1024 ** 2)
        kept_pct = 100.0 * n_kept / max(n_active, 1)
        print(f't={t:.2f}: {n_active:>8d} active → {n_kept:>8d} kept ({kept_pct:5.1f}%)  '
              f'{sz_mb:6.1f} MB  → {out_path}')


if __name__ == '__main__':
    main()
