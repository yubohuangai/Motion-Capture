"""Stage B driver: scene-specific NeuS optimisation + mesh extraction.

Usage
-----
    python -m apps.reconstruction.run_stage_b <data_root> \
        --n_iters 100000 --batch_rays 512 --device cuda

By default Stage A is read from ``<data_root>_output`` and Stage B writes
to ``<data_root>_output/neus`` so artifacts sit next to the input data,
never inside the codebase. Override with ``--stage_a_output`` / ``--output``.

Reads calibration and images from ``<data_root>`` and the sparse cloud from
Stage A (used only to set the object-space normalisation — the network is
trained from scratch against images only). Produces:

    <out>/ckpt/{iter_*.pt, final.pt}
    <out>/val/<cam>_<iter>.png       # periodic re-renderings of training views
    <out>/mesh_neus.ply              # final marching-cubes mesh in world coords
    <out>/config.json                # run parameters + scene bounds
    <out>/train.log                  # training log
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from apps.reconstruction.common.cameras import load_cameras
from apps.reconstruction.common.io_utils import ensure_dir
from apps.reconstruction.run_stage_a import _read_ply_points
from apps.reconstruction.stage_b_neus.dataset import (
    NeuSDataset, scene_bounds_from_cameras, scene_bounds_from_points,
)
from apps.reconstruction.stage_b_neus.extract_mesh import extract_mesh
from apps.reconstruction.stage_b_neus.train import TrainConfig, train


def _pick_device(req: str) -> torch.device:
    if req == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(req)


def main() -> None:
    p = argparse.ArgumentParser(description="Stage B: scene-specific NeuS")
    p.add_argument("data_root", type=str)
    p.add_argument("--stage_a_output", type=str, default=None,
                   help="Stage A output dir (default: <data_root>_output)")
    p.add_argument("--output", type=str, default=None,
                   help="output dir (default: <data_root>_output/neus)")
    p.add_argument("--frame", type=int, default=0)
    # Normalisation
    p.add_argument("--bound_padding", type=float, default=1.1,
                   help="multiplicative radius padding around the sparse cloud bbox")
    p.add_argument("--bound_robust_pct", type=float, default=1.0,
                   help="percentile clip on sparse cloud when computing bounds")
    # Training
    p.add_argument("--n_iters", type=int, default=100_000)
    p.add_argument("--batch_rays", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_eikonal", type=float, default=0.1)
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--n_importance", type=int, default=64)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--val_every", type=int, default=5000)
    p.add_argument("--ckpt_every", type=int, default=10000)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--resume_from", type=str, default=None,
                   help="checkpoint path to resume from")
    # Mesh extraction
    p.add_argument("--mesh_resolution", type=int, default=256)
    p.add_argument("--skip_mesh", action="store_true")
    p.add_argument("--only_mesh", action="store_true",
                   help="skip training, load latest checkpoint, just extract mesh")
    args = p.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(f"{str(data_root)}_output")
    default_stage_a = output_root / "stage_a" / "plane_sweep"
    stage_a_dir = Path(args.stage_a_output) if args.stage_a_output else default_stage_a
    out_dir = ensure_dir(Path(args.output) if args.output else output_root / "stage_b" / "neus")
    device = _pick_device(args.device)
    print(f"[init] device={device}")

    # -- Load cameras -------------------------------------------------------
    cams = load_cameras(data_root / "intri.yml", data_root / "extri.yml")
    print(f"[init] loaded {len(cams)} cameras")

    # -- Scene bounds -------------------------------------------------------
    if (stage_a_dir / "sparse.ply").exists():
        sparse_path = stage_a_dir / "sparse.ply"
        sparse_pts = _read_ply_points(sparse_path)
        center, radius = scene_bounds_from_points(
            sparse_pts, padding=args.bound_padding,
            robust_pct=args.bound_robust_pct)
        bounds_source = f"sparse.ply ({sparse_pts.shape[0]} pts)"
    else:
        center, radius = scene_bounds_from_cameras(cams)
        bounds_source = "camera centers"
    print(f"[init] scene bounds: center={center.tolist()} radius={radius:.3f}  "
          f"(from {bounds_source})")

    # -- Dataset ------------------------------------------------------------
    dataset = NeuSDataset(data_root=data_root, cams=cams,
                          scene_center=center, scene_radius=radius,
                          frame=args.frame, device=device)
    print(f"[init] dataset: {dataset.n_views} views, {dataset.n_pixels:,} pixels")

    cfg = TrainConfig(
        n_iters=args.n_iters, batch_rays=args.batch_rays, lr=args.lr,
        weight_eikonal=args.weight_eikonal,
        n_samples=args.n_samples, n_importance=args.n_importance,
        log_every=args.log_every, val_every=args.val_every,
        ckpt_every=args.ckpt_every, device=str(device),
    )

    # Persist scene bounds + args so --only_mesh can reproduce the transform
    with open(out_dir / "config.json", "w") as f:
        json.dump({
            "args": vars(args),
            "scene_center": center.tolist(),
            "scene_radius": radius,
            "bounds_source": bounds_source,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2, default=str)

    # -- Train --------------------------------------------------------------
    if args.only_mesh:
        from apps.reconstruction.stage_b_neus.models import (
            ColorNetwork, SDFNetwork, SingleVariance,
        )
        from apps.reconstruction.stage_b_neus.train import load_checkpoint
        sdf_net = SDFNetwork().to(device)
        color_net = ColorNetwork().to(device)
        variance = SingleVariance(init_val=0.3).to(device)
        ckpt = args.resume_from or (out_dir / "ckpt" / "final.pt")
        ckpt = Path(ckpt)
        if not ckpt.exists():
            raise FileNotFoundError(f"no checkpoint for --only_mesh: {ckpt}")
        load_checkpoint(ckpt, sdf_net, color_net, variance)
        print(f"[only_mesh] loaded {ckpt}")
        result = {"sdf_net": sdf_net, "color_net": color_net, "variance": variance}
    else:
        result = train(dataset, out_dir=out_dir, cfg=cfg,
                       resume_from=args.resume_from)

    # -- Extract mesh -------------------------------------------------------
    if not args.skip_mesh:
        extract_mesh(result["sdf_net"], result["color_net"], dataset,
                     out_path=out_dir / "mesh_neus.ply",
                     resolution=args.mesh_resolution,
                     bbox_radius=1.0, iso=0.0, device=device)
    print(f"[done] artifacts in {out_dir}")


if __name__ == "__main__":
    main()
