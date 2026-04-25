"""Stage A end-to-end driver: sparse + dense MVS + Poisson mesh.

Usage
-----
    python apps/reconstruction_classical/run_stage_a.py <data_root> \
        --n_depths 128 \
        --max_sources 4

By default outputs go to ``<data_root>_output`` so artifacts sit next to
the input data, never inside the codebase. Pass ``--output`` to override.

``<data_root>`` is expected to contain::

    <root>/
      intri.yml
      extri.yml
      images/
        01/000000.jpg
        02/000000.jpg
        ...

Outputs
-------
    <out>/sparse.ply                # triangulated feature tracks
    <out>/depth/<cam>.npz           # per-view depth + confidence + colored preview
    <out>/fused.ply                 # dense oriented + colored point cloud
    <out>/mesh.ply                  # Poisson mesh (if open3d available)
    <out>/config.json               # run parameters + summary statistics
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from apps.reconstruction_classical.common.cameras import load_cameras
from apps.reconstruction_classical.common.images import load_views, load_masks
from apps.reconstruction_classical.common.io_utils import (
    ensure_dir, timed, write_ply_points,
)
from apps.reconstruction_classical.stage_a_classical.sparse import (
    sparse_reconstruct, tracks_to_point_cloud,
)
from apps.reconstruction_classical.stage_a_classical.mvs_plane_sweep import (
    compute_all_depth_maps, DepthMap,
)
from apps.reconstruction_classical.stage_a_classical.fuse import (
    fuse_depth_maps, voxel_downsample, statistical_outlier_removal,
)
from apps.reconstruction_classical.stage_a_classical.mesh import (
    poisson_mesh, save_mesh, save_fused_cloud, crop_to_bbox,
)


# ---------------------------------------------------------------------------


def _parse_float3(s: str) -> np.ndarray:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected 'x,y,z', got {s!r}")
    return np.array(parts, dtype=np.float32)


def _save_depth_preview(dm: DepthMap, out_path: Path) -> None:
    """Write a colorized depth PNG alongside the .npz file."""
    d = dm.depth.copy()
    mask = d > 0
    if not mask.any():
        return
    d_min = float(np.percentile(d[mask], 2))
    d_max = float(np.percentile(d[mask], 98))
    d_norm = np.clip((d - d_min) / max(d_max - d_min, 1e-6), 0.0, 1.0)
    d_u8 = (d_norm * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)
    color[~mask] = 0
    cv2.imwrite(str(out_path), color)


def main() -> None:
    p = argparse.ArgumentParser(description="Stage A classical MVS reconstruction")
    p.add_argument("data_root", type=str)
    p.add_argument("--output", type=str, default=None,
                   help="output directory (default: <data_root>_output)")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--n_depths", type=int, default=128)
    p.add_argument("--max_sources", type=int, default=4)
    p.add_argument("--ncc_ksize", type=int, default=7)
    p.add_argument("--aggregate", type=str, default="top2", choices=["mean", "top2"])
    p.add_argument("--min_conf", type=float, default=0.25,
                   help="min peak-vs-mean cost-gap confidence for a depth pixel")
    p.add_argument("--median_ksize", type=int, default=5)
    p.add_argument("--bilateral_d", type=int, default=5,
                   help="joint bilateral depth smoothing diameter (0=off)")
    p.add_argument("--bilateral_sigma_color", type=float, default=20.0)
    p.add_argument("--bilateral_sigma_space", type=float, default=5.0)
    p.add_argument("--texture_var_thr", type=float, default=2.0,
                   help="min grayscale variance for a reference pixel to be trusted (0=off)")
    p.add_argument("--rel_tol", type=float, default=0.02,
                   help="fusion: relative depth tolerance for cross-view consistency")
    p.add_argument("--min_consistent", type=int, default=2,
                   help="fusion: minimum agreeing views for a point to survive")
    p.add_argument("--max_normal_deg", type=float, default=60.0,
                   help="fusion: max angular disagreement between per-view "
                        "surface normals (degrees) for a view to count as "
                        "consistent; large values effectively disable the check")
    p.add_argument("--voxel_size", type=float, default=0.01)
    p.add_argument("--poisson_depth", type=int, default=9)
    p.add_argument("--poisson_density_pct", type=float, default=5.0)
    p.add_argument("--crop_center", type=_parse_float3, default=None,
                   help="optional 3D bbox center 'x,y,z' (meters)")
    p.add_argument("--crop_extent", type=_parse_float3, default=None,
                   help="optional 3D bbox extent 'dx,dy,dz' (meters)")
    p.add_argument("--skip_sparse", action="store_true")
    p.add_argument("--skip_mvs", action="store_true")
    p.add_argument("--skip_mesh", action="store_true")
    p.add_argument("--only_sparse", action="store_true",
                   help="run sparse step only; skip MVS, fusion, mesh")
    p.add_argument("--no_masks", action="store_true",
                   help="ignore <data_root>/masks/ even if present")
    args = p.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output) if args.output else Path(f"{str(data_root)}_output")
    out_dir = ensure_dir(output_dir)
    intri = data_root / "intri.yml"
    extri = data_root / "extri.yml"
    assert intri.exists() and extri.exists(), f"missing calibration in {data_root}"

    cams = load_cameras(intri, extri)
    print(f"[init] loaded {len(cams)} cameras from {data_root}")

    # -------- Sparse step (native resolution by default) ---------------------
    sparse_ply = out_dir / "sparse.ply"
    sparse_points: np.ndarray
    if args.skip_sparse and sparse_ply.exists():
        print(f"[sparse] loading existing {sparse_ply}")
        import open3d as o3d  # noqa: F401
        from numpy.lib import recfunctions  # noqa: F401
        # lightweight loader avoiding open3d dependency for this branch
        sparse_points = _read_ply_points(sparse_ply)
    else:
        sparse_views, sparse_cams = load_views(data_root, cams, frame=args.frame)
        sparse_masks = None
        if not args.no_masks:
            target_hw = {n: v.shape[:2] for n, v in sparse_views.items()}
            sparse_masks = load_masks(data_root, sparse_cams.keys(),
                                      frame=args.frame, target_hw=target_hw)
            if sparse_masks:
                print(f"[sparse] using foreground masks for "
                      f"{len(sparse_masks)}/{len(sparse_cams)} cameras")
            else:
                print(f"[sparse] no masks/ directory found under {data_root}; "
                      f"running on full images")
        with timed("sparse"):
            tracks, _feats, _pairs = sparse_reconstruct(
                sparse_views, sparse_cams,
                ratio=0.75, max_epi_px=2.0, max_reproj_err=2.5, min_views=3,
                masks=sparse_masks)
        if not tracks:
            raise RuntimeError("sparse step produced zero tracks; aborting")
        pts, col = tracks_to_point_cloud(tracks)
        write_ply_points(sparse_ply, pts, col)
        sparse_points = pts
        print(f"[sparse] wrote {sparse_ply} ({pts.shape[0]} pts)")

    # -------- Sparse-only short-circuit -------------------------------------
    if args.only_sparse:
        summary = {
            "data_root": str(data_root),
            "frame": args.frame,
            "n_sparse_points": int(sparse_points.shape[0]),
            "cameras": list(cams.keys()),
            "mode": "sparse_only",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(out_dir / "config.json", "w") as f:
            json.dump({"args": vars(args), "summary": summary}, f, indent=2, default=str)
        print(f"[done] sparse-only run; artifacts in {out_dir}")
        return

    # -------- Dense MVS -----------------------------------------------------
    mvs_masks = None
    if args.skip_mvs:
        depth_maps = _load_depth_maps(out_dir / "depth")
        assert depth_maps, "no cached depth maps and --skip_mvs was set"
        views, cams_mvs = load_views(data_root, cams, frame=args.frame)
    else:
        views, cams_mvs = load_views(data_root, cams, frame=args.frame)
        if not args.no_masks:
            target_hw = {n: v.shape[:2] for n, v in views.items()}
            mvs_masks = load_masks(data_root, cams_mvs.keys(),
                                   frame=args.frame, target_hw=target_hw)
            if mvs_masks:
                print(f"[mvs] using foreground masks for "
                      f"{len(mvs_masks)}/{len(cams_mvs)} cameras")
        with timed("mvs"):
            depth_maps = compute_all_depth_maps(
                views, cams_mvs, sparse_points,
                n_depths=args.n_depths, max_sources=args.max_sources,
                ncc_ksize=args.ncc_ksize, aggregate=args.aggregate,
                min_conf=args.min_conf, median_ksize=args.median_ksize,
                bilateral_d=args.bilateral_d,
                bilateral_sigma_color=args.bilateral_sigma_color,
                bilateral_sigma_space=args.bilateral_sigma_space,
                texture_var_thr=args.texture_var_thr,
                masks=mvs_masks)
        depth_dir = ensure_dir(out_dir / "depth")
        for name, dm in depth_maps.items():
            np.savez_compressed(depth_dir / f"{name}.npz",
                                depth=dm.depth, confidence=dm.confidence)
            _save_depth_preview(dm, depth_dir / f"{name}.png")

    # -------- Fusion --------------------------------------------------------
    # Also plumb masks here — if MVS was cached from a pre-mask run, this
    # still excludes background depths from the fused cloud.
    if args.skip_mvs and not args.no_masks:
        target_hw = {n: v.shape[:2] for n, v in views.items()}
        mvs_masks = load_masks(data_root, cams_mvs.keys(),
                               frame=args.frame, target_hw=target_hw)
    with timed("fuse"):
        cloud = fuse_depth_maps(views, cams_mvs, depth_maps,
                                rel_tol=args.rel_tol,
                                min_consistent=args.min_consistent,
                                max_normal_deg=args.max_normal_deg,
                                masks=mvs_masks)
    print(f"[fuse] raw fused cloud: {cloud.points.shape[0]} points")
    if args.crop_center is not None and args.crop_extent is not None:
        cloud = crop_to_bbox(cloud, args.crop_center, args.crop_extent)
        print(f"[fuse] after crop: {cloud.points.shape[0]} points")
    if args.voxel_size > 0 and cloud.points.shape[0] > 0:
        cloud = voxel_downsample(cloud, args.voxel_size)
        print(f"[fuse] after voxel {args.voxel_size}m: {cloud.points.shape[0]} points")
    if cloud.points.shape[0] > 200:
        cloud = statistical_outlier_removal(cloud, k=16, std_ratio=2.0)
        print(f"[fuse] after outlier removal: {cloud.points.shape[0]} points")
    save_fused_cloud(cloud, out_dir / "fused.ply")
    print(f"[fuse] wrote {out_dir / 'fused.ply'}")

    # -------- Poisson mesh --------------------------------------------------
    if not args.skip_mesh and cloud.points.shape[0] > 500:
        with timed("poisson"):
            mesh, _ = poisson_mesh(cloud, depth=args.poisson_depth,
                                   density_percentile=args.poisson_density_pct)
        if mesh is not None:
            save_mesh(mesh, out_dir / "mesh.ply")
            print(f"[poisson] wrote {out_dir / 'mesh.ply'}")
    else:
        print("[poisson] skipped")

    # -------- Summary -------------------------------------------------------
    summary = {
        "data_root": str(data_root),
        "frame": args.frame,
        "n_depths": args.n_depths,
        "n_sparse_points": int(sparse_points.shape[0]),
        "n_fused_points": int(cloud.points.shape[0]),
        "cameras": list(cams.keys()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump({"args": vars(args), "summary": summary}, f, indent=2, default=str)
    print(f"[done] artifacts in {out_dir}")


# ---------------------------------------------------------------------------
# Tiny helpers for --skip paths
# ---------------------------------------------------------------------------


def _read_ply_points(path: Path) -> np.ndarray:
    """Minimal PLY point reader (binary, xyz only)."""
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.strip() == b"end_header":
                break
        data = f.read()
    # parse element vertex count
    for line in header.splitlines():
        if line.startswith(b"element vertex"):
            n = int(line.split()[-1])
            break
    else:
        raise RuntimeError("no vertex count in header")
    # robust enough: assume xyz are the first three float32 fields
    stride = 0
    props = []
    for line in header.splitlines():
        if line.startswith(b"property"):
            toks = line.split()
            t = toks[1]; name = toks[-1]
            props.append((t.decode(), name.decode()))
            stride += {"float": 4, "uchar": 1, "int": 4}[t.decode()]
    arr = np.frombuffer(data, dtype=np.uint8).reshape(n, stride)
    pts = np.empty((n, 3), np.float32)
    off = 0
    for i, (t, nm) in enumerate(props):
        if nm == "x":
            pts[:, 0] = arr[:, off:off + 4].view(np.float32)[:, 0]
        if nm == "y":
            pts[:, 1] = arr[:, off:off + 4].view(np.float32)[:, 0]
        if nm == "z":
            pts[:, 2] = arr[:, off:off + 4].view(np.float32)[:, 0]
        off += {"float": 4, "uchar": 1, "int": 4}[t]
    return pts


def _load_depth_maps(depth_dir: Path) -> Dict[str, DepthMap]:
    out: Dict[str, DepthMap] = {}
    for fp in sorted(depth_dir.glob("*.npz")):
        data = np.load(fp)
        out[fp.stem] = DepthMap(cam_name=fp.stem,
                                depth=data["depth"].astype(np.float32),
                                confidence=data["confidence"].astype(np.float32))
    return out


if __name__ == "__main__":
    main()
