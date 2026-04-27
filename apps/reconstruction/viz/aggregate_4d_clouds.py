"""Combine per-frame fused.ply outputs into one color-coded PLY for review.

Each frame's points get painted with a distinct turbo-colormap color based
on frame index, so the resulting PLY shows the cow's motion trajectory at
a glance: opening this single file in MeshLab is faster than juggling N
separate clouds.

Usage
-----
    # Default: read the flat layout, write aggregated_4d.ply alongside
    python -m apps.reconstruction.viz.aggregate_4d_clouds \
        /scratch/yubo/cow_1/9148_10581_output/stage_a/colmap_4d/human

    # Override with --pattern / --output if your layout differs
    python -m apps.reconstruction.viz.aggregate_4d_clouds <root> \
        --pattern 'frame_*/dense/fused.ply'        # old per-frame layout

Optional: ``--max_per_frame N`` randomly subsamples each frame to N points
to keep the PLY manageable when the full union would be hundreds of MB.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _turbo_rgb(t: float) -> tuple[int, int, int]:
    """Inline turbo colormap (Mikhailov 2019) — avoids matplotlib dep."""
    t = max(0.0, min(1.0, float(t)))
    # 4-piece polynomial fit, sufficient for a coarse temporal palette
    r = 0.13572138 + 4.61539260 * t - 42.66032258 * t**2 + 132.13108234 * t**3 \
        - 152.94239396 * t**4 + 59.28637943 * t**5
    g = 0.09140261 + 2.19418839 * t + 4.84296658 * t**2 - 14.18503333 * t**3 \
        + 4.27729857 * t**4 + 2.82956604 * t**5
    b = 0.10667330 + 12.64194608 * t - 60.58204836 * t**2 + 110.36276771 * t**3 \
        - 89.90310912 * t**4 + 27.34824973 * t**5
    return (int(np.clip(r * 255, 0, 255)),
            int(np.clip(g * 255, 0, 255)),
            int(np.clip(b * 255, 0, 255)))


def _read_ply_xyz(path: Path) -> np.ndarray:
    """Minimal PLY reader: returns (N, 3) float32. Handles both binary and ASCII."""
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"]
    return np.stack([np.asarray(v["x"], dtype=np.float32),
                     np.asarray(v["y"], dtype=np.float32),
                     np.asarray(v["z"], dtype=np.float32)], axis=1)


def _write_ply_xyz_rgb(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write a binary PLY with vertex positions + uchar RGB."""
    from plyfile import PlyData, PlyElement
    n = xyz.shape[0]
    arr = np.empty(n, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))


def _frame_index(path: Path) -> int:
    """Extract integer frame index from a path or filename containing 'frame_<N>...'."""
    # Try filename first (flat layout: frame_000030_fused.ply)
    for token in [path.stem, *path.parts]:
        if token.startswith("frame_"):
            digits = ""
            for ch in token.split("_", 1)[1]:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                return int(digits)
    return -1


def main() -> None:
    p = argparse.ArgumentParser(
        description="Aggregate 4D per-frame clouds into one color-coded PLY",
    )
    p.add_argument("root", type=str,
                   help="root to glob (typically <out>/human/ in flat layout)")
    p.add_argument("--pattern", default="frame_*_fused.ply",
                   help="glob pattern relative to <root> "
                        "(default: flat layout 'frame_*_fused.ply'). "
                        "Use 'frame_*/human/fused.ply' for the prior nested "
                        "layout, or 'frame_*/dense/fused.ply' for raw COLMAP.")
    p.add_argument("--output", "-o", default=None,
                   help="output PLY (default: <root>/aggregated_4d.ply)")
    p.add_argument("--max_per_frame", type=int, default=None,
                   help="randomly subsample each frame to N points (default: keep all)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for subsampling (default: 42)")
    args = p.parse_args()

    root = Path(args.root)
    out = Path(args.output) if args.output else (root / "aggregated_4d.ply")

    plys = sorted(root.glob(args.pattern))
    if not plys:
        raise SystemExit(f"[aggregate_4d] no PLYs match {args.pattern!r} under {root}")
    print(f"[aggregate_4d] found {len(plys)} per-frame clouds")

    frames = [_frame_index(p) for p in plys]
    f_min, f_max = min(frames), max(frames)
    span = max(1, f_max - f_min)

    rng = np.random.default_rng(args.seed)
    all_xyz, all_rgb = [], []
    for ply, f in zip(plys, frames):
        xyz = _read_ply_xyz(ply)
        n = xyz.shape[0]
        if args.max_per_frame is not None and n > args.max_per_frame:
            idx = rng.choice(n, args.max_per_frame, replace=False)
            xyz = xyz[idx]
        t = (f - f_min) / span
        r, g, b = _turbo_rgb(t)
        rgb = np.tile([r, g, b], (xyz.shape[0], 1)).astype(np.uint8)
        all_xyz.append(xyz)
        all_rgb.append(rgb)
        print(f"  frame {f:>6d}: {n:>9,d} pts  (color = turbo({t:.2f}) = "
              f"({r},{g},{b}))")

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    _write_ply_xyz_rgb(out, xyz, rgb)
    size_mb = out.stat().st_size / 1e6
    print(f"\n[aggregate_4d] wrote {out} ({xyz.shape[0]:,} pts, {size_mb:.1f} MB)")
    print(f"[aggregate_4d] open in MeshLab or 'open3d.visualization.draw_geometries' "
          f"— blue = early frames, red = late frames")


if __name__ == "__main__":
    main()
