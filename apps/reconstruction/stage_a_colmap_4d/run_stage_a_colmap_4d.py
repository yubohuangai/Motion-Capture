"""Stage A 4D extension: per-frame COLMAP MVS for a moving subject in a fixed multi-view rig.

Why this exists
---------------
The single-frame ``stage_a_colmap`` reconstructs only what the cameras see at
one instant. A half-circle rig misses the back of the cow. But if the cow
*moves* through the rig (e.g. walks 360° over a few seconds), the union of
per-frame reconstructions covers more of the surface — the cameras stayed
put, the cow rotated.

This driver runs COLMAP MVS on a sequence of timestamps, exploiting two
facts to avoid redundant work:

1. **Cameras don't move across frames.** The COLMAP sparse model
   (``cameras.bin`` + ``images.bin``) is the same for every timestamp, so
   we build it once and symlink it into per-frame workspaces.
2. **Per-frame work is just dense MVS.** image_undistorter +
   patch_match_stereo + stereo_fusion, each operating on that frame's
   images plus the shared sparse model.

Output is one ``fused.ply`` per timestamp, all in the same world coords
(so they overlay directly). An optional aggregation step over the per-frame
clouds is left to ``tools/clean_pointcloud.py`` or your viewer of choice.

Usage
-----
    python -m apps.reconstruction.stage_a_colmap_4d.run_stage_a_colmap_4d \
        /scratch/yubo/cow_1/9148_10581 --frames 0:150:30
    # 5 timestamps spaced 1 sec apart at 30 fps

    # All explicit frames
    python -m apps.reconstruction.stage_a_colmap_4d.run_stage_a_colmap_4d \
        /scratch/yubo/cow_1/9148_10581 --frames 0,30,60,90,120

Output (default ``<data_root>_output/stage_a/colmap_4d/``)::

    shared/
      sparse/0/{cameras.bin, images.bin, points3D.bin}   # built once
      images/  ...                                        # frame-0 images (from setup)
    frame_000000/
      sparse → ../shared/sparse                           # symlink
      images/{01.jpg, 02.jpg, ...}                        # symlinks to this frame
      dense/
        fused.ply                                         # this timestamp's dense cloud
    frame_000030/
      ...

To timelapse the result: open every ``frame_*/dense/fused.ply`` in
MeshLab; or aggregate via voxel-downsample in
``tools/clean_pointcloud.py`` if you trust the per-frame alignment.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("[stage_a_colmap_4d] $", " ".join(shlex.quote(c) for c in cmd), flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _parse_frames(s: str) -> list[int]:
    """``START:END[:STEP]`` (range, end-exclusive) or ``N1,N2,N3`` (explicit)."""
    if ":" in s:
        parts = [int(p) for p in s.split(":")]
        if len(parts) == 2:
            return list(range(parts[0], parts[1]))
        if len(parts) == 3:
            return list(range(parts[0], parts[1], parts[2]))
        raise ValueError(f"--frames range needs 2 or 3 colon-separated ints; got {s!r}")
    return [int(p) for p in s.split(",") if p.strip()]


def _setup_shared(data_root: Path, ref_frame: int, shared_dir: Path,
                  ext: str, intri: str, extri: str, colmap: str,
                  use_gpu: bool) -> None:
    """Build the COLMAP camera model + (empty) sparse workspace once.

    Uses ``--no-undistort`` so cameras keep distortion params (OPENCV model);
    per-frame ``image_undistorter`` will then handle undistortion using each
    timestamp's own raw images.

    Skips SIFT triangulation: it would force PINHOLE cameras in the rewrite
    step of ``export_colmap.triangulate_points`` and mismatch the OPENCV
    cameras we wrote. Instead, we compute explicit depth bounds from camera
    geometry and pass them to ``patch_match_stereo`` via ``--depth_min/max``.
    """
    if (shared_dir / "sparse" / "0" / "cameras.bin").exists():
        print(f"[stage_a_colmap_4d] shared sparse model already at {shared_dir}; reusing.")
        return
    cmd = [
        sys.executable, "-m",
        "apps.reconstruction.stage_a_colmap.export_colmap",
        str(data_root),
        "--output", str(shared_dir),
        "--frame", str(ref_frame),
        "--intri", intri,
        "--extri", extri,
        "--ext", ext,
        "--no-undistort",
        "--no-triangulate",  # depth bounds come from --depth_min/max instead
        "--no-mask",
        "--colmap", colmap,
    ]
    if use_gpu:
        cmd.append("--gpu")
    else:
        cmd.append("--no-gpu")
    _run(cmd)


def _link_frame_workspace(data_root: Path, frame: int, frame_dir: Path,
                          shared_dir: Path, cam_names: list[str], ext: str) -> None:
    """Build a per-frame COLMAP workspace via symlinks.

    Layout produced::
        <frame_dir>/
          sparse → <shared_dir>/sparse                    (symlink)
          images/<cam_name>.<ext> → <data_root>/images/<cam_name>/<frame>.<ext>
    """
    frame_dir.mkdir(parents=True, exist_ok=True)
    sparse_link = frame_dir / "sparse"
    if sparse_link.is_symlink() or sparse_link.exists():
        sparse_link.unlink() if sparse_link.is_symlink() else None
    if not sparse_link.exists():
        sparse_link.symlink_to((shared_dir / "sparse").resolve(), target_is_directory=True)

    img_dir = frame_dir / "images"
    img_dir.mkdir(exist_ok=True)
    for cam in cam_names:
        src = data_root / "images" / cam / f"{frame:06d}{ext}"
        if not src.exists():
            raise FileNotFoundError(
                f"missing image for frame {frame} cam {cam}: {src}")
        dst = img_dir / f"{cam}{ext}"
        if dst.is_symlink():
            dst.unlink()
        if not dst.exists():
            dst.symlink_to(src.resolve())


def _run_dense(frame_dir: Path, neighbor: int, gpu_index: str, colmap: str,
               depth_min: float, depth_max: float) -> None:
    cmd = [
        sys.executable, "-m",
        "apps.reconstruction.stage_a_colmap.dense_reconstruct",
        str(frame_dir),
        "--neighbor", str(neighbor),
        "--gpu_index", gpu_index,
        "--no-mask",
        "--colmap", colmap,
        "--depth_min", str(depth_min),
        "--depth_max", str(depth_max),
    ]
    _run(cmd)


def _auto_depth_bounds(data_root: Path, intri: str, extri: str) -> tuple[float, float]:
    """Compute conservative [depth_min, depth_max] from camera-rig geometry.

    Uses **rig diameter** (max pairwise camera distance) rather than
    distance-to-centroid: for a rig looking inward at a subject, the
    subject lives roughly at ~half the rig diameter from any camera.
    Distance-to-centroid is unreliable when cameras cluster asymmetrically
    (e.g. one cam near the world origin pulls the centroid off the subject).

    Returns ``(0.25 × diameter, 1.0 × diameter)``: wide enough to
    accommodate subjects of ~half-rig-radius extent, narrow enough that
    PatchMatch hypotheses concentrate on plausible depths.
    """
    import numpy as np
    from scipy.spatial.distance import pdist
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from easymocap.mytools.camera_utils import read_camera
    cams = read_camera(str(data_root / intri), str(data_root / extri))
    names = cams.pop("basenames")
    centers = np.array([-cams[n]["R"].T @ cams[n]["T"].flatten() for n in names])
    rig_diameter = float(pdist(centers).max())
    return max(0.5, rig_diameter * 0.25), rig_diameter


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stage A 4D: per-frame COLMAP MVS over a sequence",
    )
    p.add_argument("data_root", type=str)
    p.add_argument("--output", type=str, default=None,
                   help="output dir (default: <data_root>_output/stage_a/colmap_4d/)")
    p.add_argument("--frames", type=str, required=True,
                   help="frame range 'START:END[:STEP]' or explicit 'N1,N2,N3'")
    p.add_argument("--ref_frame", type=int, default=None,
                   help="frame index used to build the shared sparse model "
                        "(default: first frame in --frames)")
    p.add_argument("--intri", default="intri.yml")
    p.add_argument("--extri", default="extri.yml")
    p.add_argument("--ext", default=".jpg")
    p.add_argument("--neighbor", "-k", type=int, default=6,
                   help="K nearest cameras for PatchMatch sources")
    p.add_argument("--colmap", default="colmap")
    p.add_argument("--no-gpu", action="store_true",
                   help="disable GPU for SIFT (setup phase); patch_match still uses GPU")
    p.add_argument("--gpu_index", default="0",
                   help="GPU index for patch_match_stereo (default: 0)")
    p.add_argument("--skip_setup", action="store_true",
                   help="reuse existing shared sparse workspace; skip export_colmap")
    p.add_argument("--depth_min", type=float, default=None,
                   help="PatchMatch min depth (default: 0.3× nearest cam-to-centroid dist)")
    p.add_argument("--depth_max", type=float, default=None,
                   help="PatchMatch max depth (default: 1.5× farthest cam-to-centroid dist)")
    args = p.parse_args()

    data_root = Path(args.data_root)
    out = Path(args.output) if args.output else (
        Path(f"{data_root}_output") / "stage_a" / "colmap_4d")
    out.mkdir(parents=True, exist_ok=True)

    frames = _parse_frames(args.frames)
    if not frames:
        raise SystemExit("[stage_a_colmap_4d] empty --frames")
    ref_frame = args.ref_frame if args.ref_frame is not None else frames[0]
    print(f"[stage_a_colmap_4d] {len(frames)} frames: {frames[:5]}{'...' if len(frames) > 5 else ''}")
    print(f"[stage_a_colmap_4d] ref_frame={ref_frame}, output={out}")

    cam_names = sorted(d.name for d in (data_root / "images").iterdir() if d.is_dir())
    print(f"[stage_a_colmap_4d] cams: {cam_names}")

    # ---- Phase 1: shared sparse model (one-time) ----
    shared_dir = out / "shared"
    if not args.skip_setup:
        _setup_shared(data_root, ref_frame, shared_dir,
                      ext=args.ext, intri=args.intri, extri=args.extri,
                      colmap=args.colmap, use_gpu=not args.no_gpu)
    elif not (shared_dir / "sparse" / "0" / "cameras.bin").exists():
        raise SystemExit(f"[stage_a_colmap_4d] --skip_setup but no shared model at {shared_dir}")

    # ---- Depth bounds for PatchMatch (no sparse points → must specify) ----
    if args.depth_min is None or args.depth_max is None:
        dmin, dmax = _auto_depth_bounds(data_root, args.intri, args.extri)
        depth_min = args.depth_min if args.depth_min is not None else dmin
        depth_max = args.depth_max if args.depth_max is not None else dmax
    else:
        depth_min, depth_max = args.depth_min, args.depth_max
    print(f"[stage_a_colmap_4d] depth bounds: min={depth_min:.3f} max={depth_max:.3f}")

    # ---- Phase 2: per-frame dense ----
    for i, f in enumerate(frames, start=1):
        frame_dir = out / f"frame_{f:06d}"
        fused = frame_dir / "dense" / "fused.ply"
        if fused.exists():
            print(f"[stage_a_colmap_4d] frame {f}: fused.ply exists, skipping ({i}/{len(frames)})")
            continue
        print(f"\n{'='*60}\n[stage_a_colmap_4d] frame {f} ({i}/{len(frames)})\n{'='*60}")
        _link_frame_workspace(data_root, f, frame_dir, shared_dir, cam_names, args.ext)
        _run_dense(frame_dir, args.neighbor, args.gpu_index, args.colmap,
                   depth_min, depth_max)
        print(f"[stage_a_colmap_4d] frame {f}: {fused}")

    print(f"\n[stage_a_colmap_4d] done; per-frame clouds at {out}/frame_*/dense/fused.ply")


if __name__ == "__main__":
    main()
