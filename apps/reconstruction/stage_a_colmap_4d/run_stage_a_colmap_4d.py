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

    Skips SIFT triangulation entirely (calibration is already known) — output
    is just ``cameras.bin`` + ``images.bin``. The ``images/`` subdir is
    populated from ``ref_frame`` (raw distorted) but never used at dense time.
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
        "--no-triangulate",
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


def _run_dense(frame_dir: Path, neighbor: int, gpu_index: str, colmap: str) -> None:
    cmd = [
        sys.executable, "-m",
        "apps.reconstruction.stage_a_colmap.dense_reconstruct",
        str(frame_dir),
        "--neighbor", str(neighbor),
        "--gpu_index", gpu_index,
        "--no-mask",
        "--colmap", colmap,
    ]
    _run(cmd)


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

    # ---- Phase 2: per-frame dense ----
    for i, f in enumerate(frames, start=1):
        frame_dir = out / f"frame_{f:06d}"
        fused = frame_dir / "dense" / "fused.ply"
        if fused.exists():
            print(f"[stage_a_colmap_4d] frame {f}: fused.ply exists, skipping ({i}/{len(frames)})")
            continue
        print(f"\n{'='*60}\n[stage_a_colmap_4d] frame {f} ({i}/{len(frames)})\n{'='*60}")
        _link_frame_workspace(data_root, f, frame_dir, shared_dir, cam_names, args.ext)
        _run_dense(frame_dir, args.neighbor, args.gpu_index, args.colmap)
        print(f"[stage_a_colmap_4d] frame {f}: {fused}")

    print(f"\n[stage_a_colmap_4d] done; per-frame clouds at {out}/frame_*/dense/fused.ply")


if __name__ == "__main__":
    main()
