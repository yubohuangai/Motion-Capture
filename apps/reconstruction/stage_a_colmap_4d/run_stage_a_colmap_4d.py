"""Stage A 4D extension: per-frame COLMAP MVS for a moving subject in a fixed multi-view rig.

Why this exists
---------------
The single-frame ``stage_a_colmap`` reconstructs only what the cameras see at
one instant. A half-circle rig misses the back of a static subject. But if
the subject *moves* through the rig (e.g. a cow walks 360° over a few
seconds), the union of per-frame reconstructions covers more of the surface
— cameras stayed put, the cow rotated.

Implementation
--------------
Loops over selected frames and runs the single-frame
``stage_a_colmap.run_stage_a_colmap`` on each. Every per-frame run produces
its own sparse model + dense MVS, which means:

- COLMAP auto-derives **per-camera** depth bounds from each frame's
  triangulated SIFT points (essential — a single global depth bound across
  cameras leaves close-near-plane cams with wrong hypothesis ranges, and
  every depth gets filtered by the geometric-consistency check).
- Per-frame work includes ~5 min of SIFT/matching/triangulation. We *could*
  share that across frames since cameras don't move, but COLMAP's
  ``triangulate_points`` rewrite step hardcodes PINHOLE cameras (it's
  designed around ``--undistort=True``), and our prior attempt to skip
  triangulation entirely failed precisely because of the per-camera depth
  bounds requirement. Per-frame full pipeline is the robust path.

Output is one ``fused.ply`` per timestamp, all in the same world coords
(so they overlay directly in MeshLab / Open3D).

Usage
-----
    python -m apps.reconstruction.stage_a_colmap_4d.run_stage_a_colmap_4d \
        /scratch/yubo/cow_1/9148_10581 --frames 0:150:30
    # 5 timestamps spaced 1 sec apart at 30 fps

    # Explicit list
    python -m apps.reconstruction.stage_a_colmap_4d.run_stage_a_colmap_4d \
        /scratch/yubo/cow_1/9148_10581 --frames 0,30,60,90,120

Output (default ``<data_root>_output/stage_a/colmap_4d/``)::

    frame_000000/
      images/, masks/, sparse/0/, dense/fused.ply, ...
    frame_000030/
      ...

To inspect the timelapse: open every ``frame_*/dense/fused.ply`` together
in MeshLab; each frame shows the cow at that timestamp's position.

Timing on cow_1 (11 cams @ 4K, full A100): ~22 min/frame end-to-end.
"""

from __future__ import annotations

import argparse
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


def _run_single_frame(data_root: Path, frame: int, frame_dir: Path,
                      neighbor: int, gpu_index: str, colmap: str) -> None:
    """Run the single-frame stage_a_colmap driver against one timestamp."""
    cmd = [
        sys.executable, "-m",
        "apps.reconstruction.stage_a_colmap.run_stage_a_colmap",
        str(data_root),
        "--output", str(frame_dir),
        "--frame", str(frame),
        "--neighbor", str(neighbor),
        "--gpu_index", gpu_index,
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
    p.add_argument("--neighbor", "-k", type=int, default=6,
                   help="K nearest cameras for PatchMatch sources")
    p.add_argument("--colmap", default="colmap")
    p.add_argument("--gpu_index", default="0",
                   help="GPU index for patch_match_stereo (default: 0)")
    args = p.parse_args()

    data_root = Path(args.data_root)
    out = Path(args.output) if args.output else (
        Path(f"{data_root}_output") / "stage_a" / "colmap_4d")
    out.mkdir(parents=True, exist_ok=True)

    frames = _parse_frames(args.frames)
    if not frames:
        raise SystemExit("[stage_a_colmap_4d] empty --frames")
    print(f"[stage_a_colmap_4d] {len(frames)} frames: "
          f"{frames[:5]}{'...' if len(frames) > 5 else ''}; output={out}")

    for i, f in enumerate(frames, start=1):
        frame_dir = out / f"frame_{f:06d}"
        fused = frame_dir / "dense" / "fused.ply"
        if fused.exists() and fused.stat().st_size > 1024:
            print(f"[stage_a_colmap_4d] frame {f}: fused.ply present, skipping ({i}/{len(frames)})")
            continue
        print(f"\n{'='*60}\n[stage_a_colmap_4d] frame {f} ({i}/{len(frames)})\n{'='*60}")
        _run_single_frame(data_root, f, frame_dir, args.neighbor, args.gpu_index, args.colmap)
        print(f"[stage_a_colmap_4d] frame {f}: {fused}")

    print(f"\n[stage_a_colmap_4d] done; per-frame clouds at "
          f"{out}/frame_*/dense/fused.ply")


if __name__ == "__main__":
    main()
