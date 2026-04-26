"""Stage A (alternate backend): COLMAP-based MVS.

Wraps :mod:`export_colmap` and :mod:`dense_reconstruct` so a single command
takes posed images to a dense ``fused.ply`` produced by COLMAP's PatchMatch
stereo. Useful as a reference baseline against the hand-rolled plane-sweep
under ``stage_a_plane_sweep/``.

Defaults follow the project convention: output goes to
``<data_root>_output/colmap_ws/`` (sibling of the input dir, never inside
the codebase). The fused dense cloud lands at
``<data_root>_output/colmap_ws/dense/fused.ply``.

Usage
-----
    python -m apps.reconstruction.stage_a_colmap.run_stage_a_colmap \
        /path/to/data --frame 0 --neighbor 6

Requires the ``colmap`` binary on ``$PATH`` (``--colmap`` to override) and
GPU for ``patch_match_stereo``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("[stage_a_colmap] $", " ".join(shlex.quote(c) for c in cmd), flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    p = argparse.ArgumentParser(description="Stage A (COLMAP backend)")
    p.add_argument("data_root", type=str)
    p.add_argument("--output", type=str, default=None,
                   help="output dir (default: <data_root>_output/colmap_ws)")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--intri", default="intri.yml")
    p.add_argument("--extri", default="extri.yml")
    p.add_argument("--ext", default=".jpg")
    p.add_argument("--mask", default="masks",
                   help="mask subdir under data root (default: masks)")
    p.add_argument("--no-mask", action="store_true")
    p.add_argument("--neighbor", "-k", type=int, default=6,
                   help="K nearest cameras for matching + PatchMatch sources")
    p.add_argument("--colmap", default="colmap")
    p.add_argument("--no-gpu", action="store_true",
                   help="disable GPU for COLMAP feature extraction/matching")
    p.add_argument("--gpu_index", default="0",
                   help="GPU index for patch_match_stereo (default: 0)")
    p.add_argument("--skip_dense", action="store_true",
                   help="run sparse triangulation only; skip dense MVS")
    p.add_argument("--no-mask-sparse", dest="no_mask_sparse",
                   action="store_true",
                   help="do NOT apply masks during SIFT/triangulation, even "
                        "if --mask is set. Useful for masked dense + unmasked "
                        "sparse: SIFT then has the whole scene to match on, "
                        "preventing the depth-bound failure that happens when "
                        "a tight mask leaves <100 sparse points.")
    args = p.parse_args()

    data_root = Path(args.data_root)
    out = Path(args.output) if args.output else Path(f"{data_root}_output") / "stage_a" / "colmap"
    out.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).resolve().parent
    export = here / "export_colmap.py"
    dense = here / "dense_reconstruct.py"

    # ---- Sparse: export + COLMAP triangulation ---------------------------
    cmd = [
        sys.executable, str(export), str(data_root),
        "--output", str(out),
        "--frame", str(args.frame),
        "--intri", args.intri,
        "--extri", args.extri,
        "--ext", args.ext,
        "--triangulate",
    ]
    cmd.append("--undistort")
    # Sparse-step masks: optionally suppressed even when --mask is set, so
    # SIFT has the full scene and produces enough points for depth bounds.
    if args.no_mask or args.no_mask_sparse:
        cmd.append("--no-mask")
    else:
        cmd.extend(["--mask", args.mask])
    if args.no_gpu:
        cmd.append("--no-gpu")
    else:
        cmd.append("--gpu")
    cmd.extend(["--colmap", args.colmap])
    _run(cmd)

    if args.skip_dense:
        print(f"[stage_a_colmap] sparse-only run complete; output at {out}")
        return

    # ---- Dense: PatchMatch + fusion --------------------------------------
    cmd = [
        sys.executable, str(dense), str(out),
        "--neighbor", str(args.neighbor),
        "--colmap", args.colmap,
        "--gpu_index", args.gpu_index,
    ]
    if args.no_mask:
        cmd.append("--no-mask")
    _run(cmd)

    fused = out / "dense" / "fused.ply"
    print(f"[stage_a_colmap] done; dense cloud: {fused}")
    print(f"[stage_a_colmap] compare to plane-sweep output at "
          f"{Path(f'{data_root}_output') / 'stage_a' / 'plane_sweep' / 'fused.ply'}")


if __name__ == "__main__":
    main()
