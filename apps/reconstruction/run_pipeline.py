"""End-to-end driver: Stage A (MVS) -> Stage B (scene-specific neural).

Wraps the two stage drivers as subprocesses so each keeps its own CLI
semantics. Use this when you want a single command from images +
calibration to a final neural mesh, without caring about the intermediate
knobs.

Defaults: Stage A = COLMAP MVS, Stage B = NeuS. Override with
``--stage_a_backend {colmap,plane_sweep}`` and
``--stage_b_backend {neus,3dgs}``. Outputs land at
``<data_root>_output/{stage_a,stage_b}/<backend>/``.

Usage
-----
    python -m apps.reconstruction.run_pipeline <data_root> \
        --neus_iters 100000 --device cuda

    # Plane-sweep MVS instead of COLMAP, with extra knobs
    python -m apps.reconstruction.run_pipeline <data_root> \
        --stage_a_backend plane_sweep \
        --stage_a_args '--rel_tol 0.01 --min_consistent 3'

Stage A is skipped if the chosen backend's "done" marker file already
exists (unless ``--rerun_stage_a``).
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("[pipeline] $", " ".join(shlex.quote(c) for c in cmd), flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


# Per-backend output dir + done-marker (relative to <output_root>).
_STAGE_A = {
    "colmap": ("stage_a/colmap", "dense/fused.ply"),
    "plane_sweep": ("stage_a/plane_sweep", "fused.ply"),
}
_STAGE_B = {
    "neus": "stage_b/neus",
    "3dgs": "stage_b/3dgs",
}


def main() -> None:
    p = argparse.ArgumentParser(description="MVS + neural reconstruction pipeline")
    p.add_argument("data_root", type=str)
    p.add_argument("--output", type=str, default=None,
                   help="root output dir (default: <data_root>_output); "
                        "Stage A goes in <output>/stage_a/<backend>/, "
                        "Stage B in <output>/stage_b/<backend>/")
    p.add_argument("--frame", type=int, default=0)
    # Backend selection
    p.add_argument("--stage_a_backend", choices=list(_STAGE_A), default="colmap",
                   help="Stage A dense MVS backend (default: colmap)")
    p.add_argument("--stage_b_backend", choices=list(_STAGE_B), default="neus",
                   help="Stage B neural backend (default: neus)")
    # Stage A passthroughs (common knobs, plane-sweep-flavored)
    p.add_argument("--n_depths", type=int, default=128)
    p.add_argument("--max_sources", type=int, default=4)
    p.add_argument("--rerun_stage_a", action="store_true",
                   help="rerun Stage A even if outputs exist")
    p.add_argument("--skip_stage_a", action="store_true")
    # Stage B passthroughs
    p.add_argument("--neus_iters", type=int, default=100_000)
    p.add_argument("--batch_rays", type=int, default=512)
    p.add_argument("--mesh_resolution", type=int, default=256)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--skip_stage_b", action="store_true")
    # Escape hatches
    p.add_argument("--stage_a_args", type=str, default="",
                   help="extra CLI string forwarded to the Stage A driver")
    p.add_argument("--stage_b_args", type=str, default="",
                   help="extra CLI string forwarded to the Stage B driver")
    args = p.parse_args()

    output_root = Path(args.output) if args.output else Path(f"{str(args.data_root)}_output")
    a_subdir, a_done_marker = _STAGE_A[args.stage_a_backend]
    out_a = output_root / a_subdir
    out_b = output_root / _STAGE_B[args.stage_b_backend]

    # ----- Stage A ----------------------------------------------------------
    done = (out_a / a_done_marker).exists()
    need_stage_a = not args.skip_stage_a and (args.rerun_stage_a or not done)
    if need_stage_a:
        cmd_a = [
            sys.executable, "-m", "apps.reconstruction.run_stage_a",
            str(args.data_root),
            "--backend", args.stage_a_backend,
            "--output", str(out_a),
            "--frame", str(args.frame),
        ]
        # n_depths / max_sources are plane-sweep concepts; only pass if applicable
        if args.stage_a_backend == "plane_sweep":
            cmd_a += ["--n_depths", str(args.n_depths),
                      "--max_sources", str(args.max_sources)]
        if args.stage_a_args:
            cmd_a.extend(shlex.split(args.stage_a_args))
        _run(cmd_a)
    else:
        print(f"[pipeline] Stage A done marker found at {out_a / a_done_marker}; skipping")

    # ----- Stage B ----------------------------------------------------------
    if not args.skip_stage_b:
        if args.stage_b_backend == "neus":
            cmd_b = [
                sys.executable, "-m", "apps.reconstruction.run_stage_b",
                str(args.data_root),
                "--stage_a_output", str(out_a),
                "--output", str(out_b),
                "--frame", str(args.frame),
                "--n_iters", str(args.neus_iters),
                "--batch_rays", str(args.batch_rays),
                "--mesh_resolution", str(args.mesh_resolution),
                "--device", args.device,
            ]
        else:  # 3dgs
            # 3DGS reads a COLMAP workspace; if Stage A wasn't COLMAP, error early.
            if args.stage_a_backend != "colmap":
                raise SystemExit(
                    "[pipeline] --stage_b_backend 3dgs needs Stage A = colmap "
                    "(it consumes a COLMAP workspace). "
                    "Set --stage_a_backend colmap.")
            cmd_b = [
                sys.executable, "-m", "apps.reconstruction.stage_b_3dgs.run_3dgs",
                str(args.data_root),
                "--output", str(output_root),
                "--frame", str(args.frame),
                "--skip_export",  # workspace already built by Stage A
            ]
        if args.stage_b_args:
            cmd_b.extend(shlex.split(args.stage_b_args))
        _run(cmd_b)
    print(f"[pipeline] done; stage A at {out_a}, stage B at {out_b}")


if __name__ == "__main__":
    main()
