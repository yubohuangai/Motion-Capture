"""End-to-end driver: Stage A (classical MVS) -> Stage B (scene-specific NeuS).

Wraps the two stage drivers as subprocesses so each keeps its own CLI
semantics. Use this when you want a single command from images + calibration
to a final NeuS mesh, without caring about the intermediate knobs.

Usage
-----
    python -m apps.reconstruction.run_pipeline <data_root> \
        --neus_iters 100000 --device cuda

By default outputs go to ``<data_root>_output`` (Stage A) and
``<data_root>_output/neus`` (Stage B). Pass ``--output`` to override.

Pass ``--stage_a_args '...extra flags...'`` / ``--stage_b_args '...'`` for
tuning. Stage A is skipped if ``<output>/sparse.ply`` already exists (unless
``--rerun_stage_a``).
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


def main() -> None:
    p = argparse.ArgumentParser(description="Classical MVS + NeuS pipeline")
    p.add_argument("data_root", type=str)
    p.add_argument("--output", type=str, default=None,
                   help="root output dir (default: <data_root>_output); "
                        "Stage A goes in <output>/, Stage B in <output>/neus/")
    p.add_argument("--frame", type=int, default=0)
    # Stage A passthroughs (common knobs)
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
    # Escape hatch for extra args
    p.add_argument("--stage_a_args", type=str, default="",
                   help="extra CLI string forwarded to run_stage_a")
    p.add_argument("--stage_b_args", type=str, default="",
                   help="extra CLI string forwarded to run_stage_b")
    args = p.parse_args()

    output_root = Path(args.output) if args.output else Path(f"{str(args.data_root)}_output")
    out_a = output_root / "stage_a" / "plane_sweep"
    out_b = output_root / "stage_b" / "neus"

    # ----- Stage A ----------------------------------------------------------
    have_sparse = (out_a / "sparse.ply").exists()
    have_fused = (out_a / "fused.ply").exists()
    need_stage_a = not args.skip_stage_a and (args.rerun_stage_a or not (have_sparse and have_fused))
    if need_stage_a:
        cmd_a = [
            sys.executable, "-m", "apps.reconstruction.run_stage_a",
            str(args.data_root),
            "--output", str(out_a),
            "--frame", str(args.frame),
            "--n_depths", str(args.n_depths),
            "--max_sources", str(args.max_sources),
        ]
        if args.stage_a_args:
            cmd_a.extend(shlex.split(args.stage_a_args))
        _run(cmd_a)
    else:
        print(f"[pipeline] Stage A outputs present in {out_a}; skipping")

    # ----- Stage B ----------------------------------------------------------
    if not args.skip_stage_b:
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
        if args.stage_b_args:
            cmd_b.extend(shlex.split(args.stage_b_args))
        _run(cmd_b)
    print(f"[pipeline] done; see {out_a}")


if __name__ == "__main__":
    main()
