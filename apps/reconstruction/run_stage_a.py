"""Stage A dispatcher: select a dense MVS backend and forward args to it.

Both backends produce a dense point cloud + sparse seed cloud from the
same posed multi-view RGB inputs; they just use different dense matching
algorithms. **COLMAP (PatchMatch stereo) is the default** — visually
cleaner geometry on the reference cow_1/10465 dataset. Plane-sweep is
available as an alternate that needs no GPU and no COLMAP binary.

Usage
-----
    # default — COLMAP MVS
    python -m apps.reconstruction.run_stage_a <data_root>

    # explicit plane-sweep, with backend-specific knobs
    python -m apps.reconstruction.run_stage_a <data_root> \
        --backend plane_sweep --rel_tol 0.01 --min_consistent 3

Outputs land at ``<data_root>_output/stage_a/{colmap,plane_sweep}/``
depending on the backend.

Pass ``--help`` to see the dispatcher flags. Backend-specific flags pass
through unchanged — invoke the underlying driver's ``--help`` to discover
them:

    python -m apps.reconstruction.stage_a_colmap.run_stage_a_colmap --help
    python -m apps.reconstruction.stage_a_plane_sweep.run_stage_a_plane_sweep --help
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys


_BACKENDS = {
    "colmap": "apps.reconstruction.stage_a_colmap.run_stage_a_colmap",
    "plane_sweep": "apps.reconstruction.stage_a_plane_sweep.run_stage_a_plane_sweep",
}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stage A dispatcher (COLMAP MVS or plane-sweep)",
        # Pass-through unknown args to the chosen backend driver
    )
    p.add_argument("data_root", type=str)
    p.add_argument("--backend", choices=list(_BACKENDS), default="colmap",
                   help="Dense MVS backend (default: colmap)")
    args, extra = p.parse_known_args()

    cmd = [sys.executable, "-m", _BACKENDS[args.backend], args.data_root, *extra]
    print(f"[stage_a] backend={args.backend}; "
          f"$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
