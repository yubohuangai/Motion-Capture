#!/usr/bin/env python3
"""
Backward-compatible shim for detect_calibration_board.py --mode chessboard.

Prefer: python apps/calibration/detect_calibration_board.py --mode chessboard ...
"""
from __future__ import annotations

import sys
from pathlib import Path

_CALIB = Path(__file__).resolve().parent
if str(_CALIB) not in sys.path:
    sys.path.insert(0, str(_CALIB))

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv.insert(1, "--mode")
        sys.argv.insert(2, "chessboard")
    from detect_calibration_board import main

    main()
