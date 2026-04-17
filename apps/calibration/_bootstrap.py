"""Calibration script bootstrap helpers."""
from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_on_path() -> Path:
    """
    Add repository root to sys.path so local imports like ``easymocap`` work
    when scripts are executed by file path, e.g.:

      python apps/calibration/detect_charuco.py ...
    """
    calib_dir = Path(__file__).resolve().parent
    repo_root = calib_dir.parent.parent
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo_root

