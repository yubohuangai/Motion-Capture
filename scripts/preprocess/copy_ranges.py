#!/usr/bin/env python3
"""Deprecated: use ``copy_multiview_clip.py`` (same CLI)."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    print(
        "copy_ranges.py was renamed to copy_multiview_clip.py; forwarding…",
        file=sys.stderr,
    )
    runpy.run_path(
        str(Path(__file__).resolve().parent / "copy_multiview_clip.py"),
        run_name="__main__",
    )
