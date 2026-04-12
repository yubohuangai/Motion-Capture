#!/usr/bin/env python3
"""
Remove per-camera extracted frame folders after a partial or failed sync.py --extract.

Deletes only: <data_root>/<NN>/images/  (NN = 01, 02, …)
Does not touch VID/*.mp4 or *.csv files.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_PREP_DIR = Path(__file__).resolve().parent
_SYNCTEST_DIR = _PREP_DIR / "synctest"
if str(_SYNCTEST_DIR) not in sys.path:
    sys.path.insert(0, str(_SYNCTEST_DIR))
import session_paths  # noqa: E402

DEFAULT_SYNCTEST_CONFIG = _SYNCTEST_DIR / "config.yaml"


def _camera_dirs(data_root: Path) -> list[Path]:
    if not data_root.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(data_root.iterdir()):
        if child.is_dir() and child.name.isdigit():
            out.append(child)
    return out


def _images_dirs(data_root: Path) -> list[Path]:
    found: list[Path] = []
    for cam in _camera_dirs(data_root):
        img = cam / "images"
        if img.is_dir():
            found.append(img)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Delete <cam>/images/ under a session raw root (e.g. incomplete sync --extract). "
            "Optional session root matches sync.py / analyze_vid.py."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        metavar="ROOT",
        help=(
            "Session raw directory (…/raw with 01/, 02/, …). "
            "If omitted, uses data_root from --synctest-config."
        ),
    )
    parser.add_argument(
        "--synctest-config",
        type=Path,
        default=None,
        metavar="YAML",
        help=(
            "YAML containing data_root when ROOT is omitted. "
            f"Default: {DEFAULT_SYNCTEST_CONFIG}"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List directories that would be removed; delete nothing.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    args = parser.parse_args()

    if args.root is None:
        cfg_path = args.synctest_config or DEFAULT_SYNCTEST_CONFIG
        data_root = session_paths.read_data_root_field(cfg_path)
    else:
        data_root = Path(args.root).expanduser().resolve()

    targets = _images_dirs(data_root)
    if not targets:
        print(f"No .../<NN>/images/ directories under:\n  {data_root}")
        return 0

    print(f"Session root:\n  {data_root}\nWill remove {len(targets)} folder(s):")
    for p in targets:
        print(f"  {p}")

    if args.dry_run:
        print("[dry-run] Nothing deleted.")
        return 0

    if not args.yes:
        try:
            reply = input(f"\nDelete these {len(targets)} images/ folder(s)? [y/N]: ").strip().lower()
        except EOFError:
            print("\nAborted (no TTY). Use --yes to delete without prompting.")
            return 1
        if reply not in ("y", "yes"):
            print("Aborted.")
            return 1

    for p in targets:
        shutil.rmtree(p)
        print(f"Removed {p}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
