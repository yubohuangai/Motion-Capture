"""
Flatten foreground images produced by `apps/reconstruction/generate_masks.py`.

Input layout:
  <root>/<cam>/<frame>.jpg

Output layout (flat):
  <root>/<cam>_<frame>.jpg

Example:
  /Users/yubo/data/obj/cube/foreground_images/01/000000.jpg
    -> /Users/yubo/data/obj/cube/foreground_images/01_000000.jpg
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def gather_images(root: Path, mode: str, exts: tuple[str, ...], overwrite: bool, dry_run: bool) -> int:
    """
    Move/copy all files matching *.<ext> found at:
      root/<cam>/<frame>.<ext>
    into:
      root/<cam>_<frame>.<ext>
    """
    assert mode in {"move", "copy"}

    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root must be a directory: {root}")

    moved = 0
    # Only consider one level of subdir for cam, to avoid flattening nested postprocess trees.
    for cam_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        cam = cam_dir.name
        for p in sorted(cam_dir.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
                continue

            frame_stem = p.stem  # e.g. "000000"
            dst = root / f"{cam}_{frame_stem}{p.suffix.lower()}"

            if dst.exists():
                if not overwrite:
                    print(f"[skip] exists: {dst}")
                    continue
                if not dry_run:
                    dst.unlink()

            if dry_run:
                print(f"[dry-run] {p} -> {dst}")
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if mode == "move":
                    shutil.move(str(p), str(dst))
                else:
                    shutil.copy2(str(p), str(dst))
            moved += 1

    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatten foreground images into a single directory.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder containing <cam>/<frame>.<ext> (e.g. foreground_images).",
    )
    parser.add_argument(
        "--mode",
        choices=["move", "copy"],
        default="move",
        help="Whether to move or copy files (default: move).",
    )
    parser.add_argument(
        "--ext",
        default=".jpg",
        help="Comma-separated extensions to include (default: .jpg). Example: '.jpg,.png'",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination files if they exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print operations without changing files.")
    args = parser.parse_args()

    exts = tuple([e.strip().lower() for e in args.ext.split(",") if e.strip()])
    root = Path(args.root)

    moved = gather_images(root, mode=args.mode, exts=exts, overwrite=args.overwrite, dry_run=args.dry_run)
    print(f"[done] {moved} file(s) processed. mode={args.mode} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()

