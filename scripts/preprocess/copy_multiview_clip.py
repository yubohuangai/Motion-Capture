#!/usr/bin/env python3
"""
Copy a contiguous **multi-view clip** (frame range) from every ``NN/images/`` folder under a
session ``raw/`` root into a new dataset root with renumbered frames.

Frames are ordered by **natural sort** on filenames (same idea as ``multiview_grid_video.py``).
Output layout::

    <output_root>/images/01/000000.jpg
    <output_root>/images/02/000000.jpg
    …

Renumbered from ``000000`` so each clip starts at zero.

Example (Alliance)::

    python scripts/preprocess/copy_multiview_clip.py \\
      --input ~/scratch/cow_2_board/raw \\
      --output ~/scratch/cow_2_board/board \\
      --frame-start 1525 --frame-end 3361
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm

ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg")

# Legacy inline config (only used with --legacy-config)
CONFIG = {
    "base_root": "/mnt/yubo/obj/raw",
    "tasks": [
        {
            "name": "jug",
            "dst": "/mnt/yubo/obj/jug",
            "start": "/mnt/yubo/obj/raw/01/images/1774222649651067452.jpg",
            "end": "/mnt/yubo/obj/raw/01/images/1774222649651067452.jpg",
        },
    ],
}


def natural_key_filename(filename: str):
    stem = os.path.splitext(filename)[0]
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", stem)]


def get_camera_ids(root):
    root_path = Path(root)
    if not root_path.exists():
        return []
    cam_ids = []
    for entry in root_path.iterdir():
        if entry.is_dir() and entry.name.isdigit():
            cam_ids.append(int(entry.name))
    return sorted(cam_ids)


def list_images(img_dir):
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(ALLOWED_EXTENSIONS)]
    return sorted(files, key=natural_key_filename)


def index_from_filename(img_dir, filename):
    files = list_images(img_dir)
    if filename not in files:
        raise FileNotFoundError(f"{filename} not found in {img_dir}")
    return files.index(filename) + 1  # 1-based


def _copy_file(src_path: str, dst_path: str, mode: str) -> None:
    if mode == "copy2":
        shutil.copy2(src_path, dst_path)
    else:
        shutil.copyfile(src_path, dst_path)


def copy_range(
    src_dir,
    dst_dir,
    start_1based: int,
    end_slice_exclusive: int,
    *,
    copy_mode: str = "copyfile",
    show_progress: bool = True,
):
    """
    Copy ``files[start_1based - 1 : end_slice_exclusive]`` (Python slice).

    For inclusive 1-based frames **A … B**, pass ``start_1based=A`` and
    ``end_slice_exclusive=B`` (second slice index is exclusive, so last kept index is B-1 =
    the B-th frame). Example: 1525..3361 → ``files[1524:3361]``.
    """
    os.makedirs(dst_dir, exist_ok=True)
    files = list_images(src_dir)
    n = len(files)
    if start_1based < 1 or start_1based > n:
        raise ValueError(f"start frame {start_1based} out of range (1..{n}) for {src_dir}")
    if end_slice_exclusive < start_1based or end_slice_exclusive > n:
        raise ValueError(
            f"end slice {end_slice_exclusive} invalid with start {start_1based} (need {start_1based}..{n})"
        )
    selected = files[start_1based - 1 : end_slice_exclusive]
    iterator = selected
    if show_progress:
        iterator = tqdm(selected, desc=f"Copying {os.path.basename(src_dir)}")
    next_idx = 0
    for f in iterator:
        ext = os.path.splitext(f)[1]
        new_name = f"{next_idx:06d}{ext}"
        _copy_file(os.path.join(src_dir, f), os.path.join(dst_dir, new_name), copy_mode)
        next_idx += 1


def copy_first(src_dir, dst_dir):
    files = list_images(src_dir)
    if not files:
        raise FileNotFoundError(f"No images found in {src_dir}")
    os.makedirs(dst_dir, exist_ok=True)
    first = files[0]
    ext = os.path.splitext(first)[1]
    shutil.copy2(os.path.join(src_dir, first), os.path.join(dst_dir, f"{0:06d}{ext}"))


def run_task(base_root, task):
    cam_ids = get_camera_ids(base_root)
    if not cam_ids:
        raise RuntimeError(f"No camera folders found under {base_root}")

    cam01_dir = Path(base_root) / "01" / "images"
    if not cam01_dir.exists():
        raise FileNotFoundError(f"Camera 01 images not found: {cam01_dir}")

    if task.get("mode") == "first":
        for cam_id in cam_ids:
            src_dir = Path(base_root) / f"{cam_id:02d}" / "images"
            dst_dir = Path(task["dst"]) / "images" / f"{cam_id:02d}"
            copy_first(str(src_dir), str(dst_dir))
        return

    start_name = Path(task["start"]).name
    end_name = Path(task["end"]).name
    start_idx = index_from_filename(str(cam01_dir), start_name)
    end_idx = index_from_filename(str(cam01_dir), end_name)
    if end_idx < start_idx:
        raise ValueError(f"End index before start index for task {task['name']}")

    for cam_id in cam_ids:
        src_dir = Path(base_root) / f"{cam_id:02d}" / "images"
        dst_dir = Path(task["dst"]) / "images" / f"{cam_id:02d}"
        copy_range(str(src_dir), str(dst_dir), start_idx, end_idx)


def run_cli_clip(
    base_root: str,
    output_root: str,
    frame_start: int,
    frame_end: int,
    *,
    camera_workers: int = 1,
    copy_mode: str = "copyfile",
    progress: str = "camera",
) -> None:
    """
    Copy inclusive 1-based frame indices ``frame_start``..``frame_end`` for every camera.

    Slice: files[frame_start - 1 : frame_end] — so ``frame_end`` is **inclusive** last frame number.
    """
    base_root = str(Path(base_root).expanduser().resolve())
    output_root = str(Path(output_root).expanduser().resolve())

    if frame_start < 1:
        raise SystemExit("--frame-start must be >= 1")
    if frame_end < frame_start:
        raise SystemExit("--frame-end must be >= --frame-start")

    cam_ids = get_camera_ids(base_root)
    if not cam_ids:
        raise SystemExit(f"No camera folders found under {base_root}")

    cam01 = Path(base_root) / "01" / "images"
    if not cam01.is_dir():
        raise SystemExit(f"Missing {cam01}")

    cam01_files = list_images(str(cam01))
    n_ref = len(cam01_files)
    if frame_start > n_ref:
        raise SystemExit(f"--frame-start {frame_start} beyond camera 01 length ({n_ref} frames)")
    if frame_end > n_ref:
        raise SystemExit(f"--frame-end {frame_end} beyond camera 01 length ({n_ref} frames)")

    # Inclusive frames A..B → slice files[A-1:B] (B is Python exclusive end index).
    end_slice = frame_end

    print(
        f"[copy_multiview_clip] Input: {base_root}\n"
        f"[copy_multiview_clip] Output: {output_root}/images/<cam>/\n"
        f"[copy_multiview_clip] Frames (1-based, inclusive): {frame_start}..{frame_end} "
        f"({frame_end - frame_start + 1} per camera); ref count cam01={n_ref}"
    )

    tasks: list[tuple[str, str, int, int]] = []
    for cam_id in cam_ids:
        src_dir = Path(base_root) / f"{cam_id:02d}" / "images"
        dst_dir = Path(output_root) / "images" / f"{cam_id:02d}"
        if not src_dir.is_dir():
            raise SystemExit(f"Missing {src_dir}")
        nf = len(list_images(str(src_dir)))
        if frame_end > nf:
            raise SystemExit(
                f"Camera {cam_id:02d} has only {nf} images; --frame-end {frame_end} is too large."
            )
        tasks.append((str(src_dir), str(dst_dir), frame_start, end_slice))

    if camera_workers < 1:
        raise SystemExit("--camera-workers must be >= 1")
    camera_workers = min(camera_workers, len(tasks))
    show_per_camera = progress == "camera"
    show_summary = progress in ("camera", "overall")

    if show_summary:
        print(
            f"[copy_multiview_clip] copy_mode={copy_mode}  camera_workers={camera_workers}  "
            f"progress={progress}"
        )

    if camera_workers == 1:
        for src_dir, dst_dir, s, e in tasks:
            copy_range(
                src_dir, dst_dir, s, e, copy_mode=copy_mode, show_progress=show_per_camera
            )
        return

    with ThreadPoolExecutor(max_workers=camera_workers) as ex:
        futures = [
            ex.submit(
                copy_range,
                src_dir,
                dst_dir,
                s,
                e,
                copy_mode=copy_mode,
                show_progress=False,
            )
            for (src_dir, dst_dir, s, e) in tasks
        ]
        if show_summary:
            for _ in tqdm(futures, desc="Copying cameras", unit="cam"):
                _.result()
        else:
            for f in futures:
                f.result()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Copy a multi-view clip: contiguous frame range from raw/NN/images/ into a new root "
            "(renumbered 000000…)."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        metavar="RAW_ROOT",
        help="Session root with 01/images, 02/images, … (e.g. ~/scratch/cow_2_board/raw)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        metavar="OUT_ROOT",
        help="Output root; writes OUT_ROOT/images/NN/000000.ext (e.g. ~/scratch/cow_2_board/board)",
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        default=None,
        metavar="N",
        help="First frame index (1-based, inclusive), same ordering as sorted filenames in 01/images.",
    )
    parser.add_argument(
        "--frame-end",
        type=int,
        default=None,
        metavar="M",
        help="Last frame index (1-based, inclusive).",
    )
    parser.add_argument(
        "--camera-workers",
        type=int,
        default=1,
        help=(
            "Number of cameras copied in parallel. 1 = sequential. "
            "Try 4-8 on shared storage (default: 1)."
        ),
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copyfile", "copy2"),
        default="copyfile",
        help=(
            "copyfile: faster data-only copy (default). "
            "copy2: preserve metadata but usually slower."
        ),
    )
    parser.add_argument(
        "--progress",
        choices=("camera", "overall", "none"),
        default="camera",
        help="camera: per-camera bars, overall: one bar in parallel mode, none: silent.",
    )
    parser.add_argument(
        "--legacy-config",
        action="store_true",
        help="Run legacy CONFIG tasks below (default: off).",
    )
    parser.add_argument(
        "base_root_positional",
        nargs="?",
        default=None,
        help="(Legacy) base root if not using --input",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.legacy_config:
        base = args.base_root_positional or CONFIG["base_root"]
        for task in CONFIG["tasks"]:
            print(f"=== Task: {task['name']} → {task['dst']} ===")
            run_task(base, task)
    elif args.input and args.output and args.frame_start is not None and args.frame_end is not None:
        run_cli_clip(
            args.input,
            args.output,
            args.frame_start,
            args.frame_end,
            camera_workers=args.camera_workers,
            copy_mode=args.copy_mode,
            progress=args.progress,
        )
    else:
        raise SystemExit(
            "Usage:\n"
            "  python scripts/preprocess/copy_multiview_clip.py \\\n"
            "    --input  <raw_root> \\\n"
            "    --output <out_root> \\\n"
            "    --frame-start <N> --frame-end <M>\n"
            "\n"
            "Example:\n"
            "  python scripts/preprocess/copy_multiview_clip.py -i ~/scratch/cow_2_board/raw "
            "-o ~/scratch/cow_2_board/board --frame-start 1525 --frame-end 3361\n"
        )
