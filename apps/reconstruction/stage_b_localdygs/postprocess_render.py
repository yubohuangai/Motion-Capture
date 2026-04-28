"""Post-process LocalDyGS render output for human-friendly review.

For a model dir that has `<split>/ours_<iter>/{renders,gt}/<00000>.png`:

1. Replace gt/*.png with gt/*.jpg (quality 85). Saves ~10x storage.
   The originals live at <data>/images/, so no information loss.
2. Add `renamed/` subdir with symlinks like `cam<NN>_sceneframe_<NNN>.png`
   so it's obvious which render corresponds to which camera+time.

Usage:
    python -m apps.reconstruction.stage_b_localdygs.postprocess_render \\
        /scratch/.../stage_b/train_planB_e3/ \\
        --n-frames 60 \\
        --train-cams 02,03,04,05,06,07,08,09,10 \\
        --test-cams 01,11 \\
        [--frame-stride 5]   # optional: prepend raw frame number to symlink

Index → (cam, scene_frame) mapping (matches LocalDyGS's load_images_path):
The renders are sorted by sort_by_image_name (last-2-digits int), so
all frames for cam02 (sort key 2) come first, then cam03, etc.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from PIL import Image


def _split_cams(spec: str) -> list[str]:
    return [c.strip() for c in spec.split(",") if c.strip()]


def _process_split(split_dir: Path, cams: list[str], n_frames: int,
                   frame_stride: int | None, jpeg_quality: int,
                   keep_png_gt: bool) -> None:
    renders_dir = split_dir / "renders"
    gt_dir = split_dir / "gt"
    renamed_dir = split_dir / "renamed"
    renamed_dir.mkdir(exist_ok=True)

    expected_n = len(cams) * n_frames
    actual_n = len(sorted(renders_dir.glob("*.png")))
    if actual_n != expected_n:
        print(f"[postprocess]   WARN: {split_dir.name}: expected "
              f"{expected_n} renders ({len(cams)} cams × {n_frames} frames), "
              f"got {actual_n}. Proceeding anyway.")

    n_compressed = 0
    n_linked = 0
    for cam_pos, cam in enumerate(cams):
        for frame_idx in range(n_frames):
            idx = cam_pos * n_frames + frame_idx
            png_name = f"{idx:05d}.png"
            render_png = renders_dir / png_name
            gt_png = gt_dir / png_name
            if not render_png.exists():
                continue

            # Compose human-friendly name
            if frame_stride is not None:
                raw_frame = frame_idx * frame_stride
                stem = f"cam{cam}_frame_{raw_frame:06d}"
            else:
                stem = f"cam{cam}_sceneframe_{frame_idx:03d}"

            # Symlink render
            link_render = renamed_dir / f"{stem}.png"
            if link_render.exists() or link_render.is_symlink():
                link_render.unlink()
            os.symlink(render_png.resolve(), link_render)
            n_linked += 1

            # Compress GT to JPEG, replace PNG
            if gt_png.exists():
                gt_jpg = gt_dir / f"{idx:05d}.jpg"
                if not gt_jpg.exists():
                    Image.open(gt_png).convert("RGB").save(
                        gt_jpg, "JPEG", quality=jpeg_quality, optimize=True
                    )
                    n_compressed += 1
                if not keep_png_gt:
                    gt_png.unlink()
                # Symlink the JPEG too with the friendly name
                link_gt = renamed_dir / f"{stem}_gt.jpg"
                if link_gt.exists() or link_gt.is_symlink():
                    link_gt.unlink()
                os.symlink(gt_jpg.resolve(), link_gt)

    print(f"[postprocess]   {split_dir.parent.name}/{split_dir.name}: "
          f"{n_linked} renders linked, {n_compressed} GTs compressed → JPEG")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir", type=str, help="LocalDyGS model dir "
                   "(contains train/test/ours_<iter>/{renders,gt})")
    p.add_argument("--iter", type=int, default=30000)
    p.add_argument("--n-frames", type=int, required=True,
                   help="number of time frames in the scene (e.g. 60 or 136)")
    p.add_argument("--train-cams", type=str, required=True,
                   help="comma-separated cam labels in sort order (e.g. 02,03,...,10)")
    p.add_argument("--test-cams", type=str, required=True,
                   help="comma-separated test cam labels (e.g. 01,11)")
    p.add_argument("--frame-stride", type=int, default=None,
                   help="if given, raw frame numbers in symlink names use "
                        "scene_frame_idx * stride (e.g. stride=5 for our "
                        "stride-5-over-0..295 setup)")
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument("--keep-png-gt", action="store_true",
                   help="keep gt/*.png (default: delete after compressing to .jpg)")
    args = p.parse_args()

    model = Path(args.model_dir).resolve()
    train_cams = _split_cams(args.train_cams)
    test_cams = _split_cams(args.test_cams)

    print(f"[postprocess] model: {model}")
    print(f"[postprocess] iter:  {args.iter}")
    print(f"[postprocess] train cams ({len(train_cams)}): {train_cams}")
    print(f"[postprocess] test  cams ({len(test_cams)}): {test_cams}")

    for split, cams in [("train", train_cams), ("test", test_cams)]:
        d = model / split / f"ours_{args.iter}"
        if not d.is_dir():
            print(f"[postprocess]   skipping {split}: {d} doesn't exist")
            continue
        _process_split(d, cams, args.n_frames, args.frame_stride,
                       args.jpeg_quality, args.keep_png_gt)

    # Print final dir size
    total_bytes = sum(p.stat().st_size for p in model.rglob("*") if p.is_file())
    print(f"[postprocess] done. final size: {total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
