"""Convert Stage 1 (per-frame COLMAP MVS) outputs to LocalDyGS scene layout.

LocalDyGS's ``readColmapSceneInfo`` expects the 3DGStream / VRU-style multi-view
dynamic-scene layout::

    <scene>/
    ├── sparse/0/
    │   ├── cameras.txt           # PINHOLE intrinsics, IDs 0..N-1
    │   └── images.txt            # extrinsics,           IDs 0..N-1
    ├── frame000000/images/{01..NN}.jpg     # undistorted, per-camera
    ├── frame000001/images/{01..NN}.jpg
    ├── ...
    └── pcds/downsample_<start>_<end>.ply   # voxel-downsampled init pcd

Our Stage 1 produces, per frame::

    work/frame_<NNNNNN>/sparse/0/{cameras,images,points3D}.{bin,txt}
    work/frame_<NNNNNN>/dense/images/{01..11}.jpg
    human/frame_<NNNNNN>_fused.ply

The sparse models are nearly identical across frames (the rig is fixed; only
SIFT 3D points differ). We use frame 0's cameras + extrinsics as the global
``sparse/0/``, renumbered to 0-indexed (LocalDyGS's Colmap_Dataset assumes
``cam_extrinsics[idx]`` works for ``idx`` in ``range(N)`` — i.e. dict keys are
0..N-1, not COLMAP's default 1..N).

We *symlink* per-frame undistorted images rather than copy them — saves
~40 GB on a 60-frame run.

Usage
-----

    python -m apps.reconstruction.stage_b_localdygs.prepare_localdygs_data \\
        /scratch/yubo/cow_1/9148_10581_output/stage_a/colmap_4d \\
        --output /scratch/yubo/cow_1/9148_10581_output/stage_b/localdygs_scene \\
        --frames-start-end 0 60

The ``--frames-start-end`` arg also names the init pcd file
(``pcds/downsample_<start>_<end>.ply``), so it must match what you'll pass
to ``train.py --frames_start_end`` later.

Resolution
----------

LocalDyGS's reader hardcodes ``downsample = 2.0`` — it halves the input
resolution. Cow data is captured at 4K specifically to preserve fur
texture (see PROJECT.md), so for the actual training run you'll want to
patch that to 1.0. This script makes no resolution-changing assumptions;
images are linked at native res.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def _discover_frames(work_root: Path) -> List[int]:
    """Return sorted list of integer frame indices found under work/."""
    out = []
    for p in sorted(work_root.glob("frame_*")):
        if not p.is_dir():
            continue
        try:
            out.append(int(p.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    if not out:
        raise SystemExit(f"[prepare] no frame_* dirs under {work_root}")
    return out


def _read_cameras_txt(path: Path) -> List[str]:
    """Return non-comment data lines from a COLMAP cameras.txt."""
    lines = []
    with path.open() as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                lines.append(s)
    return lines


def _read_images_txt(path: Path) -> List[Tuple[str, str]]:
    """Return list of (header_line, points_line) for each image entry.

    COLMAP's images.txt uses two lines per image: the first holds pose
    metadata, the second holds 2D-3D feature correspondences (which we
    don't need for Gaussian splatting init, since LocalDyGS gets its init
    from pcds/). The points line MAY be empty — we must not skip blanks
    or we'll silently merge two records into one.
    """
    pairs = []
    pending = None
    with path.open() as f:
        for line in f:
            s = line.rstrip("\n")
            if s.lstrip().startswith("#"):
                continue
            if pending is None:
                if s.strip() == "":
                    continue  # leading blank between header and prior record
                pending = s
            else:
                pairs.append((pending, s))
                pending = None
    if pending is not None:
        pairs.append((pending, ""))
    return pairs


def _renumber_cameras(lines: List[str]) -> Tuple[List[str], Dict[int, int]]:
    """Subtract 1 from each CAMERA_ID. Return new lines + old->new map."""
    out = []
    remap: Dict[int, int] = {}
    for line in lines:
        parts = line.split()
        old_id = int(parts[0])
        new_id = old_id - 1  # COLMAP IDs start at 1; LocalDyGS expects 0
        remap[old_id] = new_id
        parts[0] = str(new_id)
        out.append(" ".join(parts))
    return out, remap


def _renumber_images(pairs: List[Tuple[str, str]],
                     cam_remap: Dict[int, int]) -> List[Tuple[str, str]]:
    """Renumber IMAGE_ID and CAMERA_ID columns to 0-indexed."""
    out = []
    for header, points in pairs:
        parts = header.split()
        # COLMAP images.txt header format:
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        old_image_id = int(parts[0])
        old_camera_id = int(parts[8])
        parts[0] = str(old_image_id - 1)
        parts[8] = str(cam_remap.get(old_camera_id, old_camera_id - 1))
        out.append((" ".join(parts), points))
    return out


def _write_cameras_txt(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(lines)}\n")
        for line in lines:
            f.write(line + "\n")


def _write_images_txt(path: Path, pairs: List[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(pairs)}\n")
        for header, points in pairs:
            f.write(header + "\n")
            f.write(points + "\n")


def _write_empty_points3d_txt(path: Path) -> None:
    """Write a valid-but-empty COLMAP points3D.txt.

    LocalDyGS's reader only reads sparse/0/points3D.bin as a fallback when
    pcds/downsample_*.ply is missing. We always generate the pcds/ file,
    so points3D doesn't matter — but writing an empty txt keeps the
    sparse/0/ dir self-consistent (and lets external tools open the model).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("# Number of points: 0\n")


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def _build_init_pcd(human_dir: Path, frame_indices: List[int],
                    out_ply: Path, target_points: int = 90_000,
                    min_voxel: float = 0.005,
                    voxel_step: float = 0.005) -> None:
    """Combine per-frame fused.ply files, voxel-downsample to <= target points.

    Mirrors the strategy in LocalDyGS's scripts/downsample.py: stack all
    per-frame dense clouds, then voxel-downsample with increasing voxel
    size until the count drops below ``target_points``.
    """
    try:
        import open3d as o3d
    except ImportError as e:
        raise SystemExit(
            "[prepare] open3d required for init pcd; "
            "use ~/envs/cleanply/ which has it"
        ) from e

    pcd = o3d.geometry.PointCloud()
    found = 0
    for f in frame_indices:
        candidate = human_dir / f"frame_{f:06d}_fused.ply"
        if not candidate.exists():
            print(f"[prepare] WARNING: missing {candidate.name}", file=sys.stderr)
            continue
        cur = o3d.io.read_point_cloud(str(candidate))
        if len(cur.points) == 0:
            print(f"[prepare] WARNING: {candidate.name} is empty", file=sys.stderr)
            continue
        pcd += cur
        found += 1

    if found == 0:
        raise SystemExit(
            f"[prepare] no fused.ply files for frames {frame_indices[:5]}... in {human_dir}"
        )

    print(f"[prepare] init pcd: combined {found} frames → {len(pcd.points):,} pts")

    voxel = min_voxel
    while len(pcd.points) > target_points and voxel < 1.0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel)
        print(f"[prepare] voxel_down_sample({voxel:.3f}) → {len(pcd.points):,} pts")
        voxel += voxel_step

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_ply), pcd)
    print(f"[prepare] wrote {out_ply} ({len(pcd.points):,} pts, "
          f"{out_ply.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage_a_root", type=str,
                   help="Stage 1 output dir (contains work/ and human/), "
                        "e.g. /scratch/.../stage_a/colmap_4d")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="output scene dir (default: "
                        "<data>_output/stage_b/localdygs_scene/)")
    p.add_argument("--frames-start-end", nargs=2, type=int, metavar=("START", "END"),
                   default=None,
                   help="frame index range [START, END) into the discovered "
                        "list, INCLUSIVE start, EXCLUSIVE end. Defaults to "
                        "[0, len(frames)). Names the init pcd file — must "
                        "match what you'll pass to train.py --frames_start_end.")
    p.add_argument("--copy", action="store_true",
                   help="copy images instead of symlinking (default: symlink)")
    p.add_argument("--target-points", type=int, default=90_000,
                   help="voxel-downsample target for init pcd (default: 90000)")
    args = p.parse_args()

    stage_a_root = Path(args.stage_a_root).resolve()
    work_root = stage_a_root / "work"
    human_dir = stage_a_root / "human"
    if not work_root.is_dir():
        raise SystemExit(f"[prepare] no work/ dir under {stage_a_root}")
    if not human_dir.is_dir():
        raise SystemExit(f"[prepare] no human/ dir under {stage_a_root}")

    # Default puts the scene next to other stage_b outputs:
    #   <data>_output/stage_b/localdygs_scene/  (sibling of stage_a/colmap_4d)
    if args.output:
        out = Path(args.output).resolve()
    else:
        # stage_a_root looks like <data>_output/stage_a/colmap_4d
        # → walk up to <data>_output, then into stage_b/localdygs_scene
        data_output = stage_a_root.parents[1]  # <data>_output
        out = (data_output / "stage_b" / "localdygs_scene").resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"[prepare] output: {out}")

    all_frames = _discover_frames(work_root)
    print(f"[prepare] discovered {len(all_frames)} frames "
          f"({all_frames[0]:06d} ... {all_frames[-1]:06d})")

    if args.frames_start_end is None:
        start_idx, end_idx = 0, len(all_frames)
    else:
        start_idx, end_idx = args.frames_start_end
    if not (0 <= start_idx < end_idx <= len(all_frames)):
        raise SystemExit(
            f"[prepare] --frames-start-end {start_idx} {end_idx} out of range "
            f"[0, {len(all_frames)}]"
        )
    frames = all_frames[start_idx:end_idx]
    print(f"[prepare] using {len(frames)} frames "
          f"(indices {start_idx}..{end_idx - 1} → "
          f"{frames[0]:06d}..{frames[-1]:06d})")

    # 1. Renumber + write sparse/0/
    src_sparse = work_root / f"frame_{all_frames[0]:06d}" / "sparse" / "0"
    cam_lines = _read_cameras_txt(src_sparse / "cameras.txt")
    cam_lines, cam_remap = _renumber_cameras(cam_lines)
    img_pairs = _read_images_txt(src_sparse / "images.txt")
    img_pairs = _renumber_images(img_pairs, cam_remap)

    dst_sparse = out / "sparse" / "0"
    _write_cameras_txt(dst_sparse / "cameras.txt", cam_lines)
    _write_images_txt(dst_sparse / "images.txt", img_pairs)
    _write_empty_points3d_txt(dst_sparse / "points3D.txt")
    print(f"[prepare] wrote sparse/0/ "
          f"({len(cam_lines)} cams, {len(img_pairs)} images, ids 0..{len(cam_lines) - 1})")

    # 2. Symlink per-frame images
    n_linked = 0
    for new_idx, orig_frame in enumerate(frames):
        src_img_dir = work_root / f"frame_{orig_frame:06d}" / "dense" / "images"
        if not src_img_dir.is_dir():
            raise SystemExit(f"[prepare] missing {src_img_dir}")
        dst_img_dir = out / f"frame{new_idx:06d}" / "images"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(src_img_dir.iterdir()):
            if src.suffix.lower() in (".jpg", ".jpeg", ".png"):
                _link_or_copy(src, dst_img_dir / src.name, args.copy)
                n_linked += 1
    print(f"[prepare] {'copied' if args.copy else 'symlinked'} "
          f"{n_linked} images across {len(frames)} frames")

    # 3. Build init pcd matching the [start, end) the user will train on
    pcd_name = f"downsample_{start_idx}_{end_idx}.ply"
    _build_init_pcd(human_dir, frames, out / "pcds" / pcd_name,
                    target_points=args.target_points)

    print(f"\n[prepare] done. Train with:\n"
          f"  cd ~/github/LocalDyGS && python train.py "
          f"-s {out} -m output/<run_name> "
          f"--frames_start_end {start_idx} {end_idx} "
          f"--configs arguments/vrugz/basketball.py")


if __name__ == "__main__":
    main()
