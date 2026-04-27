"""Stage A 4D extension: per-frame COLMAP MVS for a moving subject in a fixed multi-view rig.

Why this exists
---------------
The single-frame ``stage_a_colmap`` reconstructs only what the cameras see
at one instant. A half-circle rig misses the back of a static subject. But
if the subject *moves* through the rig (e.g. a cow walks around so different
sides face the cameras over time), the **sequence** of per-frame
reconstructions covers more of the surface than any single frame.

This driver does NOT solve cross-frame deformation — it produces N
independent dense clouds, one per timestamp, all in the same world coords.
For a true canonical (deformable) cow model, see notes on Path C
(LocalDyGS / BANMo) in the wider project plan.

Implementation
--------------
For each frame in ``--frames``, invokes the proven single-frame
``stage_a_colmap.run_stage_a_colmap`` via subprocess. Each per-frame run
produces its own sparse + dense, which means COLMAP auto-derives per-camera
depth bounds from each frame's triangulated SIFT points (essential — a
single global depth bound across cameras leaves close-near-plane cams with
hypothesis ranges that miss the surface).

Output layout (flattened 2026-04-26 so a single rsync of ``human/`` grabs
all deliverables — no per-frame subdir spelunking)::

    <out>/
      human/                                    # rsync this whole dir
        aggregated_4d.ply                       # color-coded union of frames
        frame_<NNNNNN>_fused.ply                # dense cow cloud per frame
        frame_<NNNNNN>_sparse.ply               # SIFT 3D points per frame
        frame_<NNNNNN>_thumb.jpg                # one input view per frame
      work/                                     # intermediate, safe to delete
        frame_<NNNNNN>/
          images/  masks/  database.db
          sparse/0/  dense/...                  # ~1.5 GB per frame

Usage
-----
    # 5-frame proof-of-concept (1 sec spacing at 30 fps)
    python -m apps.reconstruction.stage_a_colmap_4d.run_stage_a_colmap_4d \
        /scratch/yubo/cow_1/9148_10581 --frames 0:150:30

Timing on cow_1 (11 cams @ 4K, full A100): ~10 min/frame end-to-end.
For parallelism use the SLURM array template at
``scripts/slurm/run_4d_array.sh.template``.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("[stage_a_colmap_4d] $", " ".join(shlex.quote(c) for c in cmd), flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _parse_frames(s: str) -> list[int]:
    """``START:END[:STEP]`` (range, end-exclusive) or ``N1,N2,N3`` (explicit)."""
    if ":" in s:
        parts = [int(p) for p in s.split(":")]
        if len(parts) == 2:
            return list(range(parts[0], parts[1]))
        if len(parts) == 3:
            return list(range(parts[0], parts[1], parts[2]))
        raise ValueError(f"--frames range needs 2 or 3 colon-separated ints; got {s!r}")
    return [int(p) for p in s.split(",") if p.strip()]


def _export_sparse_ply(points3d_bin: Path, out_ply: Path) -> int:
    """Convert COLMAP points3D.bin → simple ASCII PLY for human review."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from easymocap.mytools.colmap_structure import read_points3d_binary
    pts = read_points3d_binary(str(points3d_bin))
    if not pts:
        return 0
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    with open(out_ply, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p in pts.values():
            xyz, rgb = p.xyz, p.rgb
            f.write(f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} "
                    f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n")
    return len(pts)


def _save_thumbnail(src_jpg: Path, out_jpg: Path, max_side: int = 720) -> None:
    """Downscale one input image so the human dir has a quick visual reference."""
    import cv2
    img = cv2.imread(str(src_jpg))
    if img is None:
        return
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_jpg), img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])


def _expose_human_outputs(data_root: Path, frame: int,
                          work_frame_dir: Path, human_dir: Path) -> None:
    """Copy/convert deliverables from work/frame_<N>/ → human/frame_<N>_*."""
    human_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"frame_{frame:06d}_"

    # Dense cloud
    fused_src = work_frame_dir / "dense" / "fused.ply"
    if fused_src.exists() and fused_src.stat().st_size > 1024:
        shutil.copy2(fused_src, human_dir / f"{prefix}fused.ply")
        print(f"[stage_a_colmap_4d] frame {frame}: human/{prefix}fused.ply "
              f"({fused_src.stat().st_size/1e6:.1f} MB)")
    else:
        print(f"[stage_a_colmap_4d] WARNING: fused.ply missing/empty for frame {frame}")

    # Sparse cloud (converted from COLMAP binary → ASCII PLY)
    sparse_bin = work_frame_dir / "sparse" / "0" / "points3D.bin"
    if sparse_bin.exists():
        n = _export_sparse_ply(sparse_bin, human_dir / f"{prefix}sparse.ply")
        print(f"[stage_a_colmap_4d] frame {frame}: human/{prefix}sparse.ply ({n} pts)")

    # Thumbnail of one input view (middle camera in the rig)
    images_root = data_root / "images"
    if images_root.is_dir():
        cam_names = sorted(d.name for d in images_root.iterdir() if d.is_dir())
        if cam_names:
            anchor = cam_names[len(cam_names) // 2]
            src = images_root / anchor / f"{frame:06d}.jpg"
            if src.exists():
                _save_thumbnail(src, human_dir / f"{prefix}thumb.jpg")


def _run_single_frame(data_root: Path, frame: int, work_frame_dir: Path,
                      human_dir: Path, neighbor: int, gpu_index: str,
                      colmap: str) -> None:
    """Run single-frame stage_a_colmap into work/, then expose into human/."""
    work_frame_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m",
        "apps.reconstruction.stage_a_colmap.run_stage_a_colmap",
        str(data_root),
        "--output", str(work_frame_dir),
        "--frame", str(frame),
        "--neighbor", str(neighbor),
        "--gpu_index", gpu_index,
        "--colmap", colmap,
        # Always triangulate on the unmasked image: a tight cow mask leaves
        # too few SIFT points to derive per-cam depth bounds and PatchMatch
        # silently fails. Dense step still applies the mask.
        "--no-mask-sparse",
    ]
    _run(cmd)
    _expose_human_outputs(data_root, frame, work_frame_dir, human_dir)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stage A 4D: per-frame COLMAP MVS over a sequence",
    )
    p.add_argument("data_root", type=str)
    p.add_argument("--output", type=str, default=None,
                   help="output dir (default: <data_root>_output/stage_a/colmap_4d/)")
    p.add_argument("--frames", type=str, required=True,
                   help="frame range 'START:END[:STEP]' or explicit 'N1,N2,N3'")
    p.add_argument("--neighbor", "-k", type=int, default=6,
                   help="K nearest cameras for PatchMatch sources")
    p.add_argument("--colmap", default="colmap")
    p.add_argument("--gpu_index", default="0",
                   help="GPU index for patch_match_stereo (default: 0)")
    args = p.parse_args()

    data_root = Path(args.data_root)
    out = Path(args.output) if args.output else (
        Path(f"{data_root}_output") / "stage_a" / "colmap_4d")
    out.mkdir(parents=True, exist_ok=True)

    frames = _parse_frames(args.frames)
    if not frames:
        raise SystemExit("[stage_a_colmap_4d] empty --frames")
    print(f"[stage_a_colmap_4d] {len(frames)} frames: "
          f"{frames[:5]}{'...' if len(frames) > 5 else ''}; output={out}")

    human_dir = out / "human"
    work_root = out / "work"
    human_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(frames, start=1):
        prefix = f"frame_{f:06d}_"
        fused_human = human_dir / f"{prefix}fused.ply"
        if fused_human.exists() and fused_human.stat().st_size > 1024:
            print(f"[stage_a_colmap_4d] frame {f}: {fused_human.name} present, skipping ({i}/{len(frames)})")
            continue
        print(f"\n{'='*60}\n[stage_a_colmap_4d] frame {f} ({i}/{len(frames)})\n{'='*60}")
        _run_single_frame(data_root, f, work_root / f"frame_{f:06d}",
                          human_dir, args.neighbor, args.gpu_index, args.colmap)

    print(f"\n[stage_a_colmap_4d] done; human deliverables in {human_dir}/")


if __name__ == "__main__":
    main()
