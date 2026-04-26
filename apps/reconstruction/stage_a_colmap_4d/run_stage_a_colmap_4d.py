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

Output layout (rewritten 2026-04-26 to separate human deliverables from
COLMAP intermediates)::

    <out>/frame_<NNNNNN>/
      human/                       # for human review / downstream
        fused.ply                  # dense cow point cloud (the deliverable)
        sparse.ply                 # COLMAP-triangulated SIFT 3D points
        thumb_<cam>.jpg            # one input view for context
      work/                        # intermediate; safe to delete to free disk
        images/  masks/  database.db
        sparse/0/  dense/...  (~1.5 GB per frame)

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


def _expose_human_outputs(data_root: Path, frame: int, frame_dir: Path) -> None:
    """Copy / convert the deliverables from work/ → human/ for easy review."""
    work = frame_dir / "work"
    human = frame_dir / "human"
    human.mkdir(parents=True, exist_ok=True)

    # Dense cloud
    fused_src = work / "dense" / "fused.ply"
    if fused_src.exists() and fused_src.stat().st_size > 1024:
        shutil.copy2(fused_src, human / "fused.ply")
        print(f"[stage_a_colmap_4d] frame {frame}: human/fused.ply ({fused_src.stat().st_size/1e6:.1f} MB)")
    else:
        print(f"[stage_a_colmap_4d] WARNING: fused.ply missing/empty for frame {frame}")

    # Sparse cloud (converted from COLMAP binary → ASCII PLY)
    sparse_bin = work / "sparse" / "0" / "points3D.bin"
    if sparse_bin.exists():
        n = _export_sparse_ply(sparse_bin, human / "sparse.ply")
        print(f"[stage_a_colmap_4d] frame {frame}: human/sparse.ply ({n} pts)")

    # Thumbnail of one input view (cam 06 is usually the most central)
    images_root = data_root / "images"
    if images_root.is_dir():
        cam_names = sorted(d.name for d in images_root.iterdir() if d.is_dir())
        if cam_names:
            anchor = cam_names[len(cam_names) // 2]  # middle of the rig
            src = images_root / anchor / f"{frame:06d}.jpg"
            if src.exists():
                _save_thumbnail(src, human / f"thumb_cam{anchor}.jpg")


def _run_single_frame(data_root: Path, frame: int, frame_dir: Path,
                      neighbor: int, gpu_index: str, colmap: str) -> None:
    """Run the single-frame stage_a_colmap driver against one timestamp."""
    work_dir = frame_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m",
        "apps.reconstruction.stage_a_colmap.run_stage_a_colmap",
        str(data_root),
        "--output", str(work_dir),
        "--frame", str(frame),
        "--neighbor", str(neighbor),
        "--gpu_index", gpu_index,
        "--colmap", colmap,
    ]
    _run(cmd)
    _expose_human_outputs(data_root, frame, frame_dir)


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

    for i, f in enumerate(frames, start=1):
        frame_dir = out / f"frame_{f:06d}"
        fused_human = frame_dir / "human" / "fused.ply"
        if fused_human.exists() and fused_human.stat().st_size > 1024:
            print(f"[stage_a_colmap_4d] frame {f}: human/fused.ply present, skipping ({i}/{len(frames)})")
            continue
        print(f"\n{'='*60}\n[stage_a_colmap_4d] frame {f} ({i}/{len(frames)})\n{'='*60}")
        _run_single_frame(data_root, f, frame_dir, args.neighbor, args.gpu_index, args.colmap)

    print(f"\n[stage_a_colmap_4d] done; human deliverables at "
          f"{out}/frame_*/human/{{fused,sparse}}.ply")


if __name__ == "__main__":
    main()
