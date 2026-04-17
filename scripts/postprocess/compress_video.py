#!/usr/bin/env python3
"""
Compress a video for easier sharing/upload (e.g. Slack/Canvas).

Requires ffmpeg in PATH.

Examples:
  python scripts/postprocess/compress_video.py /path/in.mp4
  python scripts/postprocess/compress_video.py /path/in.mp4 --target-mb 40
  python scripts/postprocess/compress_video.py /path/in.mp4 --max-width 1280 --crf 30
"""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path


def _require_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found in PATH. Please install ffmpeg first.")
    if not shutil.which("ffprobe"):
        raise SystemExit("ffprobe not found in PATH. Please install ffmpeg/ffprobe first.")


def _probe_duration_seconds(video: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    try:
        dur = float(out)
    except ValueError as exc:
        raise RuntimeError(f"Failed to parse duration from ffprobe output: {out!r}") from exc
    if dur <= 0:
        raise RuntimeError(f"Invalid video duration: {dur}")
    return dur


def _derive_output_path(input_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg).expanduser().resolve()
    stem = input_path.stem
    return input_path.with_name(f"{stem}_compressed.mp4")


def _build_scale_filter(max_width: int | None) -> str | None:
    if max_width is None:
        return None
    # Keep aspect ratio; ensure even height for H.264 compatibility.
    return f"scale='min(iw,{max_width})':-2"


def _run_ffmpeg(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"ffmpeg failed with exit code {exc.returncode}") from exc


def compress_crf(
    input_path: Path,
    output_path: Path,
    *,
    crf: int,
    preset: str,
    max_width: int | None,
    fps: int | None,
    audio_bitrate_k: int,
) -> None:
    vf = _build_scale_filter(max_width)
    cmd = ["ffmpeg", "-y", "-i", str(input_path)]
    if vf:
        cmd += ["-vf", vf]
    if fps:
        cmd += ["-r", str(fps)]
    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        f"{audio_bitrate_k}k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    _run_ffmpeg(cmd)


def compress_target_size(
    input_path: Path,
    output_path: Path,
    *,
    target_mb: float,
    preset: str,
    max_width: int | None,
    fps: int | None,
    audio_bitrate_k: int,
) -> None:
    duration = _probe_duration_seconds(input_path)
    target_total_kbps = int((target_mb * 8192) / duration)
    video_kbps = max(300, target_total_kbps - audio_bitrate_k)
    maxrate_kbps = int(video_kbps * 1.25)
    bufsize_kbps = int(video_kbps * 2.0)

    vf = _build_scale_filter(max_width)
    passlog = str(output_path.with_suffix("")) + ".2passlog"

    base = ["ffmpeg", "-y", "-i", str(input_path)]
    if vf:
        base += ["-vf", vf]
    if fps:
        base += ["-r", str(fps)]

    pass1 = base + [
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-b:v",
        f"{video_kbps}k",
        "-maxrate",
        f"{maxrate_kbps}k",
        "-bufsize",
        f"{bufsize_kbps}k",
        "-pass",
        "1",
        "-passlogfile",
        passlog,
        "-an",
        "-f",
        "mp4",
        "/dev/null",
    ]
    _run_ffmpeg(pass1)

    pass2 = base + [
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-b:v",
        f"{video_kbps}k",
        "-maxrate",
        f"{maxrate_kbps}k",
        "-bufsize",
        f"{bufsize_kbps}k",
        "-pass",
        "2",
        "-passlogfile",
        passlog,
        "-c:a",
        "aac",
        "-b:a",
        f"{audio_bitrate_k}k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    _run_ffmpeg(pass2)

    for p in [
        Path(passlog + "-0.log"),
        Path(passlog + "-0.log.mbtree"),
        Path(passlog + ".log"),
        Path(passlog + ".log.mbtree"),
    ]:
        if p.exists():
            p.unlink()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compress MP4 for sharing/upload.")
    p.add_argument("input", help="Input video path")
    p.add_argument("-o", "--output", default=None, help="Output path (default: <input>_compressed.mp4)")
    p.add_argument(
        "--target-mb",
        type=float,
        default=None,
        help="Target output size in MB (uses 2-pass bitrate mode).",
    )
    p.add_argument(
        "--crf",
        type=int,
        default=28,
        help="Quality mode CRF when --target-mb is not set (lower=better quality, larger file).",
    )
    p.add_argument(
        "--preset",
        default="medium",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower"],
        help="x264 preset (slower = better compression).",
    )
    p.add_argument("--max-width", type=int, default=1280, help="Max output width (default: 1280).")
    p.add_argument("--fps", type=int, default=24, help="Output FPS cap (default: 24).")
    p.add_argument("--audio-kbps", type=int, default=96, help="AAC audio bitrate kbps (default: 96).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _require_ffmpeg()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")
    output_path = _derive_output_path(input_path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.target_mb is not None:
        if args.target_mb <= 0:
            raise SystemExit("--target-mb must be > 0")
        compress_target_size(
            input_path,
            output_path,
            target_mb=args.target_mb,
            preset=args.preset,
            max_width=args.max_width,
            fps=args.fps,
            audio_bitrate_k=args.audio_kbps,
        )
    else:
        compress_crf(
            input_path,
            output_path,
            crf=args.crf,
            preset=args.preset,
            max_width=args.max_width,
            fps=args.fps,
            audio_bitrate_k=args.audio_kbps,
        )

    in_mb = input_path.stat().st_size / (1024 * 1024)
    out_mb = output_path.stat().st_size / (1024 * 1024)
    ratio = out_mb / in_mb if in_mb > 0 else math.nan
    print(f"[compress_video] input : {input_path} ({in_mb:.1f} MB)")
    print(f"[compress_video] output: {output_path} ({out_mb:.1f} MB)")
    print(f"[compress_video] ratio : {ratio:.3f}")


if __name__ == "__main__":
    main()

