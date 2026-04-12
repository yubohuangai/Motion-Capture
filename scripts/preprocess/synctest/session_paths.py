"""
Resolve capture session layout from a single data root.

Layout (per camera folder under data_root):
  <camera>/VID/VID_*.mp4
  <camera>/<YYYYMMDD_HHMMSS>.csv   (stem matches the VID_* video stem without "VID_" prefix)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_NAME = "config.yaml"


def resolve_config_path(config_path: str | Path) -> Path:
    p = Path(config_path)
    if p.is_absolute():
        return p
    return CONFIG_DIR / p


def _camera_dir_sort_key(name: str) -> tuple:
    if name.isdigit():
        return (0, int(name), name)
    return (1, name)


def _is_crop_backup_mp4(path: Path) -> bool:
    """True if this looks like crop_video.py's renamed original (*_ori.mp4)."""
    return path.suffix.lower() == ".mp4" and path.stem.endswith("_ori")


def discover_session_videos(
    data_root: Path,
    *,
    vid_subdir: str = "VID",
    glob_pattern: str = "VID_*.mp4",
) -> list[Path]:
    """
    Return absolute paths to one mp4 per camera folder, sorted by camera folder name.
    Files named ``*_ori.mp4`` (crop backups) are ignored when picking a clip.
    """
    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root is not a directory: {data_root}")

    candidates: list[tuple[str, Path]] = []
    for child in sorted(data_root.iterdir(), key=lambda p: _camera_dir_sort_key(p.name)):
        if not child.is_dir():
            continue
        vid_dir = child / vid_subdir
        if not vid_dir.is_dir():
            continue
        mp4s = sorted(p for p in vid_dir.glob(glob_pattern) if not _is_crop_backup_mp4(p))
        if not mp4s:
            continue
        if len(mp4s) > 1:
            warnings.warn(
                f"Multiple {glob_pattern} in {vid_dir}; using {mp4s[0].name}",
                stacklevel=2,
            )
        candidates.append((child.name, mp4s[0].resolve()))

    if not candidates:
        raise FileNotFoundError(
            f"No camera folders with {vid_subdir}/{glob_pattern} under {data_root}"
        )
    return [p for _, p in candidates]


def _filter_cameras(videos: list[Path], cameras: list[str]) -> list[Path]:
    by_folder = {p.parent.parent.name: p for p in videos}
    out = []
    missing = []
    for c in cameras:
        if c in by_folder:
            out.append(by_folder[c])
        else:
            missing.append(c)
    if missing:
        raise FileNotFoundError(
            f"cameras not found under data_root (no {missing[0]}/VID/...): {missing}"
        )
    return out


def apply_data_root(config: dict[str, Any]) -> None:
    """
    If config contains non-empty `data_root`, replace all `video_path_*` entries
    with paths discovered under that root (ordered by camera folder name).

    Optional `cameras: ["01", "02", ...]` restricts and orders cameras.
    """
    root = config.get("data_root")
    if root is None:
        return
    s = str(root).strip()
    if not s:
        return

    data_root = Path(s)
    videos = discover_session_videos(data_root)
    cameras = config.get("cameras")
    if cameras:
        if not isinstance(cameras, list):
            raise TypeError("config 'cameras' must be a list of folder names")
        videos = _filter_cameras(videos, [str(x) for x in cameras])

    for k in list(config.keys()):
        if k.startswith("video_path_"):
            del config[k]

    for i, vp in enumerate(videos):
        config[f"video_path_{i}"] = str(vp)


def read_data_root_field(path: str | Path | None = None) -> Path:
    """
    Return ``data_root`` from synctest YAML without running ``apply_data_root``
    (so ``video_path_*`` are left unchanged). Use this when only the session root path is needed.
    """
    cfg_path = resolve_config_path(path or DEFAULT_CONFIG_NAME)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not config:
        config = {}
    root = config.get("data_root")
    if root is None or not str(root).strip():
        raise ValueError(
            f"Set data_root in {cfg_path} (or pass the session root directory on the command line)."
        )
    return Path(str(root).strip()).expanduser().resolve()


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML from src/config.yaml (or given path), then apply data_root expansion."""
    cfg_path = resolve_config_path(path or DEFAULT_CONFIG_NAME)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config is None:
        config = {}
    apply_data_root(config)
    return config


def get_ordered_video_paths(config: dict[str, Any]) -> list[str]:
    items: list[tuple[int, str]] = []
    for key, val in config.items():
        if not key.startswith("video_path_"):
            continue
        suffix = key[len("video_path_") :]
        if not suffix.isdigit():
            continue
        items.append((int(suffix), str(val)))
    if not items:
        raise ValueError(
            "No video_path_* entries in config. Set data_root or list video_path_* explicitly."
        )
    return [vp for _, vp in sorted(items)]


def get_reference_camera_index(config: dict[str, Any]) -> int:
    """
    Single camera index for draw_grid and crop_video.
    0-based in discovery order; -1 means last camera.
    Prefer ``reference_camera_index``; else legacy ``draw_grid_camera_index`` or ``crop_camera_index``.
    """
    if "reference_camera_index" in config:
        v = config["reference_camera_index"]
    elif "draw_grid_camera_index" in config:
        v = config["draw_grid_camera_index"]
    elif "crop_camera_index" in config:
        v = config["crop_camera_index"]
    else:
        return 0
    return int(v)


def video_path_at(config: dict[str, Any], index: int) -> str:
    paths = get_ordered_video_paths(config)
    if index < 0:
        index += len(paths)
    if not 0 <= index < len(paths):
        raise IndexError(f"camera index {index} out of range for {len(paths)} videos")
    return paths[index]
