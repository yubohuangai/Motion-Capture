"""
Crop the reference-camera video (see config: reference_camera_index, crop_region).
All crop parameters are read from config.yaml — edit ``crop_region`` there only.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import cv2
from tqdm import tqdm

from session_paths import get_reference_camera_index, load_config, video_path_at


def parse_crop_region(config: dict[str, Any]) -> tuple[int, int, int, int]:
    """
    Read ``crop_region`` from config.

    Preferred: inclusive grid edges ``x_left``, ``x_right``, ``y_top``, ``y_bottom``
    (all pixel coordinates inside the crop).

    Alternatives: dict ``x``, ``y``, ``w``, ``h`` or list ``[x, y, w, h]``.
    """
    cr = config.get("crop_region")
    if cr is None:
        raise ValueError(
            'config.yaml: add "crop_region" (x_left/x_right/y_top/y_bottom or x/y/w/h).'
        )
    if isinstance(cr, (list, tuple)):
        if len(cr) != 4:
            raise ValueError("crop_region list must be exactly [x, y, w, h].")
        return int(cr[0]), int(cr[1]), int(cr[2]), int(cr[3])
    if isinstance(cr, Mapping):
        if all(k in cr for k in ("x_left", "x_right", "y_top", "y_bottom")):
            xl = int(cr["x_left"])
            xr = int(cr["x_right"])
            yt = int(cr["y_top"])
            yb = int(cr["y_bottom"])
            if xr < xl or yb < yt:
                raise ValueError(
                    "crop_region: need x_left <= x_right and y_top <= y_bottom (inclusive indices)."
                )
            w = xr - xl + 1
            h = yb - yt + 1
            return xl, yt, w, h
        try:
            return int(cr["x"]), int(cr["y"]), int(cr["w"]), int(cr["h"])
        except KeyError as e:
            raise KeyError(
                'crop_region: use x_left, x_right, y_top, y_bottom (inclusive) or x, y, w, h.'
            ) from e
    raise TypeError("crop_region must be a dict or a list of four integers.")


def load_crop_video_path(config_path):
    config = load_config(config_path)
    idx = get_reference_camera_index(config)
    return video_path_at(config, idx)


def crop_video(video_path, crop_region: tuple[int, int, int, int]):
    x, y, w, h = crop_region

    original_dir = os.path.dirname(video_path)
    base_name = os.path.basename(video_path)
    name, ext = os.path.splitext(base_name)

    renamed_original_path = os.path.join(original_dir, f"{name}_ori{ext}")
    cropped_output_path = os.path.join(original_dir, f"{name}{ext}")

    # If renamed file already exists, assume cropping was done
    # if os.path.exists(cropped_output_path):
    #     print(f"Cropped file already exists: {cropped_output_path}. Aborting.")
    #     return

    os.rename(video_path, renamed_original_path)
    print(f"Original file renamed to: {renamed_original_path}")

    cap = cv2.VideoCapture(renamed_original_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {renamed_original_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(cropped_output_path, fourcc, fps, (w, h))

    with tqdm(total=total_frames, desc="Cropping video", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cropped_frame = frame[y:y+h, x:x+w]
            out.write(cropped_frame)
            pbar.update(1)

    cap.release()
    out.release()
    print(f"Cropped video saved to: {cropped_output_path}")


if __name__ == "__main__":
    config_file = Path(__file__).resolve().parent / "config.yaml"
    config = load_config(config_file)
    video_path = video_path_at(config, get_reference_camera_index(config))
    crop_video(video_path, crop_region=parse_crop_region(config))
