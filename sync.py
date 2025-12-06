import multiprocessing
import os
import cv2
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import re
import logging
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import shutil
import os
import re
from pathlib import Path
import time
import re
from concurrent.futures import ProcessPoolExecutor


ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.npy', '.png', '.pcd']


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-video timestamp matching")

    parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing 01, 02, 03, ... camera folders"
    )

    parser.add_argument(
        "--cams",
        type=int,
        required=True,
        help="Number of camera folders"
    )

    parser.add_argument(
        "--threshold",
        type=str,
        default="16ms",
        help="Matching tolerance (ns, us, ms, s). Example: 16ms"
    )

    parser.add_argument(
        "--extract",
        type=str,
        default="false",
        help="Extract frames (true/false)"
    )

    return parser.parse_args()


def collect_video_paths(root, num_cams):
    video_paths = []
    valid_indices = []

    for i in range(1, num_cams + 1):
        cam_dir = Path(root) / f"{i:02d}" / "VID"
        mp4s = list(cam_dir.glob("*.mp4"))

        if len(mp4s) == 0:
            logging.warning(f"[WARN] No video found in {cam_dir}, skipping this camera.")
            continue

        if len(mp4s) > 1:
            logging.warning(f"[WARN] Multiple videos in {cam_dir}, using the first one.")

        video_paths.append(str(mp4s[0]))
        valid_indices.append(i)

    if len(video_paths) == 0:
        raise RuntimeError("No valid cameras found.")

    logging.info(f"Using cameras: {valid_indices}")
    return video_paths


def extract_frames_cpu(video_path_output):
    """
    Extract frames from a video using CPU only (no GPU).
    Each video runs in its own process using 1 CPU core.
    """
    video_path, output_dir = video_path_output
    video_path = str(video_path)
    output_dir = str(output_dir)

    # Clean and recreate directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_pattern = os.path.join(output_dir, "%06d.jpg")
    video_name = os.path.basename(video_path)

    # CPU extraction, no multi-thread per process
    cmd = f'ffmpeg -threads 1 -vsync 0 -i "{video_path}" -q:v 1 "{output_pattern}"'
    logging.info(f"[{video_name}] Starting FFmpeg extraction...")
    os.system(cmd)

    logging.info(f"[{video_name}] Extraction complete → {output_dir}")
    return video_path


def extract_frames(video_path, output_dir, gpu=True, script_only=False):
    """
    Extract exact original frames from a video using FFmpeg.
    - Uses GPU decoding if available (hevc_cuvid / h264_cuvid)
    - Preserves all original frames (no resampling)
    - Optionally generates and saves a .sh script instead of executing directly
    """

    # Remove output dir if exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.basename(video_path)
    output_pattern = os.path.join(output_dir, "%06d.jpg")

    # Decide GPU codec
    # ffprobe can tell us, but we infer from extension
    if gpu:
        if video_name.lower().endswith(".hevc") or video_name.lower().endswith(".h265") or "hevc" in video_name.lower():
            decoder = "hevc_cuvid"
        else:
            decoder = "h264_cuvid"
    else:
        decoder = "h264"

    cmd = (
        f'ffmpeg -vsync 0 -i "{video_path}" -q:v 1 "{output_pattern}"'
    )

    # Save a script for reproducibility
    script_path = os.path.join(output_dir, f"extract_{Path(video_name).stem}.sh")
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(cmd + "\n")

    os.chmod(script_path, 0o755)

    if script_only:
        logging.info(f"[{video_name}] Extraction script saved to {script_path}")
        return

    logging.info(f"Running FFmpeg GPU extraction for {video_name}")
    start_time = time.time()

    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"FFmpeg extraction failed for {video_path}")

    elapsed = time.time() - start_time
    logging.info(f"[{video_name}] Extracted frames → {output_dir} in {elapsed:.1f}s")


def format_threshold_filename(threshold_ns):
    """Convert nanoseconds to a compact filename-safe string like 30ms or 250us."""
    if threshold_ns % 1e9 == 0:
        return f"{int(threshold_ns / 1e9)}s"
    elif threshold_ns % 1e6 == 0:
        return f"{int(threshold_ns / 1e6)}ms"
    elif threshold_ns % 1e3 == 0:
        return f"{int(threshold_ns / 1e3)}us"
    else:
        return f"{threshold_ns}ns"


def parse_duration_ns(duration_str):
    """Parse human-friendly duration strings to nanoseconds."""
    units = {
        "ns": 1,
        "us": int(1e3),
        "ms": int(1e6),
        "s":  int(1e9)
    }

    match = re.fullmatch(r"(\d+(?:\.\d+)?)(ns|us|ms|s)", duration_str.strip())
    if not match:
        raise ValueError(f"Invalid duration format: '{duration_str}'")

    value, unit = match.groups()
    return int(float(value) * units[unit])


def setup_logger(log_file_path):
    """
    Configures logging to write messages to both a file and the terminal.
    If the file exists, new messages will be appended.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def match_frames_from_csv(csv_path_left, csv_path_right, output_csv_path, threshold_ns):
    """
    Match frame timestamps from two CSV files instead of image filenames.
    Each CSV should have one timestamp per line (as integer nanoseconds).
    """
    # Load CSVs
    df_left = pd.read_csv(csv_path_left, header=None, names=["t"])
    df_right = pd.read_csv(csv_path_right, header=None, names=["t"])

    # Prepare dataframes
    left = pd.DataFrame({
        't': df_left["t"].astype(np.int64),
        'left': df_left["t"].astype(np.int64)
    })

    right = pd.DataFrame({
        't': df_right["t"].astype(np.int64),
        'right_int': df_right["t"].astype(np.int64),
        'right': df_right["t"].astype(str)
    })

    # Perform matching
    df = pd.merge_asof(
        left.sort_values('t'),
        right.sort_values('t'),
        on='t',
        tolerance=threshold_ns,
        allow_exact_matches=True,
        direction='nearest'
    )
    df = df.dropna()
    df = df.drop(['t', 'right_int'], axis=1).reset_index(drop=True)

    # Save result
    df.to_csv(output_csv_path, index=False)
    logging.info(f"Matched {df.shape[0]} frame pairs from CSVs (threshold = {threshold_ns / 1e6:.1f} ms) → {output_csv_path}")


def match_frames_full_from_csv(csv_path_left, csv_path_right, output_csv_path, threshold_ns):
    """
    Match all timestamps from left CSV with right CSV (full match), including unmatched.
    Each CSV should have one timestamp per line (as integer nanoseconds).
    """
    # Load CSVs
    df_left = pd.read_csv(csv_path_left, header=None, names=["t"])
    df_right = pd.read_csv(csv_path_right, header=None, names=["t"])

    # Prepare dataframes
    left_df = pd.DataFrame({
        't': df_left["t"].astype(np.int64),
        'left': df_left["t"].astype(np.int64)
    })

    right_df = pd.DataFrame({
        't': df_right["t"].astype(np.int64),
        'right_int': df_right["t"].astype(np.int64),
        'right': df_right["t"].astype(str)
    })

    # Match frames with merge_asof
    matched_df = pd.merge_asof(
        left_df.sort_values('t'),
        right_df.sort_values('t'),
        on='t',
        tolerance=threshold_ns,
        allow_exact_matches=True,
        direction='nearest'
    )

    matched_df = matched_df.reset_index(drop=True)
    full_output_df = matched_df[['left', 'right']].copy()

    # Fill unmatched with empty string
    full_output_df['left'] = full_output_df['left'].astype("Int64")
    full_output_df['right'] = full_output_df['right'].fillna('').astype(str)

    # Add match status
    full_output_df["matched"] = full_output_df["right"].apply(lambda x: x != "")
    full_output_df.to_csv(output_csv_path, index=False)

    logging.info(f"Full match completed: {len(full_output_df)} entries → {output_csv_path}")

def match_frames_from_csv_multi(csv_path_left, csv_path_right_list, output_csv_path, threshold_ns):
    df_left = pd.read_csv(csv_path_left, header=None, names=["t"])
    base_df = pd.DataFrame({'t': df_left["t"].astype(np.int64), 'left': df_left["t"].astype(str)})

    for idx, csv_path_right in enumerate(csv_path_right_list):
        df_right = pd.read_csv(csv_path_right, header=None, names=["t"])
        shifted_idx = idx + 1
        right_df = pd.DataFrame({
            't': df_right["t"].astype(np.int64),
            f'right{shifted_idx}_int': df_right["t"].astype(np.int64),
            f'right{shifted_idx}': df_right["t"].astype(str)
        })

        base_df = pd.merge_asof(
            base_df.sort_values('t'),
            right_df.sort_values('t'),
            on='t',
            tolerance=threshold_ns,
            allow_exact_matches=True,
            direction='nearest'
        )

        base_df.drop(columns=[f'right{shifted_idx}_int'], inplace=True)

    # Drop all rows with any missing values
    base_df.dropna(inplace=True)

    # Drop the 't' column used for merge key
    base_df.drop(columns=['t'], inplace=True)

    base_df.to_csv(output_csv_path, index=False, header=False)
    logging.info(f"Multi-match → {output_csv_path}")


def match_frames_full_from_csv_multi(csv_path_left, csv_path_right_list, output_csv_path, threshold_ns, duration_0):
    df_left = pd.read_csv(csv_path_left, header=None, names=["t"])
    base_df = pd.DataFrame({'t': df_left["t"].astype(np.int64), 'left': df_left["t"].astype(np.int64)})

    # Merge each right stream
    for idx, csv_path_right in enumerate(csv_path_right_list, start=1):
        df_right = pd.read_csv(csv_path_right, header=None, names=["t"])
        right_df = pd.DataFrame({
            't': df_right["t"].astype(np.int64),
            f'right{idx}_int': df_right["t"].astype(np.int64),
            f'right{idx}': df_right["t"].astype(str)
        })
        base_df = pd.merge_asof(
            base_df.sort_values('t'),
            right_df.sort_values('t'),
            on='t',
            tolerance=threshold_ns,
            allow_exact_matches=True,
            direction='nearest'
        )
        base_df.drop(columns=[f'right{idx}_int'], inplace=True)

    base_df = base_df.reset_index(drop=True)

    # Create matched flags for each right stream
    num_rights = len(csv_path_right_list)
    for idx in range(1, num_rights + 1):
        base_df[f'right{idx}'] = base_df[f'right{idx}'].fillna('').astype(str)
        base_df[f'matched_{idx}'] = base_df[f'right{idx}'].apply(lambda x: x != '')

    # We don't need 't' in the output
    base_df.drop(columns=['t'], inplace=True)

    # ---- Stats (generalized) ----
    total_frames = len(base_df)
    matched_counts = {idx: int(base_df[f'matched_{idx}'].sum()) for idx in range(1, num_rights + 1)}

    all_mask = base_df[[f'matched_{idx}' for idx in range(1, num_rights + 1)]].all(axis=1)
    any_mask = base_df[[f'matched_{idx}' for idx in range(1, num_rights + 1)]].any(axis=1)

    matched_all = int(all_mask.sum())
    matched_any = int(any_mask.sum())

    # matched_i_only: matched to stream i and to no other streams
    only_counts = {}
    for idx in range(1, num_rights + 1):
        others = [f'matched_{j}' for j in range(1, num_rights + 1) if j != idx]
        only_mask = base_df[f'matched_{idx}'] & (~base_df[others].any(axis=1))
        only_counts[idx] = int(only_mask.sum())

    # ---- Logging ----
    logging.info(f"Total frames in device_0: {total_frames}")
    for idx in range(1, num_rights + 1):
        logging.info(f"matched_{idx}: {matched_counts[idx]}")
    logging.info(f"matched_all: {matched_all}")
    for idx in range(1, num_rights + 1):
        logging.info(f"matched_{idx}_only: {only_counts[idx]}")
    logging.info(f"matched_any (matched to at least one right stream): {matched_any}")

    frame_rate_ratio = matched_all / duration_0 if duration_0 > 0 else 0
    logging.info(f"Output video FPS = matched_all / length of video 0 = {frame_rate_ratio:.2f}")

    # Save
    base_df.to_csv(output_csv_path, index=False)
    logging.info(f"Multi-match (full) → {output_csv_path}")


def parse_duration_seconds(duration_str):
    units = {
        'ns': 1e-9,
        'us': 1e-6,
        'ms': 1e-3,
        's': 1.0
    }
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(ns|us|ms|s)", duration_str.strip())
    if not match:
        raise ValueError(f"Invalid duration format: {duration_str}")
    value, unit = match.groups()
    return float(value) * units[unit]


def get_csv_path_from_video(video_path):
    """
    Given a video file path, returns the corresponding timestamp CSV file path
    by parsing the video name and assuming the CSV is located in the parent directory.
    """
    video_name = Path(video_path).stem
    match = re.search(r"VID_((\d|_)+)", video_name)
    if not match:
        raise ValueError(f"[ERROR] Video name format is incorrect: {video_name}")
    video_date = match.group(1)
    csv_path = Path(video_path).parent.parent / f"{video_date}.csv"
    return csv_path


def extract_frame_data(target_dir, video_path):
    """
    Renames extracted frames in `target_dir` using timestamps
    from the original CSV associated with `video_path`.
    Assumes frame count == timestamp count.
    """
    csv_path = get_csv_path_from_video(video_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"[ERROR] Timestamp CSV not found: {csv_path}")

    timestamps = [line.strip() for line in csv_path.open() if line.strip()]
    target_dir = Path(target_dir)

    def natural_key(f):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', f.name)]

    frame_files = sorted(
        [f for f in target_dir.iterdir() if f.suffix.lower() in ALLOWED_EXTENSIONS],
        key=natural_key
    )
    if len(frame_files) != len(timestamps):
        raise ValueError(
            f"Frame count ({len(frame_files)}) does not match timestamp count ({len(timestamps)})"
        )

    for frame_file, ts in zip(frame_files, timestamps):
        new_name = target_dir / f"{ts}{frame_file.suffix}"
        frame_file.rename(new_name)

    logging.info(
        f"[{video_path}] Renamed {len(frame_files)} frames in {target_dir} using original CSV timestamps."
    )

def main():
    # --- Parse command-line arguments ---
    args = parse_args()
    root = args.root
    num_videos = args.cams
    threshold_str = args.threshold
    threshold_ns = parse_duration_ns(threshold_str)
    extract_flag = args.extract.lower() == "true"

    # --- Collect video paths ---
    video_paths = collect_video_paths(root, num_videos)
    num_videos = len(video_paths)   # update real count
    # video_path_0 = video_paths[0]
    video_name_0 = Path(video_path_0).stem

    # --- Prepare output directory ---
    base_output = os.path.join("output", "exp", video_name_0)
    os.makedirs(base_output, exist_ok=True)

    # --- Setup logger ---
    log_file_path = os.path.join(base_output, "match.log")
    setup_logger(log_file_path)
    logging.info("==== New matching session started ====")
    logging.info(f"Loaded {num_videos} video paths from command line")

    # --- Get video 0 duration ---
    cap = cv2.VideoCapture(video_path_0)
    frame_count_0 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_0 = frame_count_0 / fps if fps > 0 else 0
    cap.release()

    # --- Extract frames if requested ---
    if extract_flag:
        logging.info(f"Extracting frames for all {num_videos} videos using parallel CPU extraction...")

        video_output_pairs = []
        for video_path in video_paths:
            video_dir = Path(video_path).parent
            output_dir = video_dir.parent / "images"
            output_dir.mkdir(parents=True, exist_ok=True)
            video_output_pairs.append((video_path, str(output_dir)))

        max_workers = min(len(video_paths), multiprocessing.cpu_count())
        logging.info(f"Using {max_workers} CPU workers")

        # Parallel extraction
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for video_path in executor.map(extract_frames_cpu, video_output_pairs):
                output_dir = Path(video_path).parent.parent / "images"
                extract_frame_data(str(output_dir), video_path)
    else:
        logging.info("Frame extraction is disabled.")

    # --- CSV paths ---
    csv_paths = [get_csv_path_from_video(vp) for vp in video_paths]
    csv_0 = csv_paths[0]
    right_csvs = csv_paths[1:]

    threshold_str_for_filename = format_threshold_filename(threshold_ns)
    matched_csv_path = os.path.join(base_output, f"matched_multi_{threshold_str_for_filename}.csv")
    matched_csv_full_path = os.path.join(base_output, f"matched_multi_{threshold_str_for_filename}_full.csv")

    # --- Multi-match ---
    match_frames_from_csv_multi(csv_0, right_csvs, matched_csv_path, threshold_ns)
    match_frames_full_from_csv_multi(csv_0, right_csvs, matched_csv_full_path, threshold_ns, duration_0)


if __name__ == "__main__":
    main()

