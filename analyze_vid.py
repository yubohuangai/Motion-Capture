import cv2
import os
import subprocess
from datetime import datetime
from pathlib import Path
import re
import shutil
import yaml


CONFIG = dict(
    base_root=r"C:\Users\yuboh\GitHub\data\battery1223",
    log_file=r"output\video_analysis.log",
    start_cam=1,
    end_cam=2,   # inclusive
    truncate=False,
    truncate_from="begin",
    clear_log=False,
)


def ensure_file_exists(file_path):
    file_path = Path(file_path)
    if file_path.parent:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    if not file_path.exists():
        file_path.touch()


def load_video_path_from_config(config_path="config.yaml", key="video_path_1"):
    """Load a specific video path from YAML config."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if key not in cfg:
        raise KeyError(f"'{key}' not found in {config_path}")
    return cfg[key]


def get_csv_path_from_video(video_path):
    video_name = Path(video_path).stem
    match = re.search(r"VID_((\d|_)+)", video_name)
    if not match:
        return None
    video_date = match.group(1)
    csv_path = Path(video_path).parent.parent / f"{video_date}.csv"
    return csv_path

def find_video_and_csv(root_path):
    """
    Input:
        root_path = C:\...\raw\09

    Automatically finds:
        video_path = root_path / VID / *.mp4   (must contain exactly one)
        csv_path   = root_path / <timestamp>.csv
    """
    root = Path(root_path)
    vid_dir = root / "VID"

    # Find all .mp4 files
    videos = list(vid_dir.glob("*.mp4"))
    if len(videos) == 0:
        return None, None  # instead of raising error
    if len(videos) > 1:
        raise RuntimeError(f"Multiple .mp4 files found in {vid_dir}, expected only one.")

    video_path = videos[0]
    video_name = video_path.stem  # VID_20251116_121905

    # Extract timestamp
    match = re.search(r"VID_(\d{8}_\d{6})", video_name)
    if not match:
        raise ValueError(f"Filename does not contain timestamp: {video_name}")

    timestamp = match.group(1)

    # CSV should sit in the root directory
    csv_path = root / f"{timestamp}.csv"

    return str(video_path), str(csv_path)


def count_csv_lines(csv_path):
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def get_frame_count_ffmpeg(video_path):
    """Get reliable frame count using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return int(result.stdout.strip())
    except:
        return None


def get_frame_count_fast(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return int(result.stdout.strip())
    except:
        return None


def analyze_video(video_path: str, log_file: str, truncate=False, truncate_from="end"):
    """
    Analyze video & optionally truncate CSV if timestamp_count > frame_count.
    truncate_from: "end" (default) → cut extra lines at the end,
                   "begin" → cut extra lines at the beginning.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    csv_path = get_csv_path_from_video(video_path)
    timestamp_count = None
    if csv_path is None:
        csv_msg = "Video name format did not match expected pattern. No CSV checked."
    else:
        timestamp_count = count_csv_lines(csv_path)
        if timestamp_count is None:
            csv_msg = f"CSV not found: {csv_path}"
        else:
            csv_msg = f"CSV path  : {csv_path} (lines: {timestamp_count})"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n============= Video Analysis: {datetime.now()} =============\n")
        f.write(f"Video path: {video_path}\n")
        f.write(f"{csv_msg}\n")

        # --- OpenCV properties ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            f.write(f"Failed to open video with OpenCV: {video_path}\n")
            frame_count = None
        else:
            frame_count_ffmpeg = get_frame_count_fast(video_path)
            frame_count = frame_count_ffmpeg or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0

            f.write("OpenCV video properties:\n")
            if timestamp_count is not None:
                f.write(f"Timestamp count  : {timestamp_count}\n")
            f.write(f"Frame count      : {frame_count}\n")
            f.write(f"Frame rate (FPS) : {fps:.2f}\n")
            f.write(f"Duration (sec)   : {duration:.2f}\n")

        cap.release()

        # --- Truncate CSV if needed ---
        if truncate and csv_path and timestamp_count and frame_count:
            if timestamp_count > frame_count:
                backup_csv = csv_path.parent / f"{csv_path.stem}_ori{csv_path.suffix}"
                shutil.copy(csv_path, backup_csv)
                f.write(f"[TRUNCATE] CSV backed up to {backup_csv}\n")

                with open(csv_path, "r", encoding="utf-8") as infile:
                    lines = infile.readlines()

                if truncate_from == "end":
                    new_lines = lines[:frame_count]
                elif truncate_from == "begin":
                    new_lines = lines[-frame_count:]
                else:
                    raise ValueError(f"Invalid truncate_from='{truncate_from}', must be 'begin' or 'end'.")

                with open(csv_path, "w", encoding="utf-8") as outfile:
                    outfile.writelines(new_lines)

                f.write(f"[TRUNCATE] CSV truncated ({truncate_from}) from {timestamp_count} → {frame_count} lines\n")

        # --- FFmpeg probe ---
        try:
            ffmpeg_cmd = ["ffmpeg", "-i", video_path]
            result = subprocess.run(ffmpeg_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            # f.write("FFmpeg output:\n")
            # f.write(result.stderr)
        except FileNotFoundError:
            f.write("FFmpeg not found. Please ensure it is installed and in PATH.\n")



if __name__ == "__main__":
    base_root = Path(CONFIG["base_root"])
    log_file = CONFIG["log_file"]

    ensure_file_exists(log_file)

    if CONFIG["clear_log"]:
        open(log_file, "w", encoding="utf-8").close()

    for i in range(CONFIG["start_cam"], CONFIG["end_cam"] + 1):
        root_path = base_root / f"{i:02d}"
        print(f"\n--- Processing {root_path} ---")

        try:
            video_path, _ = find_video_and_csv(root_path)
            if video_path is None:
                print(f"Skipping {root_path} (no video found)")
                continue

            analyze_video(
                video_path,
                log_file,
                truncate=CONFIG["truncate"],
                truncate_from=CONFIG["truncate_from"]
            )
            print(f"✓ Finished: {video_path}")

        except Exception as e:
            print(f"✗ Error in {root_path}: {e}")

