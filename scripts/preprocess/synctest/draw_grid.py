from pathlib import Path

import cv2

from session_paths import get_reference_camera_index, load_config, video_path_at


def load_reference_video(config_path):
    config = load_config(config_path)
    idx = get_reference_camera_index(config)
    return video_path_at(config, idx)


def draw_grid(image, step=100, color=(0, 255, 0), thickness=1, label=True):
    height, width = image.shape[:2]

    for x in range(0, width, step):
        cv2.line(image, (x, 0), (x, height), color, thickness)
        if label:
            cv2.putText(image, f'{x}', (x + 5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)

    for y in range(0, height, step):
        cv2.line(image, (0, y), (width, y), color, thickness)
        if label:
            cv2.putText(image, f'{y}', (5, y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)

    return image


def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Failed to read the first frame of the video: {video_path}")
    return frame


def extract_last_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    # Go to the last frame (index = total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Failed to read the last frame of the video: {video_path}")
    return frame


def resize_to_fit_screen(image, max_width=1920, max_height=1080):
    height, width = image.shape[:2]
    if width <= max_width and height <= max_height:
        return image  # No need to resize

    scale = min(max_width / width, max_height / height)
    new_size = (int(width * scale), int(height * scale))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized


def main(config_path, use_last_frame=False):
    path0 = load_reference_video(config_path)

    print(f"Processing {path0}")
    if use_last_frame:
        frame = extract_last_frame(path0)
    else:
        frame = extract_first_frame(path0)

    grid_frame = draw_grid(frame.copy(), step=100)
    resized_grid = resize_to_fit_screen(grid_frame)

    window_name = "Grid - reference camera"
    cv2.imshow(window_name, resized_grid)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    config_file = Path(__file__).resolve().parent / "config.yaml"
    main(config_file, use_last_frame=True)  # Set to True for last frame
