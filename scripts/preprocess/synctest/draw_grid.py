from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml

from session_paths import get_reference_camera_index, load_config, video_path_at


def load_reference_video(config_path):
    config = load_config(config_path)
    idx = get_reference_camera_index(config)
    return video_path_at(config, idx)


def _put_text_readable(
    image,
    text,
    org,
    font_scale,
    color_bgr,
    thickness,
):
    """Dark outline + fill so coordinates stay readable on light frames."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    outline = thickness + 2
    cv2.putText(image, text, org, font, font_scale, (0, 0, 0), outline, cv2.LINE_AA)
    cv2.putText(image, text, org, font, font_scale, color_bgr, thickness, cv2.LINE_AA)


def draw_grid(
    image,
    step=100,
    line_color=(70, 150, 70),
    line_thickness=1,
    label=True,
    label_color=(0, 80, 230),
    label_scale=1.15,
    label_thickness=2,
):
    """
    line_color / label_color: BGR. Defaults tuned for light backgrounds:
    darker green grid lines, orange-blue labels with black outline.
    """
    height, width = image.shape[:2]

    for x in range(0, width, step):
        cv2.line(image, (x, 0), (x, height), line_color, line_thickness)
        if label:
            _put_text_readable(
                image,
                f"{x}",
                (x + 5, 28),
                label_scale,
                label_color,
                label_thickness,
            )

    for y in range(0, height, step):
        cv2.line(image, (0, y), (width, y), line_color, line_thickness)
        if label:
            _put_text_readable(
                image,
                f"{y}",
                (5, y + 28),
                label_scale,
                label_color,
                label_thickness,
            )

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


def preview_max_dimensions():
    """Scale previews to most of the primary monitor (avoids tiny windows on large displays)."""
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return max(800, int(w * 0.92)), max(600, int(h * 0.92))
    except Exception:
        return 2560, 1440


def resize_to_fit_screen(image, max_width=1920, max_height=1080):
    height, width = image.shape[:2]
    if width <= max_width and height <= max_height:
        return image  # No need to resize

    scale = min(max_width / width, max_height / height)
    new_size = (int(width * scale), int(height * scale))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized


def _run_preview_loop(window_name: str) -> None:
    """Exit on q, Esc, or closing the window (title bar X)."""
    while True:
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q") or k == 27:
            break
        try:
            prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            break
        if prop < 0:
            break
    cv2.destroyAllWindows()


def _display_to_original_xy(
    xd: int, yd: int, orig_h: int, orig_w: int, disp_h: int, disp_w: int
) -> tuple[int, int]:
    """Map pixel from displayed (possibly scaled) image back to full-resolution frame coords."""
    xo = int(round(xd * orig_w / max(disp_w, 1)))
    yo = int(round(yd * orig_h / max(disp_h, 1)))
    xo = max(0, min(orig_w - 1, xo))
    yo = max(0, min(orig_h - 1, yo))
    return xo, yo


def _crop_yaml_block(x_left: int, x_right: int, y_top: int, y_bottom: int) -> str:
    return (
        "crop_region:\n"
        f"  x_left: {x_left}\n"
        f"  x_right: {x_right}\n"
        f"  y_top: {y_top}\n"
        f"  y_bottom: {y_bottom}\n"
    )


def interactive_pick_crop(
    config_path: Path,
    *,
    use_last_frame: bool,
    write_config: bool,
    min_drag_px: int = 5,
) -> None:
    """
    Show gridded frame; drag an axis-aligned rectangle. On mouse release, print crop_region
    (inclusive pixel indices, same as crop_video.py). Optional --write-config updates config.yaml.
    """
    path0 = load_reference_video(config_path)
    print(f"Video: {path0}")
    if use_last_frame:
        frame = extract_last_frame(path0)
    else:
        frame = extract_first_frame(path0)

    grid_frame = draw_grid(frame.copy(), step=100)
    mw, mh = preview_max_dimensions()
    display = resize_to_fit_screen(grid_frame, mw, mh)
    orig_h, orig_w = grid_frame.shape[:2]
    disp_h, disp_w = display.shape[:2]

    window_name = "Grid — drag crop box (release to apply)"
    state: dict = {
        "drawing": False,
        "x0": 0,
        "y0": 0,
        "x1": 0,
        "y1": 0,
    }

    def draw_overlay(vis, x0, y1, x2, y2):
        x_min, x_max = sorted((x0, x2))
        y_min, y_max = sorted((y1, y2))
        cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["x0"] = state["x1"] = x
            state["y0"] = state["y1"] = y
        elif event == cv2.EVENT_MOUSEMOVE and state.get("drawing"):
            state["x1"], state["y1"] = x, y
            vis = display.copy()
            draw_overlay(vis, state["x0"], state["y0"], state["x1"], state["y1"])
            cv2.imshow(window_name, vis)
        elif event == cv2.EVENT_LBUTTONUP and state.get("drawing"):
            state["drawing"] = False
            state["x1"], state["y1"] = x, y
            dx = abs(state["x1"] - state["x0"])
            dy = abs(state["y1"] - state["y0"])
            if dx < min_drag_px or dy < min_drag_px:
                cv2.imshow(window_name, display)
                print("(Ignored tiny drag — try a larger box.)")
                return
            x_min_d = min(state["x0"], state["x1"])
            x_max_d = max(state["x0"], state["x1"])
            y_min_d = min(state["y0"], state["y1"])
            y_max_d = max(state["y0"], state["y1"])
            x_left, y_top = _display_to_original_xy(
                x_min_d, y_min_d, orig_h, orig_w, disp_h, disp_w
            )
            x_right, y_bottom = _display_to_original_xy(
                x_max_d, y_max_d, orig_h, orig_w, disp_h, disp_w
            )
            if x_right < x_left:
                x_left, x_right = x_right, x_left
            if y_bottom < y_top:
                y_top, y_bottom = y_bottom, y_top

            block = _crop_yaml_block(x_left, x_right, y_top, y_bottom)
            print("\n--- Paste under # crop_video.py in config.yaml ---\n")
            print(block, end="")
            print("---\n")

            if write_config:
                with config_path.open("r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                cfg["crop_region"] = {
                    "x_left": x_left,
                    "x_right": x_right,
                    "y_top": y_top,
                    "y_bottom": y_bottom,
                }
                with config_path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(
                        cfg,
                        f,
                        sort_keys=False,
                        default_flow_style=False,
                        allow_unicode=True,
                    )
                print(f"Updated {config_path} (YAML rewrite; comments above may need restoring.)\n")

            vis = display.copy()
            draw_overlay(vis, state["x0"], state["y0"], state["x1"], state["y1"])
            cv2.putText(
                vis,
                "Saved — drag again to redo | q / Esc / close window",
                (10, disp_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, vis)

    # AUTOSIZE keeps image pixels 1:1 with the window (no stretch from manual resize).
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)
    cv2.imshow(window_name, display)
    hint = display.copy()
    cv2.putText(
        hint,
        "Drag crop box (release = print YAML) | q / Esc / close window",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 200, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow(window_name, hint)

    _run_preview_loop(window_name)


def main(config_path, use_last_frame=False):
    path0 = load_reference_video(config_path)

    print(f"Processing {path0}")
    if use_last_frame:
        frame = extract_last_frame(path0)
    else:
        frame = extract_first_frame(path0)

    grid_frame = draw_grid(frame.copy(), step=100)
    mw, mh = preview_max_dimensions()
    resized_grid = resize_to_fit_screen(grid_frame, mw, mh)

    window_name = "Grid - reference camera"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, resized_grid)

    _run_preview_loop(window_name)


if __name__ == "__main__":
    config_file = Path(__file__).resolve().parent / "config.yaml"
    parser = argparse.ArgumentParser(
        description="Pick crop_region on the gridded reference frame (default), or only show the grid (--no-pick-crop)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=config_file,
        help="Path to synctest config.yaml",
    )
    parser.set_defaults(pick_crop=True)
    parser.add_argument(
        "--pick-crop",
        dest="pick_crop",
        action="store_true",
        help="Interactive crop picking (default).",
    )
    parser.add_argument(
        "--no-pick-crop",
        dest="pick_crop",
        action="store_false",
        help="Only show the gridded frame (no drag rectangle / YAML output).",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Write crop_region into --config (full-file YAML dump; comments may be lost).",
    )
    parser.add_argument(
        "--first-frame",
        action="store_true",
        help="Use first video frame (default: last frame, same as before).",
    )
    args = parser.parse_args()
    use_last = not args.first_frame

    if args.pick_crop:
        interactive_pick_crop(
            args.config,
            use_last_frame=use_last,
            write_config=args.write_config,
        )
    else:
        if args.write_config:
            parser.error("--write-config only applies with interactive crop (omit --no-pick-crop).")
        main(args.config, use_last_frame=use_last)
