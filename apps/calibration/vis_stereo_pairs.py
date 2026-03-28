"""
Visualize matched ChArUco / chessboard point pairs between two cameras.

For each common frame, draws detected 2D keypoints on both images (color-coded
by corner ID), creates a side-by-side composite with connecting lines for
matched points, and saves to an output directory.

Usage:
  python apps/calibration/vis_stereo_pairs.py /mnt/yubo/charuco/board \
      --cam1 02 --cam2 03 --step 5

Works over SSH (OpenCV only, no GUI required).
"""
from __future__ import annotations

import argparse
import os
from glob import glob
from os.path import join

import cv2
import numpy as np

from easymocap.mytools import read_json


def id_color(corner_id: int, n_corners: int) -> tuple[int, int, int]:
    """Distinct BGR color for each corner ID via HSV colormap."""
    hue = int(180 * corner_id / max(n_corners, 1))
    hsv = np.uint8([[[hue, 255, 220]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_keypoints(img, k2d, n_corners, radius=6, font_scale=0.4):
    """Draw keypoints on image. k2d is Nx3 with confidence in column 2."""
    for i in range(k2d.shape[0]):
        conf = k2d[i, 2]
        x, y = int(round(k2d[i, 0])), int(round(k2d[i, 1]))
        if conf > 0:
            color = id_color(i, n_corners)
            cv2.circle(img, (x, y), radius, color, -1)
            cv2.putText(img, str(i), (x + radius + 2, y + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
        else:
            cv2.circle(img, (x, y), 2, (128, 128, 128), -1)


def draw_matches(canvas, k2d_left, k2d_right, w_left, n_corners, thickness=1):
    """Draw horizontal lines between matched points on side-by-side canvas."""
    for i in range(k2d_left.shape[0]):
        if k2d_left[i, 2] > 0 and k2d_right[i, 2] > 0:
            color = id_color(i, n_corners)
            x1 = int(round(k2d_left[i, 0]))
            y1 = int(round(k2d_left[i, 1]))
            x2 = int(round(k2d_right[i, 0])) + w_left
            y2 = int(round(k2d_right[i, 1]))
            cv2.line(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def sample_list(lst, step):
    if step <= 1:
        return lst
    return lst[::step]


def check_keypoints3d(k3d_prev, k3d_curr, name, cam1, cam2, tol=1e-5):
    """Warn if the 3D templates differ between cameras for the same frame."""
    if k3d_prev.shape != k3d_curr.shape:
        print(f'  [WARN] {name}: keypoints3d shape mismatch: '
              f'{cam1}={k3d_prev.shape} vs {cam2}={k3d_curr.shape}')
        return False
    diff = np.abs(k3d_prev - k3d_curr).max()
    if diff > tol:
        print(f'  [WARN] {name}: keypoints3d differ by {diff:.6f} '
              f'between {cam1} and {cam2}')
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Visualize stereo point pairs between two cameras')
    parser.add_argument('path', type=str, help='Dataset root')
    parser.add_argument('--cam1', type=str, required=True, help='First camera name (e.g. 02)')
    parser.add_argument('--cam2', type=str, required=True, help='Second camera name (e.g. 03)')
    parser.add_argument('--image', type=str, default='images')
    parser.add_argument('--ext', type=str, default='.jpg')
    parser.add_argument('--step', type=int, default=1, help='Frame subsampling step')
    parser.add_argument('--out', type=str, default=None,
                        help='Output dir (default: <path>/output/debug_stereo/<cam1>_<cam2>)')
    parser.add_argument('--max_frames', type=int, default=-1,
                        help='Max frames to visualize (-1 = all)')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='Resize scale for output images')
    args = parser.parse_args()

    cam1, cam2 = args.cam1, args.cam2
    if args.out is None:
        args.out = join(args.path, 'output', 'debug_stereo', f'{cam1}_{cam2}')
    os.makedirs(args.out, exist_ok=True)

    jsons_1 = sorted(glob(join(args.path, 'chessboard', cam1, '*.json')))
    jsons_2 = sorted(glob(join(args.path, 'chessboard', cam2, '*.json')))
    jsons_1 = sample_list(jsons_1, args.step)
    jsons_2 = sample_list(jsons_2, args.step)

    map_1 = {os.path.basename(p): p for p in jsons_1}
    map_2 = {os.path.basename(p): p for p in jsons_2}
    common = sorted(set(map_1.keys()) & set(map_2.keys()))

    if not common:
        print(f'No common chessboard files between {cam1} and {cam2}')
        return

    if args.max_frames > 0:
        common = common[:args.max_frames]

    print(f'[vis_stereo] {cam1} vs {cam2}: {len(common)} common frames (step={args.step})')

    k3d_mismatch_count = 0
    frame_stats = []

    for name in common:
        data_1 = read_json(map_1[name])
        data_2 = read_json(map_2[name])

        k3d_1 = np.array(data_1['keypoints3d'], np.float32)
        k2d_1 = np.array(data_1['keypoints2d'], np.float32)
        k3d_2 = np.array(data_2['keypoints3d'], np.float32)
        k2d_2 = np.array(data_2['keypoints2d'], np.float32)
        n_corners = k2d_1.shape[0]

        if not check_keypoints3d(k3d_1, k3d_2, name, cam1, cam2):
            k3d_mismatch_count += 1

        valid_1 = k2d_1[:, 2] > 0
        valid_2 = k2d_2[:, 2] > 0
        valid_both = valid_1 & valid_2
        n_valid = int(valid_both.sum())

        stem = os.path.splitext(name)[0]
        img_path_1 = join(args.path, args.image, cam1, stem + args.ext)
        img_path_2 = join(args.path, args.image, cam2, stem + args.ext)

        img1 = cv2.imread(img_path_1)
        img2 = cv2.imread(img_path_2)
        if img1 is None or img2 is None:
            print(f'  [{name}] Cannot read images, skipping')
            continue

        draw_keypoints(img1, k2d_1, n_corners)
        draw_keypoints(img2, k2d_2, n_corners)

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h = max(h1, h2)
        canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = img1
        canvas[:h2, w1:] = img2

        draw_matches(canvas, k2d_1, k2d_2, w1, n_corners)

        info_text = (f'{cam1} vs {cam2} | {name} | '
                     f'valid: {cam1}={int(valid_1.sum())}, {cam2}={int(valid_2.sum())}, '
                     f'both={n_valid}')
        cv2.putText(canvas, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        if args.scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=args.scale, fy=args.scale,
                                interpolation=cv2.INTER_AREA)

        outname = join(args.out, f'{stem}.jpg')
        cv2.imwrite(outname, canvas)
        frame_stats.append((name, n_valid, int(valid_1.sum()), int(valid_2.sum())))
        print(f'  [{name}] valid_both={n_valid} (cam1={int(valid_1.sum())}, cam2={int(valid_2.sum())})')

    print(f'\n[vis_stereo] Saved {len(frame_stats)} images to {args.out}')
    if k3d_mismatch_count > 0:
        print(f'[vis_stereo] WARNING: keypoints3d mismatch in {k3d_mismatch_count} frames!')

    valid_counts = [s[1] for s in frame_stats]
    if valid_counts:
        print(f'[vis_stereo] Matched points per frame: '
              f'min={min(valid_counts)}, max={max(valid_counts)}, '
              f'mean={np.mean(valid_counts):.1f}')
    low_frames = [(s[0], s[1]) for s in frame_stats if s[1] < 6]
    if low_frames:
        print(f'[vis_stereo] {len(low_frames)} frames with <6 matched points '
              f'(would be skipped by stereoCalibrate):')
        for fname, cnt in low_frames:
            print(f'    {fname}: {cnt} points')


if __name__ == '__main__':
    main()
