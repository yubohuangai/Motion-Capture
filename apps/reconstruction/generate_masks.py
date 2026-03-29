"""
Generate foreground masks for multi-view object reconstruction.

Two modes:
  1. **background subtraction** (default): requires a background-only frame.
     Computes per-pixel difference, thresholds, and cleans with morphology.
  2. **SAM** (Segment Anything Model): uses a bounding-box or point prompt to
     segment the foreground in each view.  Requires `segment_anything` to be
     installed.

Output layout mirrors the image directory:
    <data>/masks/<cam>/NNNNNN.png   (255 = foreground, 0 = background)

Usage:
    # Background subtraction (fast, works well in controlled studios)
    python apps/reconstruction/generate_masks.py /path/to/data \\
        --frame 0 --bg_frame 100 --threshold 30

    # SAM-based segmentation (needs segment_anything + model checkpoint)
    python apps/reconstruction/generate_masks.py /path/to/data \\
        --frame 0 --mode sam --sam_checkpoint /path/to/sam_vit_h.pth
"""

import argparse
import os
import sys
from glob import glob
from os.path import join

import cv2
import numpy as np

sys.path.insert(0, join(os.path.dirname(__file__), '..', '..'))


def get_cam_names(data_root):
    img_dir = join(data_root, 'images')
    cams = sorted([
        d for d in os.listdir(img_dir)
        if os.path.isdir(join(img_dir, d))
    ])
    return cams


def find_image(data_root, cam, frame, ext):
    path = join(data_root, 'images', cam, f'{frame:06d}{ext}')
    if os.path.exists(path):
        return path
    candidates = sorted(glob(join(data_root, 'images', cam, f'*{ext}')))
    if len(candidates) == 0:
        raise FileNotFoundError(f'No images for camera {cam}')
    if frame < len(candidates):
        return candidates[frame]
    return candidates[0]


def save_mask_overlay(img_bgr, mask, vis_dir, cam, frame):
    """Save a side-by-side image: original | mask overlay (green tint on foreground)."""
    os.makedirs(vis_dir, exist_ok=True)
    overlay = img_bgr.copy()
    green = np.zeros_like(img_bgr)
    green[:, :, 1] = 255
    overlay[mask > 0] = cv2.addWeighted(overlay, 0.5, green, 0.5, 0)[mask > 0]

    # draw contour outline in red
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

    combined = np.hstack([img_bgr, overlay])
    out_path = join(vis_dir, f'{cam}_{frame:06d}.jpg')
    cv2.imwrite(out_path, combined)
    return out_path


def pick_object_blob(mask, max_area_ratio=0.10, min_area_ratio=0.002,
                     bottom_ratio=0.7):
    """
    Pick the single contour most likely to be the object on the table.

    Strategy (avoids fusing person + object before selection):
      1. Only consider blobs whose centroid is in the bottom portion of the
         image (the object sits on a table, not floating at the top).
      2. Reject blobs smaller than min_area or larger than max_area.
      3. Among survivors, pick the one with the largest area (= the object,
         not a small noise speck).

    The caller should run this on a mask that has NOT been morphologically
    closed, so the person and object stay as separate contours.
    """
    h, w = mask.shape[:2]
    img_area = h * w
    min_area = img_area * min_area_ratio
    max_area = img_area * max_area_ratio
    y_cutoff = h * (1.0 - bottom_ratio)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        blob_cy = y + bh / 2.0
        if blob_cy < y_cutoff:
            continue
        candidates.append((area, cnt))

    result = np.zeros_like(mask)
    if candidates:
        candidates.sort(key=lambda t: t[0], reverse=True)
        cv2.drawContours(result, [candidates[0][1]], -1, 255, cv2.FILLED)
    return result


def background_subtraction(data_root, cam_names, frame, bg_frame, ext,
                           threshold, morph_kernel, output_dir, vis_dir=None,
                           bg_data=None, center_only=False, dilate=0,
                           max_area_ratio=0.10):
    """Frame-difference masking with optional smart blob selection."""
    bg_root = bg_data or data_root
    for cam in cam_names:
        fg_path = find_image(data_root, cam, frame, ext)
        bg_path = find_image(bg_root, cam, bg_frame, ext)

        fg = cv2.imread(fg_path).astype(np.float32)
        bg = cv2.imread(bg_path).astype(np.float32)

        diff = np.abs(fg - bg).max(axis=2)
        mask = (diff > threshold).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel),
        )

        if center_only:
            # OPEN only -- removes small noise but does NOT bridge nearby
            # regions, so the person and the object stay separate contours.
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
            mask = pick_object_blob(mask, max_area_ratio=max_area_ratio)
            # Now close + fill the selected blob to get a solid mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        else:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                filled = np.zeros_like(mask)
                cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
                mask = filled

        if dilate > 0:
            dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
            mask = cv2.dilate(mask, dk, iterations=1)

        out_dir = join(output_dir, cam)
        os.makedirs(out_dir, exist_ok=True)
        out_path = join(out_dir, f'{frame:06d}.png')
        cv2.imwrite(out_path, mask)
        print(f'  {cam}: {out_path}  (fg pixels: {mask.sum() // 255})')

        if vis_dir is not None:
            img_bgr = cv2.imread(fg_path)
            vis_path = save_mask_overlay(img_bgr, mask, vis_dir, cam, frame)
            print(f'    vis: {vis_path}')


def sam_segmentation(data_root, cam_names, frame, ext, output_dir,
                     sam_checkpoint, sam_model_type, vis_dir=None):
    """Use Segment Anything to produce masks from auto-detected boxes."""
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError:
        print('ERROR: segment_anything is not installed.')
        print('  pip install segment-anything')
        print('  Download checkpoint from https://github.com/facebookresearch/segment-anything')
        sys.exit(1)

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'[SAM] Loading model {sam_model_type} from {sam_checkpoint} ...')
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=1000,
    )

    for cam in cam_names:
        img_path = find_image(data_root, cam, frame, ext)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        masks = mask_generator.generate(img_rgb)

        combined = np.zeros(img.shape[:2], dtype=np.uint8)
        for m in masks:
            combined[m['segmentation']] = 255

        out_dir = join(output_dir, cam)
        os.makedirs(out_dir, exist_ok=True)
        out_path = join(out_dir, f'{frame:06d}.png')
        cv2.imwrite(out_path, combined)
        print(f'  {cam}: {out_path}  ({len(masks)} segments)')

        if vis_dir is not None:
            vis_path = save_mask_overlay(img, combined, vis_dir, cam, frame)
            print(f'    vis: {vis_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Generate foreground masks for multi-view reconstruction',
    )
    parser.add_argument('data', help='Root data path (with images/<cam>/)')
    parser.add_argument('--frame', type=int, default=0, help='Target frame index')
    parser.add_argument('--ext', default='.jpg', help='Image extension')
    parser.add_argument('--output', default=None,
                        help='Output mask directory (default: <data>/masks)')
    parser.add_argument('--mode', choices=['bg_sub', 'sam'], default='bg_sub',
                        help='Mask generation mode')

    bg_group = parser.add_argument_group('background subtraction')
    bg_group.add_argument('--bg_frame', type=int, default=None,
                          help='Background-only frame index (required for bg_sub)')
    bg_group.add_argument('--bg_data', default=None,
                          help='Separate data root for background frames '
                               '(if background is in a different directory)')
    bg_group.add_argument('--threshold', type=float, default=30.0,
                          help='Pixel difference threshold (0-255)')
    bg_group.add_argument('--morph_kernel', type=int, default=7,
                          help='Morphology kernel size')
    bg_group.add_argument('--center_only', action='store_true',
                          help='Keep only the single best foreground blob near '
                               'the image center (removes people/clutter)')
    bg_group.add_argument('--max_area_ratio', type=float, default=0.15,
                          help='Max blob area as fraction of image (blobs '
                               'larger than this are discarded, default 0.15)')
    bg_group.add_argument('--dilate', type=int, default=0,
                          help='Dilate final mask by N pixels (adds margin around object)')

    sam_group = parser.add_argument_group('SAM segmentation')
    sam_group.add_argument('--sam_checkpoint', default=None,
                           help='Path to SAM model checkpoint')
    sam_group.add_argument('--sam_model_type', default='vit_h',
                           help='SAM model type (vit_h, vit_l, vit_b)')

    parser.add_argument('--vis', action='store_true',
                        help='Save mask overlay visualizations to mask_vis/')

    args = parser.parse_args()

    output_dir = args.output or join(args.data, 'masks')
    vis_dir = join(args.data, 'mask_vis') if args.vis else None
    cam_names = get_cam_names(args.data)
    print(f'[generate_masks] Cameras: {cam_names}')
    print(f'[generate_masks] Mode: {args.mode}, frame: {args.frame}')
    print(f'[generate_masks] Output: {output_dir}')
    if vis_dir:
        print(f'[generate_masks] Vis overlays: {vis_dir}')

    if args.mode == 'bg_sub':
        if args.bg_frame is None:
            parser.error('--bg_frame is required for background subtraction mode')
        background_subtraction(
            args.data, cam_names, args.frame, args.bg_frame, args.ext,
            args.threshold, args.morph_kernel, output_dir, vis_dir,
            bg_data=args.bg_data, center_only=args.center_only,
            dilate=args.dilate, max_area_ratio=args.max_area_ratio,
        )
    elif args.mode == 'sam':
        if args.sam_checkpoint is None:
            parser.error('--sam_checkpoint is required for SAM mode')
        sam_segmentation(
            args.data, cam_names, args.frame, args.ext, output_dir,
            args.sam_checkpoint, args.sam_model_type, vis_dir,
        )

    print('[generate_masks] Done.')


if __name__ == '__main__':
    main()
