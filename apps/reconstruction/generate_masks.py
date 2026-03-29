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


def select_object_blob(mask, erode_sizes=(41, 31, 21), max_area_ratio=0.05,
                       max_aspect=3.0):
    """
    Isolate the object (compact blob on the table) from people/clutter.

    Strategy:
      1. Try several erosion strengths to break connections between the
         object and nearby clutter/person.
      2. Reject components that are too large, too elongated, too close to
         the border, or too high in the image.
      3. Pick the remaining component whose centroid is closest to the
         expected object region near the image center / lower half.
      4. Recover the object from a tight local window around that eroded
         component instead of globally growing the mask back.
    """
    h, w = mask.shape[:2]
    img_area = h * w

    ref_x, ref_y = w / 2, h * 0.65
    border_margin = int(min(h, w) * 0.03)

    for erode_size in erode_sizes:
        ek = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_size, erode_size),
        )
        eroded = cv2.erode(mask, ek, iterations=3)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            eroded, connectivity=8,
        )
        if num_labels <= 1:
            continue

        best_label = -1
        best_score = float('inf')
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < 120:
                continue
            if area > img_area * max_area_ratio:
                continue

            x = stats[lbl, cv2.CC_STAT_LEFT]
            y = stats[lbl, cv2.CC_STAT_TOP]
            bw = stats[lbl, cv2.CC_STAT_WIDTH]
            bh = stats[lbl, cv2.CC_STAT_HEIGHT]
            x2, y2 = x + bw, y + bh

            # Border-touching blobs are usually rig edges / stray reflections.
            if (x <= border_margin or y <= border_margin or
                    x2 >= w - border_margin or y2 >= h - border_margin):
                continue

            cx, cy = centroids[lbl]
            # The object sits on the platform, not at the very top of frame.
            if cy < h * 0.30:
                continue

            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
            if aspect > max_aspect:
                continue

            # Favor candidates near image center, with extra penalty for being
            # too high in the frame. This prevents cases like camera 09 where
            # a stray upper-right blob was selected instead of the box.
            dx = (cx - ref_x) / w
            dy = (cy - ref_y) / h
            score = dx * dx + 1.8 * dy * dy
            if score < best_score:
                best_score = score
                best_label = lbl

        if best_label < 0:
            continue

        selected_eroded = ((labels == best_label) * 255).astype(np.uint8)
        ys, xs = np.where(selected_eroded > 0)
        if len(xs) == 0:
            continue

        # Recover only a local neighborhood around the selected eroded blob.
        # This is more stable than a huge global grow-back and removes most
        # table spill in the problematic views.
        pad_x = max(int((xs.max() - xs.min() + 1) * 1.2), erode_size)
        pad_y = max(int((ys.max() - ys.min() + 1) * 1.2), erode_size)
        x0 = max(xs.min() - pad_x, 0)
        x1 = min(xs.max() + pad_x + 1, w)
        y0 = max(ys.min() - pad_y, 0)
        y1 = min(ys.max() + pad_y + 1, h)

        local = np.zeros_like(mask)
        local[y0:y1, x0:x1] = mask[y0:y1, x0:x1]

        num_local, local_labels, local_stats, _ = cv2.connectedComponentsWithStats(
            local, connectivity=8,
        )
        if num_local <= 1:
            return local

        overlap = cv2.dilate(
            selected_eroded,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (max(erode_size // 2, 3), max(erode_size // 2, 3)),
            ),
            iterations=1,
        )
        best_local = -1
        best_overlap = -1
        for lbl in range(1, num_local):
            comp = (local_labels == lbl).astype(np.uint8) * 255
            ov = cv2.countNonZero(cv2.bitwise_and(comp, overlap))
            if ov > best_overlap:
                best_overlap = ov
                best_local = lbl

        if best_local > 0 and best_overlap > 0:
            return ((local_labels == best_local) * 255).astype(np.uint8)

    return mask


def background_subtraction(data_root, cam_names, frame, bg_frame, ext,
                           threshold, morph_kernel, output_dir, vis_dir=None,
                           bg_data=None, center_only=False, dilate=0):
    """Simple frame-difference masking."""
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
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            filled = np.zeros_like(mask)
            cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
            mask = filled

        if center_only:
            mask = select_object_blob(mask)

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
                          help='Keep only foreground blobs that overlap the '
                               'center of the image (removes people/clutter at edges)')
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
            dilate=args.dilate,
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
