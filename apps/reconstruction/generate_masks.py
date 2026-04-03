"""
Generate foreground masks for multi-view object reconstruction.

Modes:
  1. **background subtraction** (default): requires a background-only frame.
     Computes per-pixel difference, thresholds, and cleans with morphology.
  2. **SAM**: `SamAutomaticMaskGenerator` over the full image (union of masks).
     Requires `segment_anything` to be installed.
  3. **hybrid**: background subtraction to estimate a tight box, then SAM
     (`SamPredictor` box prompt) for a cleaner boundary—good when diff masks
     include extra table/background.  Use `--hybrid_combine intersect` to trim
     to the overlap of both masks.

Output layout mirrors the image directory:
    <data>/masks/<cam>/NNNNNN.png   (255 = foreground, 0 = background)

Usage:
    # Background subtraction (fast, works well in controlled studios)
    python apps/reconstruction/generate_masks.py /path/to/data \\
        --frame 0 --bg_frame 100 --threshold 30

    # SAM automatic masks (needs segment_anything + model checkpoint)
    python apps/reconstruction/generate_masks.py /path/to/data \\
        --frame 0 --mode sam --sam_checkpoint /path/to/sam_vit_h.pth

    # Hybrid: bg_sub box + SAM box prompt (object in one root, bg in another)
    python apps/reconstruction/generate_masks.py /mnt/yubo/obj/cube \\
        --bg_data /mnt/yubo/obj/background --frame 0 --bg_frame 0 \\
        --mode hybrid --sam_checkpoint /path/to/sam_vit_h.pth

    # Shrink a slightly bloated diff mask (no SAM)
    python apps/reconstruction/generate_masks.py /path/to/data \\
        --frame 0 --bg_frame 100 --erode 3
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


def locate_object_bbox(mask, max_area_ratio=0.10, min_area_ratio=0.0005,
                       bottom_ratio=0.75):
    """
    Find the bounding box of the object blob on the table.

    Works on a lightly-cleaned mask (opening only, no closing) so the person
    and object stay as separate contours.  Returns (x, y, w, h) of the best
    candidate or None.
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
        bx, by, bw, bh = cv2.boundingRect(cnt)
        blob_cy = by + bh / 2.0
        if blob_cy < y_cutoff:
            continue
        candidates.append((area, (bx, by, bw, bh)))

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def compute_background_subtraction_mask(fg_path, bg_path, threshold, morph_kernel,
                                        center_only, max_area_ratio):
    """Return uint8 mask (0 / 255) from frame differencing; no dilate/erode."""
    fg = cv2.imread(fg_path)
    bg = cv2.imread(bg_path)
    if fg is None:
        raise FileNotFoundError(f'Cannot read foreground: {fg_path}')
    if bg is None:
        raise FileNotFoundError(f'Cannot read background: {bg_path}')
    fg_f = fg.astype(np.float32)
    bg_f = bg.astype(np.float32)

    diff = np.abs(fg_f - bg_f).max(axis=2)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel),
    )

    if center_only:
        # --- Pass 1: locate the object with conservative settings ---
        coarse = (diff > threshold).astype(np.uint8) * 255
        small_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        coarse = cv2.morphologyEx(coarse, cv2.MORPH_OPEN, small_k,
                                  iterations=1)
        bbox = locate_object_bbox(coarse, max_area_ratio=max_area_ratio)

        if bbox is not None:
            bx, by, bw, bh = bbox
            h, w = diff.shape[:2]
            # tight bbox: 10% padding (dilate adds margin later)
            pad_x, pad_y = int(bw * 0.1), int(bh * 0.1)
            x1 = max(0, bx - pad_x)
            y1 = max(0, by - pad_y)
            x2 = min(w, bx + bw + pad_x)
            y2 = min(h, by + bh + pad_y)

            # --- Pass 2: same threshold but restricted to bbox ---
            mask = np.zeros((h, w), dtype=np.uint8)
            roi = diff[y1:y2, x1:x2]
            mask[y1:y2, x1:x2] = (roi > threshold).astype(np.uint8) * 255

            # close to fill gaps within the box, open to remove speckles
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                                    iterations=3)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_k,
                                    iterations=1)

            # keep only the largest contour, use convex hull for clean shape
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                biggest = max(cnts, key=cv2.contourArea)
                hull = cv2.convexHull(biggest)
                mask = np.zeros_like(mask)
                cv2.drawContours(mask, [hull], -1, 255, cv2.FILLED)
        else:
            mask = np.zeros(diff.shape[:2], dtype=np.uint8)
    else:
        mask = (diff > threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            filled = np.zeros_like(mask)
            cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
            mask = filled

    return mask


def apply_mask_postprocess(mask, dilate=0, erode=0):
    """Optional dilate then erode on binary uint8 mask."""
    out = mask
    if dilate > 0:
        dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        out = cv2.dilate(out, dk, iterations=1)
    if erode > 0:
        ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode))
        out = cv2.erode(out, ek, iterations=1)
    return out


def bbox_from_mask(mask, pad_ratio=0.0):
    """
    Tight axis-aligned box around the largest foreground contour.
    pad_ratio expands each side by a fraction of (w, h).  Returns xyxy numpy
    array or None if empty.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    bx, by, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    h, w = mask.shape[:2]
    if bw <= 0 or bh <= 0:
        return None
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    x1 = max(0, bx - pad_x)
    y1 = max(0, by - pad_y)
    x2 = min(w, bx + bw + pad_x)
    y2 = min(h, by + bh + pad_y)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def background_subtraction(data_root, cam_names, frame, bg_frame, ext,
                           threshold, morph_kernel, output_dir, vis_dir=None,
                           bg_data=None, center_only=False, dilate=0, erode=0,
                           max_area_ratio=0.10, save_fg_dir=None):
    """Frame-difference masking with optional smart blob selection.

    When --center_only is set, uses a two-pass strategy:
      Pass 1 (locate): light opening on high-threshold diff → find object bbox
      Pass 2 (refine): same threshold inside tight bbox → largest blob only
    """
    bg_root = bg_data or data_root
    for cam in cam_names:
        fg_path = find_image(data_root, cam, frame, ext)
        bg_path = find_image(bg_root, cam, bg_frame, ext)

        mask = compute_background_subtraction_mask(
            fg_path, bg_path, threshold, morph_kernel, center_only,
            max_area_ratio,
        )
        if center_only and mask.sum() == 0:
            print(f'  {cam}: WARNING - no object blob found')

        mask = apply_mask_postprocess(mask, dilate=dilate, erode=erode)

        out_dir = join(output_dir, cam)
        os.makedirs(out_dir, exist_ok=True)
        out_path = join(out_dir, f'{frame:06d}.png')
        cv2.imwrite(out_path, mask)
        print(f'  {cam}: {out_path}  (fg pixels: {mask.sum() // 255})')

        if save_fg_dir is not None:
            img_bgr = cv2.imread(fg_path)
            fg_img = img_bgr.copy()
            fg_img[mask == 0] = 0
            out_fg_dir = join(save_fg_dir, cam)
            os.makedirs(out_fg_dir, exist_ok=True)
            out_fg_path = join(out_fg_dir, f'{frame:06d}{ext}')
            cv2.imwrite(out_fg_path, fg_img)
            print(f'    fg: {out_fg_path}')

        if vis_dir is not None:
            img_bgr = cv2.imread(fg_path)
            vis_path = save_mask_overlay(img_bgr, mask, vis_dir, cam, frame)
            print(f'    vis: {vis_path}')


def sam_segmentation(data_root, cam_names, frame, ext, output_dir,
                     sam_checkpoint, sam_model_type, vis_dir=None,
                     save_fg_dir=None):
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

        if save_fg_dir is not None:
            fg_img = img.copy()
            fg_img[combined == 0] = 0
            out_fg_dir = join(save_fg_dir, cam)
            os.makedirs(out_fg_dir, exist_ok=True)
            out_fg_path = join(out_fg_dir, f'{frame:06d}{ext}')
            cv2.imwrite(out_fg_path, fg_img)
            print(f'    fg: {out_fg_path}')

        if vis_dir is not None:
            vis_path = save_mask_overlay(img, combined, vis_dir, cam, frame)
            print(f'    vis: {vis_path}')


def hybrid_bg_sam(data_root, cam_names, frame, bg_frame, ext,
                  threshold, morph_kernel, output_dir, vis_dir=None,
                  bg_data=None, center_only=False, dilate=0, erode=0,
                  max_area_ratio=0.10, save_fg_dir=None,
                  sam_checkpoint=None, sam_model_type='vit_h',
                  hybrid_combine='sam', sam_box_pad_ratio=0.12):
    """
    Background subtraction for a coarse mask → bbox → SAM box prompt.
    ``hybrid_combine``:
      * ``sam`` — use SAM mask only (often tighter than diff/convex hull).
      * ``intersect`` — bitwise AND of diff mask and SAM (trim SAM leakage).
    """
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError:
        print('ERROR: segment_anything is not installed.')
        print('  pip install segment-anything')
        print('  Download checkpoint from https://github.com/facebookresearch/segment-anything')
        sys.exit(1)

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'[hybrid] Loading SAM {sam_model_type} from {sam_checkpoint} ...')
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    bg_root = bg_data or data_root
    for cam in cam_names:
        fg_path = find_image(data_root, cam, frame, ext)
        bg_path = find_image(bg_root, cam, bg_frame, ext)

        mask_bg = compute_background_subtraction_mask(
            fg_path, bg_path, threshold, morph_kernel, center_only,
            max_area_ratio,
        )
        if center_only and mask_bg.sum() == 0:
            print(f'  {cam}: WARNING - no object blob found (bg_sub empty)')

        img = cv2.imread(fg_path)
        if img is None:
            raise FileNotFoundError(f'Cannot read foreground: {fg_path}')
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        box = bbox_from_mask(mask_bg, pad_ratio=sam_box_pad_ratio)
        if box is None or mask_bg.sum() == 0:
            mask = mask_bg
            print(f'  {cam}: hybrid fallback — no bbox, using bg_sub only')
        else:
            predictor.set_image(img_rgb)
            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False,
            )
            m = np.asarray(masks)
            if m.ndim == 4:
                m = m[0]
            if m.ndim == 3:
                m = m[0]
            mask_sam = (m.astype(np.uint8) > 0).astype(np.uint8) * 255
            if hybrid_combine == 'intersect':
                mask = cv2.bitwise_and(mask_bg, mask_sam)
            else:
                mask = mask_sam

        mask = apply_mask_postprocess(mask, dilate=dilate, erode=erode)

        out_dir = join(output_dir, cam)
        os.makedirs(out_dir, exist_ok=True)
        out_path = join(out_dir, f'{frame:06d}.png')
        cv2.imwrite(out_path, mask)
        print(f'  {cam}: {out_path}  (fg pixels: {mask.sum() // 255})')

        if save_fg_dir is not None:
            fg_img = img.copy()
            fg_img[mask == 0] = 0
            out_fg_dir = join(save_fg_dir, cam)
            os.makedirs(out_fg_dir, exist_ok=True)
            out_fg_path = join(out_fg_dir, f'{frame:06d}{ext}')
            cv2.imwrite(out_fg_path, fg_img)
            print(f'    fg: {out_fg_path}')

        if vis_dir is not None:
            vis_path = save_mask_overlay(img, mask, vis_dir, cam, frame)
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
    parser.add_argument('--mode', choices=['bg_sub', 'sam', 'hybrid'], default='bg_sub',
                        help='Mask generation mode (hybrid = bg_sub bbox + SAM box prompt)')

    bg_group = parser.add_argument_group('background subtraction')
    bg_group.add_argument('--bg_frame', type=int, default=None,
                          help='Background-only frame index (required for bg_sub / hybrid)')
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
    bg_group.add_argument('--erode', type=int, default=0,
                          help='Erode final mask by N pixels (shrinks mask; reduces '
                               'included background/table halo)')

    sam_group = parser.add_argument_group('SAM / hybrid')
    sam_group.add_argument('--sam_checkpoint', default=None,
                           help='Path to SAM model checkpoint (required for sam / hybrid)')
    sam_group.add_argument('--sam_model_type', default='vit_h',
                           help='SAM model type (vit_h, vit_l, vit_b)')
    sam_group.add_argument('--hybrid_combine', choices=['sam', 'intersect'],
                           default='sam',
                           help='hybrid only: use SAM mask only, or AND with bg_sub')
    sam_group.add_argument('--sam_box_pad_ratio', type=float, default=0.12,
                           help='hybrid only: expand bg_sub bbox by this fraction '
                                'before SAM box prompt (each side)')

    parser.add_argument('--vis', action='store_true',
                        help='Save mask overlay visualizations to mask_vis/')
    parser.add_argument('--save_fg_images', action='store_true',
                        help='Save foreground-only RGB images (background set to black) '
                             'to <data>/foreground_images/<cam>/')

    args = parser.parse_args()

    output_dir = args.output or join(args.data, 'masks')
    vis_dir = join(args.data, 'mask_vis') if args.vis else None
    fg_out_dir = join(args.data, 'foreground_images') if args.save_fg_images else None
    cam_names = get_cam_names(args.data)
    print(f'[generate_masks] Cameras: {cam_names}')
    print(f'[generate_masks] Mode: {args.mode}, frame: {args.frame}')
    print(f'[generate_masks] Output: {output_dir}')
    if vis_dir:
        print(f'[generate_masks] Vis overlays: {vis_dir}')
    if fg_out_dir:
        print(f'[generate_masks] Foreground images: {fg_out_dir}')

    if args.mode == 'bg_sub':
        if args.bg_frame is None:
            parser.error('--bg_frame is required for background subtraction mode')
        background_subtraction(
            args.data, cam_names, args.frame, args.bg_frame, args.ext,
            args.threshold, args.morph_kernel, output_dir, vis_dir,
            bg_data=args.bg_data, center_only=args.center_only,
            dilate=args.dilate, erode=args.erode,
            max_area_ratio=args.max_area_ratio,
            save_fg_dir=fg_out_dir,
        )
    elif args.mode == 'sam':
        if args.sam_checkpoint is None:
            parser.error('--sam_checkpoint is required for SAM mode')
        sam_segmentation(
            args.data, cam_names, args.frame, args.ext, output_dir,
            args.sam_checkpoint, args.sam_model_type, vis_dir,
            save_fg_dir=fg_out_dir,
        )
    elif args.mode == 'hybrid':
        if args.bg_frame is None:
            parser.error('--bg_frame is required for hybrid mode')
        if args.sam_checkpoint is None:
            parser.error('--sam_checkpoint is required for hybrid mode')
        hybrid_bg_sam(
            args.data, cam_names, args.frame, args.bg_frame, args.ext,
            args.threshold, args.morph_kernel, output_dir, vis_dir,
            bg_data=args.bg_data, center_only=args.center_only,
            dilate=args.dilate, erode=args.erode,
            max_area_ratio=args.max_area_ratio,
            save_fg_dir=fg_out_dir,
            sam_checkpoint=args.sam_checkpoint,
            sam_model_type=args.sam_model_type,
            hybrid_combine=args.hybrid_combine,
            sam_box_pad_ratio=args.sam_box_pad_ratio,
        )

    print('[generate_masks] Done.')


if __name__ == '__main__':
    main()
