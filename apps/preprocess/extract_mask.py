"""
Extract person masks using Segment Anything (SAM) for multiview image folders.

This script follows the same dataset layout/CLI style as extract_keypoints.py:
    {path}/images/{sub}/%06d.jpg
    {path}/{annot}/{sub}/%06d.json

Default behavior:
    write per-person mask points into each json annotation as `annots[*].mask`
Optional:
    save binary mask image to {path}/{mask}/{sub}/%06d.png via --save_mask_images
"""
import os
from os.path import join
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import json


def load_subs(path, subs):
    if len(subs) == 0:
        subs = sorted(os.listdir(join(path, "images")))
    subs = [sub for sub in subs if os.path.isdir(join(path, "images", sub))]
    if len(subs) == 0:
        subs = [""]
    return subs


def load_annots(annotname):
    if not os.path.exists(annotname):
        return None, []
    with open(annotname, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        annots = data.get("annots", [])
    elif isinstance(data, list):
        annots = data
    else:
        annots = []
    return data, annots


def save_annots(annotname, data, annots):
    if data is None:
        data = {"annots": annots}
    elif isinstance(data, dict):
        data["annots"] = annots
    else:
        data = annots
    with open(annotname, "w") as f:
        json.dump(data, f, indent=4)


def clip_bbox_xyxy(bbox, width, height):
    x1, y1, x2, y2 = bbox[:4]
    x1 = int(max(0, min(width - 1, x1)))
    y1 = int(max(0, min(height - 1, y1)))
    x2 = int(max(0, min(width - 1, x2)))
    y2 = int(max(0, min(height - 1, y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def mask_from_annot(predictor, annot, image_shape, args):
    h, w = image_shape[:2]
    mask = None
    # Prefer bbox prompts for stability.
    if "bbox" in annot and len(annot["bbox"]) >= 4:
        bbox = clip_bbox_xyxy(annot["bbox"], w, h)
        if bbox is not None and (len(annot["bbox"]) < 5 or annot["bbox"][4] >= args.bbox_thres):
            masks, scores, _ = predictor.predict(
                box=bbox,
                multimask_output=True,
            )
            best = int(np.argmax(scores))
            mask = masks[best]
    # Optional keypoint prompt fallback.
    if mask is None and args.prompt == "keypoints" and "keypoints" in annot:
        kpts = np.asarray(annot["keypoints"], dtype=np.float32)
        if kpts.ndim == 2 and kpts.shape[1] >= 3:
            valid = kpts[:, 2] > args.kpt_thres
            if np.any(valid):
                points = kpts[valid, :2]
                labels = np.ones((points.shape[0],), dtype=np.int32)
                masks, scores, _ = predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=True,
                )
                best = int(np.argmax(scores))
                mask = masks[best]
    return mask


def points_from_mask(mask, max_points, rng):
    ys, xs = np.where(mask > 0)
    if xs.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    if points.shape[0] > max_points:
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
    return points


def points_from_mask_contour(mask, max_points, rng):
    """Extract points along the mask contour instead of random interior points.

    Contour points are much better for silhouette Chamfer loss because the loss
    measures distance from projected mesh vertices to the silhouette *boundary*.
    Interior points provide biased gradients since they pull mesh vertices inward
    rather than toward the correct outline.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    all_pts = np.concatenate([c.reshape(-1, 2) for c in contours], axis=0)
    all_pts = all_pts.astype(np.float32)
    if all_pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if all_pts.shape[0] > max_points:
        idx = np.linspace(0, all_pts.shape[0] - 1, max_points, dtype=int)
        all_pts = all_pts[idx]
    return all_pts


def extract_mask_one_sub(image_root, annot_root, mask_root, args):
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except Exception as e:
        raise RuntimeError(
            "segment_anything is required. Install with "
            "`pip install git+https://github.com/facebookresearch/segment-anything.git`"
        ) from e

    if args.save_mask_images:
        os.makedirs(mask_root, exist_ok=True)
    device = args.device
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    rng = np.random.default_rng(0)

    point_fn = points_from_mask_contour if args.contour else points_from_mask

    imgnames = sorted([n for n in os.listdir(image_root) if n.endswith(args.ext)])
    # Apply frame range / step so SAM only runs on the frames we need.
    total = len(imgnames)
    start = max(0, min(args.start, total))
    end = min(args.end, total)
    step = max(1, args.step)
    selected_indices = set(range(start, end, step))

    for idx, imgname in enumerate(
        tqdm(imgnames, desc=f"mask {os.path.basename(annot_root) or 'root'}")
    ):
        if idx < start or idx >= end:
            continue
        if idx not in selected_indices:
            continue

        base = imgname.replace(args.ext, "")
        outname = join(mask_root, base + ".png")
        annotname = join(annot_root, base + ".json")
        data, annots = load_annots(annotname)
        if annots is None or len(annots) == 0:
            continue
        if not args.force and all(("mask" in ann and len(ann["mask"]) > 0) for ann in annots):
            if not args.save_mask_images:
                continue
            if args.save_mask_images and os.path.exists(outname):
                continue

        if args.save_mask_images and os.path.exists(outname) and not args.force:
            continue

        image = cv2.imread(join(image_root, imgname))
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        mask_all = np.zeros(image.shape[:2], dtype=np.uint8)
        for annot in annots:
            pred = mask_from_annot(predictor, annot, image.shape, args)
            if pred is None:
                annot["mask"] = []
                continue
            mask_person = np.zeros(image.shape[:2], dtype=np.uint8)
            mask_person[pred] = 255
            if args.erosion > 0:
                kernel = np.ones((args.erosion, args.erosion), np.uint8)
                mask_person = cv2.erode(mask_person, kernel, iterations=1)
            if args.dilation > 0:
                kernel = np.ones((args.dilation, args.dilation), np.uint8)
                mask_person = cv2.dilate(mask_person, kernel, iterations=1)
            points = point_fn(mask_person, args.mask_max_points, rng)
            annot["mask"] = points.tolist()
            mask_all = np.maximum(mask_all, mask_person)

        save_annots(annotname, data, annots)

        if args.save_mask_images:
            cv2.imwrite(outname, mask_all)
        if args.vis:
            vis = image.copy()
            vis[mask_all == 0] = 0
            cv2.imshow("mask", cv2.resize(vis, (vis.shape[1] // 2, vis.shape[0] // 2)))
            cv2.waitKey(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--subs", type=str, nargs="+", default=[])
    parser.add_argument("--annot", type=str, default="annots")
    parser.add_argument("--mask", type=str, default="masks")
    parser.add_argument("--ext", type=str, default=".jpg")
    parser.add_argument("--prompt", type=str, default="keypoints", choices=["bbox", "keypoints"])
    parser.add_argument("--bbox_thres", type=float, default=0.1)
    parser.add_argument("--kpt_thres", type=float, default=0.2)
    parser.add_argument("--sam_checkpoint", type=str, default="data/models/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam_model_type", type=str, default="vit_b", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mask_max_points", type=int, default=2000)
    parser.add_argument("--contour", action="store_true", default=True,
        help="extract contour points (default); much better for silhouette Chamfer loss")
    parser.add_argument("--interior", dest="contour", action="store_false",
        help="sample random interior points instead of contour (legacy behavior)")
    parser.add_argument("--start", type=int, default=0, help="first frame index")
    parser.add_argument("--end", type=int, default=100000, help="last frame index (exclusive)")
    parser.add_argument("--step", type=int, default=1,
        help="process every N-th frame; shape is constant so a subset suffices "
             "(e.g. --step 5 processes 20%% of frames)")
    parser.add_argument("--save_mask_images", action="store_true")
    parser.add_argument("--erosion", type=int, default=0)
    parser.add_argument("--dilation", type=int, default=0)
    parser.add_argument("--gpus", type=int, nargs="+", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(join(args.path, "images")) and os.path.exists(join(args.path, "videos")):
        cmd = f"python3 apps/preprocess/extract_image.py {args.path}"
        os.system(cmd)

    subs = load_subs(args.path, args.subs)
    if len(args.gpus) != 0:
        from easymocap.mytools.debug_utils import run_cmd
        nproc = len(args.gpus)
        plist = []
        for i in range(nproc):
            sublist = subs[i::nproc]
            if len(sublist) == 0:
                continue
            cmd = f"export CUDA_VISIBLE_DEVICES={args.gpus[i]} && python3 apps/preprocess/extract_mask.py {args.path}"
            cmd += f" --subs {' '.join(sublist)} --annot {args.annot} --mask {args.mask} --ext {args.ext}"
            cmd += f" --prompt {args.prompt} --bbox_thres {args.bbox_thres} --kpt_thres {args.kpt_thres}"
            cmd += f" --sam_checkpoint {args.sam_checkpoint} --sam_model_type {args.sam_model_type} --device cuda"
            cmd += f" --mask_max_points {args.mask_max_points}"
            cmd += f" --start {args.start} --end {args.end} --step {args.step}"
            cmd += f" --erosion {args.erosion} --dilation {args.dilation}"
            if args.contour:
                cmd += " --contour"
            else:
                cmd += " --interior"
            if args.save_mask_images:
                cmd += " --save_mask_images"
            if args.force:
                cmd += " --force"
            if args.vis:
                cmd += " --vis"
            cmd += " &"
            print(cmd)
            p = run_cmd(cmd, bg=False)
            plist.extend(p)
        for p in plist:
            p.join()
        return

    for sub in subs:
        image_root = join(args.path, "images", sub)
        annot_root = join(args.path, args.annot, sub)
        mask_root = join(args.path, args.mask, sub)
        if not os.path.exists(image_root):
            print(f"[WARN] image root not found: {image_root}")
            continue
        if not os.path.exists(annot_root):
            print(f"[WARN] annot root not found: {annot_root}")
            continue
        extract_mask_one_sub(image_root, annot_root, mask_root, args)


if __name__ == "__main__":
    main()
