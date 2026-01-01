"""
coding: utf-8
FilePath: rect2circle.py
@Descrip : Batch convert rectangular (ERP-like) images to fisheye images

"""

import cv2
import numpy as np
import os
import glob

def build_rect_to_fisheye_map(Wr, Hr, Wf, Hf, fov_deg=180):
    fov = np.deg2rad(fov_deg)

    ys, xs = np.indices((Hf, Wf), np.float32)
    x = xs - Wf / 2.0
    y = Hf / 2.0 - ys

    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    phi = r * fov / Hf
    valid = phi <= (fov / 2)

    xsph = np.sin(phi) * np.cos(theta)
    ysph = np.sin(phi) * np.sin(theta)
    zsph = np.cos(phi)

    lon = np.arctan2(xsph, zsph)
    lat = np.arcsin(ysph)

    x_rect = (lon / fov + 0.5) * Wr
    y_rect = (0.5 - lat / np.pi) * Hr

    x_rect[~valid] = -1
    y_rect[~valid] = -1

    return x_rect.astype(np.float32), y_rect.astype(np.float32)


def rect_to_fisheye(rect_img, fov=180):
    Hr, Wr = rect_img.shape[:2]
    Hf, Wf = Hr, Wr  # keep fisheye image same size as original

    xmap, ymap = build_rect_to_fisheye_map(Wr, Hr, Wf, Hf, fov)

    fisheye = cv2.remap(
        rect_img,
        xmap,
        ymap,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )
    return fisheye


if __name__ == "__main__":
    input_dir = "/Users/yubo/data/s2/seq1/view_32"
    output_dir = "/Users/yubo/data/s2/seq1/view_32_fisheye"
    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))

    for img_path in image_paths:
        rect_img = cv2.imread(img_path)
        if rect_img is None:
            print(f"Failed to load {img_path}, skipping...")
            continue

        fisheye_img = rect_to_fisheye(rect_img, fov=180)

        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, fisheye_img)
