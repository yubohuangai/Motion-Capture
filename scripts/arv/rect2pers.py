"""
File: rect2pers.py
Description: Convert equirectangular image to perspective image.
"""


import cv2
import numpy as np
import os
import glob


def rect_to_perspective_direct(
    rect_img,
    rect_fov_deg=180,
    persp_fov_deg=150,
    out_size=None,
    k1=0.025,
    k2=0.005,
    interpolation=cv2.INTER_LINEAR
):
    Hr, Wr = rect_img.shape[:2]
    if out_size is None:
        Wp, Hp = Wr, Hr
    else:
        Wp, Hp = out_size

    rect_fov = np.deg2rad(rect_fov_deg)
    persp_fov = np.deg2rad(persp_fov_deg)

    # Pinhole focal length (horizontal FOV)
    f = (Wp / 2.0) / np.tan(persp_fov / 2.0)

    ys, xs = np.indices((Hp, Wp), np.float32)
    x = xs - Wp / 2.0
    y = Hp / 2.0 - ys

    # Ray in camera coords
    X = x
    Y = y
    Z = np.full_like(X, f)

    # Normalize rays
    norm = np.sqrt(X * X + Y * Y + Z * Z)
    X /= norm
    Y /= norm
    Z /= norm

    # (2) Exact alternative to arccos(Z) for unit vectors:
    #     phi_p = atan2(sqrt(X^2+Y^2), Z)
    r_xy = np.sqrt(X * X + Y * Y)
    phi_p = np.arctan2(r_xy, Z)

    # Fisheye polynomial radius (same as your previous code)
    # Note: (min/2)/(pi/2) = min/pi
    f_fish = (min(Wr, Hr) / np.pi)

    r = f_fish * (phi_p + k1 * phi_p**3 + k2 * phi_p**5)

    # Fold into rect pano polar angle (same logic as before)
    phi_eff = (r * rect_fov) / Hr

    # Valid region (same as your original: only hemisphere)
    valid = phi_p <= (np.pi / 2.0)

    # (3) Avoid theta + sin/cos(theta)
    #     cos(theta)=X/r_xy, sin(theta)=Y/r_xy
    # Handle center ray where r_xy==0
    eps = 1e-8
    inv_rxy = 1.0 / np.maximum(r_xy, eps)
    cos_t = X * inv_rxy
    sin_t = Y * inv_rxy

    # Sphere coords without theta:
    s = np.sin(phi_eff)
    xsph = s * cos_t
    ysph = s * sin_t
    zsph = np.cos(phi_eff)

    lon = np.arctan2(xsph, zsph)
    lat = np.arcsin(np.clip(ysph, -1.0, 1.0))

    # lon/lat -> rect pixel coords (same as before)
    x_rect = (lon / rect_fov + 0.5) * Wr
    y_rect = (0.5 - lat / np.pi) * Hr

    x_rect[~valid] = -1
    y_rect[~valid] = -1

    return cv2.remap(
        rect_img,
        x_rect.astype(np.float32),
        y_rect.astype(np.float32),
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT
    )

if __name__ == "__main__":
    input_dir = "/Users/yubo/data/s2/seq1/360/view32/images/view32"
    output_dir = "/Users/yubo/github/Motion-Capture/output"
    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(input_dir, "000244.jpg")))

    for img_path in image_paths:
        rect_img = cv2.imread(img_path)
        if rect_img is None:
            print(f"Failed to load {img_path}, skipping...")
            continue

        # One-step replacement (matches your original settings)
        pers_img = rect_to_perspective_direct(
            rect_img,
            rect_fov_deg=180,
            persp_fov_deg=170,
            out_size=None,
            k1=0.025,
            k2=0.005
        )

        base, ext = os.path.splitext(os.path.basename(img_path))
        out_path = os.path.join(output_dir, f"{base}_fov170{ext}")
        cv2.imwrite(out_path, pers_img)