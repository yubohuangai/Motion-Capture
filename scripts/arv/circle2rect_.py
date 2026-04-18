"""
File path: circle2rect.py
Description : Batch convert fisheye images to rectangular (ERP-like or perspective) images
"""

import cv2
import numpy as np
import os
import glob


def build_fisheye_to_rect_map(
    Wf, Hf, Wr, Hr,
    lon_fov_deg=180,
    lat_fov_deg=180
):
    """
    Build mapping from fisheye image to rect (ERP-like) image
    with independent longitude and latitude FOV control.
    """

    lon_fov = np.deg2rad(lon_fov_deg)
    lat_fov = np.deg2rad(lat_fov_deg)

    # Rectangular pixel grid
    ys, xs = np.indices((Hr, Wr), np.float32)

    # Longitude [-lon_fov/2, lon_fov/2]
    lon = (xs / Wr - 0.5) * lon_fov

    # Latitude [-lat_fov/2, lat_fov/2]
    lat = (0.5 - ys / Hr) * lat_fov

    # Spherical → Cartesian
    xsph = np.cos(lat) * np.sin(lon)
    ysph = np.sin(lat)
    zsph = np.cos(lat) * np.cos(lon)

    # Cartesian → fisheye polar
    phi = np.arccos(zsph)          # angle from optical axis
    theta = np.arctan2(ysph, xsph)

    # Fisheye radius
    r = phi * Hf / lon_fov

    # Image coordinates
    x_fish = r * np.cos(theta) + Wf / 2.0
    y_fish = Hf / 2.0 - r * np.sin(theta)

    # Valid region
    valid = phi <= (lon_fov / 2)

    x_fish[~valid] = -1
    y_fish[~valid] = -1

    return x_fish.astype(np.float32), y_fish.astype(np.float32)


def fisheye_to_rect(
    fisheye_img,
    out_size=None,
    lon_fov=180,
    lat_fov=180
):
    Hf, Wf = fisheye_img.shape[:2]

    if out_size is None:
        Hr, Wr = Hf, Wf
    else:
        Wr, Hr = out_size

    xmap, ymap = build_fisheye_to_rect_map(
        Wf, Hf, Wr, Hr,
        lon_fov_deg=lon_fov,
        lat_fov_deg=lat_fov
    )

    rect = cv2.remap(
        fisheye_img,
        xmap,
        ymap,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )
    return rect


def fisheye_to_perspective(
    fisheye_img,
    fov_deg=150,
    out_size=None
):
    Hf, Wf = fisheye_img.shape[:2]

    if out_size is None:
        Hp, Wp = Hf, Wf
    else:
        Wp, Hp = out_size

    fov = np.deg2rad(fov_deg)
    f = (Wp / 2) / np.tan(fov / 2)

    ys, xs = np.indices((Hp, Wp), np.float32)
    x = xs - Wp / 2
    y = Hp / 2 - ys

    Z = f
    X = x
    Y = y

    norm = np.sqrt(X*X + Y*Y + Z*Z)
    X /= norm
    Y /= norm
    Z /= norm

    phi = np.arccos(Z)
    theta = np.arctan2(Y, X)

    f_fish = (min(Wf, Hf) / 2) / (np.pi / 2)
    # r = f_fish * phiaeqq  37
    k1 = 0.025
    k2 = 0.005
    r = f_fish * (phi + k1 * phi ** 3 + k2 * phi ** 5)

    x_fish = r * np.cos(theta) + Wf / 2
    y_fish = Hf / 2 - r * np.sin(theta)

    valid = phi <= np.pi / 2
    x_fish[~valid] = -1
    y_fish[~valid] = -1

    return cv2.remap(
        fisheye_img,
        x_fish.astype(np.float32),
        y_fish.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )


if __name__ == "__main__":
    input_dir = "/Users/yubo/data/s2/seq1/view32_fisheye"
    # input_dir = "/Users/yubo/github/Motion-Capture/output"
    output_dir = "/Users/yubo/data/s2/seq1/view32_fisheye_fov150"
    # output_dir = "/Users/yubo/github/Motion-Capture/output"
    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))

    for img_path in image_paths:
        fisheye_img = cv2.imread(img_path)
        if fisheye_img is None:
            print(f"Failed to load {img_path}, skipping...")
            continue

        # rect_img = fisheye_to_rect(fisheye_img, lon_fov=180, lat_fov=140)
        rect_img = fisheye_to_perspective(fisheye_img, fov_deg=150)

        base, ext = os.path.splitext(os.path.basename(img_path))
        out_path = os.path.join(output_dir, f"{base}{ext}")
        cv2.imwrite(out_path, rect_img)