import cv2
import numpy as np
import os
import glob


def build_erp_to_perspective_map(
    Wr, Hr,          # ERP image size
    Wp, Hp,          # Perspective image size
    fov_deg=150,     # Perspective horizontal FOV
    lon_center_deg=0,
    lat_center_deg=0
):
    """
    Build mapping from ERP (lon-lat) image to perspective image.
    """

    fov = np.deg2rad(fov_deg)
    lon0 = np.deg2rad(lon_center_deg)
    lat0 = np.deg2rad(lat_center_deg)

    # Perspective camera intrinsics
    f = (Wp / 2) / np.tan(fov / 2)

    ys, xs = np.indices((Hp, Wp), np.float32)
    x = xs - Wp / 2
    y = Hp / 2 - ys

    # Camera coordinates
    X = x
    Y = y
    Z = f * np.ones_like(x)

    # Normalize to unit sphere
    norm = np.sqrt(X*X + Y*Y + Z*Z)
    X /= norm
    Y /= norm
    Z /= norm

    # Rotate camera (yaw = lon, pitch = lat)
    # Yaw (around Y axis)
    Xr =  np.cos(lon0) * X + np.sin(lon0) * Z
    Yr =  Y
    Zr = -np.sin(lon0) * X + np.cos(lon0) * Z

    # Pitch (around X axis)
    Xp = Xr
    Yp =  np.cos(lat0) * Yr - np.sin(lat0) * Zr
    Zp =  np.sin(lat0) * Yr + np.cos(lat0) * Zr

    # Cartesian → lon/lat
    lon = np.arctan2(Xp, Zp)
    lat = np.arcsin(Yp)

    # Lon/lat → ERP pixel
    x_erp = (lon / (2 * np.pi) + 0.5) * Wr
    y_erp = (0.5 - lat / np.pi) * Hr

    return x_erp.astype(np.float32), y_erp.astype(np.float32)


def erp_to_perspective(
    erp_img,
    out_size=None,
    fov_deg=150,
    lon_center=0,
    lat_center=0
):
    Hr, Wr = erp_img.shape[:2]

    if out_size is None:
        Hp, Wp = Hr, Wr
    else:
        Wp, Hp = out_size

    xmap, ymap = build_erp_to_perspective_map(
        Wr, Hr, Wp, Hp,
        fov_deg=fov_deg,
        lon_center_deg=lon_center,
        lat_center_deg=lat_center
    )

    return cv2.remap(
        erp_img,
        xmap,
        ymap,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )


if __name__ == "__main__":
    input_dir = "/Users/yubo/data/s2/seq1/360/view32/images/view32"
    output_dir = "/Users/yubo/data/s2/seq1/360/view32_fov150_/images/view32"
    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(input_dir, "000000.jpg")))

    for img_path in image_paths:
        erp_img = cv2.imread(img_path)
        if erp_img is None:
            continue

        persp = erp_to_perspective(
            erp_img,
            fov_deg=160,
            lon_center=0,   # look direction (yaw)
            lat_center=0    # look direction (pitch)
        )

        out_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, persp)