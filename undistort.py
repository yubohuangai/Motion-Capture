# coding: utf-8
# @Descrip : Undistort image using EasyMocap / OpenCV intri.yml (no extri needed)

import os
import cv2
import numpy as np


def read_intri_only(intri_path, cam_name):
    fs = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open {intri_path}")

    K = fs.getNode(f"K_{cam_name}").mat()
    if K is None:
        raise KeyError(f"K_{cam_name} not found in intri.yml")

    dist = fs.getNode(f"dist_{cam_name}").mat()
    if dist is None:
        dist = fs.getNode(f"D_{cam_name}").mat()

    if dist is None:
        raise KeyError(f"dist_{cam_name} or D_{cam_name} not found in intri.yml")

    fs.release()
    return K, dist


def undistort_image(img_path, intri_path, cam_name, output_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    print(f"H: {h}, W: {w}")
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    K, dist = read_intri_only(intri_path, cam_name)

    # undistorted = cv2.undistort(img, K, dist, None)

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, dist, np.eye(3), K, (w, h), cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, undistorted_img)
    print(f"[OK] Undistorted image saved to:\n{output_path}")


if __name__ == "__main__":
    img_path = "/Users/yubo/github/Motion-Capture/output/000150_fisheye.jpg"
    intri_path = "/Users/yubo/data/omni/calib/omni1230_selected/output/intri_edit.yml"
    cam_name = "VID_20251230_111803_00_008"
    output_path = "/Users/yubo/github/Motion-Capture/output/000150_fisheye_undistorted.jpg"

    undistort_image(img_path, intri_path, cam_name, output_path)
