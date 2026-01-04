import numpy as np
import cv2
import glob
import os

# Create output directory
os.makedirs('./output', exist_ok=True)

# Define the chessboard rows and columns
CHECKERBOARD = (6, 9)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + \
                    cv2.fisheye.CALIB_FIX_SKEW

# Prepare object points (0,0,0), (1,0,0), (2,0,0), ...
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane
image_shape = None
counter = 0

# Loop through calibration images
for path in glob.glob('/Users/yubo/data/omni/calib/omni1230_front_fisheye/output/VID_20251230_111803_00_008_used/*.jpg'):
    img = cv2.imread(path)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if image_shape is None:
        image_shape = gray.shape[::-1]  # (width, height)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)
    # if ret:
    #     objpoints.append(objp)
    #     cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
    #     imgpoints.append(corners)

    # Inside loop over images
    if ret:
        # Skip degenerate frames
        if corners.shape[0] < 6 or np.linalg.norm(corners.max(axis=0) - corners.min(axis=0)) < 1e-3:
            continue
        objpoints.append(objp.copy())  # copy for each frame
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        imgpoints.append(corners)

    counter += 1

if counter == 0:
    raise RuntimeError("No calibration images found in datasets/*.png")

N_imm = counter  # number of calibration images

# Initialize calibration arrays
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_imm)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_imm)]

# Perform fisheye calibration
rms, _, _, _, _ = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    image_shape,
    K,
    D,
    rvecs,
    tvecs,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
)

import yaml

intrinsics = {'K': K.tolist(), 'D': D.tolist()}
with open('./output/fisheye_intrinsics.yml', 'w') as f:
    yaml.dump(intrinsics, f)
print("Saved intrinsic parameters to ./output/fisheye_intrinsics.yml")

print("K:\n", K)
print("D:\n", D)

# Load the image to undistort
img = cv2.imread("/Users/yubo/github/Motion-Capture/output/000150_fisheye.jpg")
h, w = img.shape[:2]

# Generate undistort map
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_16SC2)

# Apply remap to get undistorted image
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# Save results
cv2.imwrite('./output/undistorted.jpg', undistorted_img)
cv2.imshow('Original Image', img)
cv2.imshow('Undistorted Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
