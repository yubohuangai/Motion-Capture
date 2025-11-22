import cv2
import numpy as np
import os

# --- CONFIG ---
image_path = '/mnt/yubo/emily/extri_data/images/01/000000.jpg'
output_path = '/mnt/yubo/emily/extri_data/images/01/000000_detected.jpg'
pattern_size = (9, 6)  # number of inner corners (columns, rows)
# ----------------

# Load image
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Cannot read image: {image_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find chessboard corners
ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

if ret:
    print(f"Chessboard detected with {len(corners)} corners.")
    # Refine corner positions
    corners2 = cv2.cornerSubPix(
        gray, corners, (11,11), (-1,-1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )
    # Draw corners
    cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
    # Save annotated image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Annotated image saved to {output_path}")
else:
    print("Cannot find chessboard in the image.")
