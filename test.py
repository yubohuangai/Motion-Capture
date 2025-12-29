import os
import cv2
import numpy as np

HALPE_IN_BODY25 = [0, 16, 15, 18, 17, 6, 5, 2, 7, 4, 12, 9, 13, 10, 14, 11, 1, 8, 19, 22, 20, 23, 21, 24]


def halpe2body25(points2d):
    """
    Convert HALPE26 (26x3) keypoints to BODY25 (25x3)
    Input:  (N, 26, 3)
    Output: (N, 25, 3)
    """
    assert points2d.shape[1] == 26

    kpts = np.zeros((points2d.shape[0], 25, 3), dtype=points2d.dtype)

    # copy x, y
    kpts[:, HALPE_IN_BODY25, :2] = points2d[:, :, :2]

    # copy confidence
    kpts[:, HALPE_IN_BODY25, 2] = points2d[:, :, 2]

    return kpts


# one fake person
pts = np.random.rand(1, 26, 3)
out = halpe2body25(pts)

print(out.shape)  # must be (1, 25, 3)