"""
Filepath: scripts/postprocess/check_equal.py

Check if two NPZ files have identical content.
"""

import numpy as np

npz1_path = "/Users/yubo/data/s2/seq1/360/output/poseformerv2/view32_fov150/input_2D/keypoints.npz"
npz2_path = "/Users/yubo/data/s2/seq1/360/output/poseformerv2/view32_fov150/input_2D/keypoints_.npz"

# Load NPZ files
npz1 = np.load(npz1_path, allow_pickle=True)
npz2 = np.load(npz2_path, allow_pickle=True)

# Check keys
keys1 = set(npz1.files)
keys2 = set(npz2.files)

if keys1 != keys2:
    print("NPZ files have different keys")
    print("Keys in first:", keys1)
    print("Keys in second:", keys2)
else:
    all_equal = True
    for key in keys1:
        arr1 = npz1[key]
        arr2 = npz2[key]

        if arr1.shape != arr2.shape:
            print(f"Key '{key}' has different shapes: {arr1.shape} vs {arr2.shape}")
            all_equal = False
            continue

        if not np.array_equal(arr1, arr2):
            print(f"Key '{key}' has different values")
            all_equal = False

    if all_equal:
        print("NPZ files are identical")

npz1.close()
npz2.close()
