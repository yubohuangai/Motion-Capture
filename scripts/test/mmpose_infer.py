import os
import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer


# COCO17 → BODY25 mapping
COCO17_IN_BODY25 = [0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]

def coco17tobody25(points2d):
    """
    Convert COCO17 (17x3) keypoints to BODY25 (25x3), preserving confidence.
    Input: (N, 17, 3) -> x, y, score
    Output: (N, 25, 3)
    """
    kpts = np.zeros((points2d.shape[0], 25, 3))

    # copy x, y
    kpts[:, COCO17_IN_BODY25, :2] = points2d[:, :, :2]

    # copy confidence
    kpts[:, COCO17_IN_BODY25, 2] = points2d[:, :, 2]

    # pelvis = midpoint(hip_l, hip_r)
    kpts[:, 8, :2] = kpts[:, [9, 12], :2].mean(axis=1)
    kpts[:, 8, 2] = kpts[:, [9, 12], 2].min(axis=1)  # confidence: use min of hips

    # neck = midpoint(shoulder_l, shoulder_r)
    kpts[:, 1, :2] = kpts[:, [2, 5], :2].mean(axis=1)
    kpts[:, 1, 2] = kpts[:, [2, 5], 2].min(axis=1)  # confidence: use min of shoulders

    return kpts


def process_image(image_path, inferencer):
    """
    Run inference on one image using MMPoseInferencer.
    Return annotation in BODY25 format.
    """
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # run inference → generator
    results = inferencer(image)

    # take first output (because generator may yield many frames)
    output = next(results)

    persons = output['predictions'][0]   # list of N persons

    annots = []
    for pid, person in enumerate(persons):
        kpts17 = np.array(person['keypoints'])  # (17,3)
        kpts17 = np.array(person['keypoints'])
        if 'keypoint_scores' in person:
            conf = np.array(person['keypoint_scores'])
        else:
            conf = np.ones((kpts17.shape[0],), dtype=kpts17.dtype)

        if kpts17.shape[1] == 2:
            kpts17 = np.concatenate([kpts17, conf[:,None]], axis=1)
        else:
            kpts17[:,2] = conf
        kpts25 = coco17tobody25(kpts17[None])[0]  # convert

        bbox = person['bbox']  # correct bbox format

        annots.append({
            "personID": pid,
            "bbox": bbox,
            "keypoints": kpts25.tolist(),
            "isKeyframe": False
        })

    return {
        "filename": os.path.relpath(image_path),
        "height": h,
        "width": w,
        "annots": annots,
        "isKeyframe": False
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str,
                        default='/mnt/yubo/forest/seq1_1/images/01/000000.jpg')
    parser.add_argument('--model_cfg', type=str,
                        default='/mnt/yubo/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-384x288.py')
    parser.add_argument('--model_weights', type=str,
                        default='data/models/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth')

    args = parser.parse_args()

    inferencer = MMPoseInferencer(
        pose2d=args.model_cfg,
        pose2d_weights=args.model_weights,
        device='cuda:0'
    )

    annot = process_image(args.image, inferencer)
    print(annot)


# 
# output: /mnt/yubo/forest/seq1_1/annots-mm/000000.json file example:
# {
#     "filename": "images/01/000000.jpg",
#     "height": 2160,
#     "width": 3840,
#     "annots": [
#         {
#             "personID": 0,
#             "bbox": [1238.63, 957.00, 1643.00, 1987.33, 0.99],
#             "keypoints": [
#               [1476.00, 1075.00,    0.72], 
#               [1461.25, 1159.73,    0.88], 
#               [1566.05, 1166.08,    0.90], 
#               [1611.00, 1350.00,    0.82], 
#               [1535.00, 1374.00,    0.00], 
#               [1356.45, 1153.38,    0.88], 
#               [1273.88, 1293.11,    0.88], 
#               [1340.00, 1400.00,    0.83], 
#               [1442.19, 1448.72,    0.69], 
#               [1496.18, 1458.25,    0.70], 
#               [1458.07, 1718.66,    0.91], 
#               [1337.40, 1921.91,    0.89], 
#               [1388.21, 1439.20,    0.69], 
#               [1534.00, 1566.00,    0.00], 
#               [1450.00, 1767.00,    0.00], 
#               [1521.59, 1051.75,    0.82], 
#               [1458.00, 1040.00,    0.72], 
#               [1541.00, 1074.00,    0.87], 
#               [1426.32, 1058.11,    0.86], 
#               [   0.00,    0.00,    0.00], 
#               [   0.00,    0.00,    0.00], 
#               [   0.00,    0.00,    0.00], 
#               [   0.00,    0.00,    0.00], 
#               [   0.00,    0.00,    0.00], 
#               [   0.00,    0.00,    0.00]
#             ],
#             "isKeyframe": false
#         }
#     ],
#     "isKeyframe": false
# }

