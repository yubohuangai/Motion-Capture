import os
import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer

# COCO17 -> Body25 mapping
COCO17_IN_BODY25 = [0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]

def coco17tobody25(points2d):
    """
    Convert batch of keypoints from COCO17 to Body25 format.
    points2d: numpy array of shape (N_people, 17, 3)
    Returns: (N_people, 25, 3)
    """
    kpts = np.zeros((points2d.shape[0], 25, 3))
    kpts[:, COCO17_IN_BODY25, :2] = points2d[:, :, :2]
    kpts[:, COCO17_IN_BODY25, 2:3] = points2d[:, :, 2:3]

    # pelvis
    kpts[:, 8, :2] = kpts[:, [9, 12], :2].mean(axis=1)
    kpts[:, 8, 2] = kpts[:, [9, 12], 2].min(axis=1)
    # neck
    kpts[:, 1, :2] = kpts[:, [2, 5], :2].mean(axis=1)
    kpts[:, 1, 2] = kpts[:, [2, 5], 2].min(axis=1)
    return kpts

def process_image(image_path, inferencer):
    """
    Process a single image and return annotation in Body25 format.
    """
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    results = inferencer(image)

    annots = []

    # For each person detected
    for person in results['predictions'][0]:
        keypoints = np.array(person['keypoints'])  # shape (17,3)
        keypoints_body25 = coco17tobody25(keypoints[None, :, :])[0]

        # Use the bounding box from the first element
        bbox = person['bbox'][0]  # [x1, y1, x2, y2, score]

        annot = {
            "personID": 0,
            "bbox": bbox,
            "keypoints": keypoints_body25.tolist(),
            "isKeyframe": False
        }
        annots.append(annot)

    return {
        "filename": os.path.relpath(image_path),
        "height": h,
        "width": w,
        "annots": annots,
        "isKeyframe": False
    }

if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='/Users/yubo/github/data/forest/seq1_1/images/01/000000.jpg', required=False, help="Input image path")
    parser.add_argument('--model', type=str, default='rtmpose-l_8xb256-420e_aic-coco-384x288', help="MMPose model")
    args = parser.parse_args()

    inferencer = MMPoseInferencer(args.model, device='cuda:0')

    annot = process_image(args.image, inferencer)
    print(annot)



# inferencer = MMPoseInferencer('rtmpose-l_8xb256-420e_aic-coco-384x288')


# COCO17_IN_BODY25 = [0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]
# pairs = [[1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14], [1, 0], [0,15], [15,17], [0,16], [16,18], [14,19], [19,20], [14,21], [11,22], [22,23], [11,24]]
# def coco17tobody25(points2d):
#     kpts = np.zeros((points2d.shape[0], 25, 3))
#     kpts[:, COCO17_IN_BODY25, :2] = points2d[:, :, :2]
#     kpts[:, COCO17_IN_BODY25, 2:3] = points2d[:, :, 2:3]
#     # pelvis
#     kpts[:, 8, :2] = kpts[:, [9, 12], :2].mean(axis=1)
#     kpts[:, 8, 2] = kpts[:, [9, 12], 2].min(axis=1)
#     # neck
#     kpts[:, 1, :2] = kpts[:, [2, 5], :2].mean(axis=1)
#     kpts[:, 1, 2] = kpts[:, [2, 5], 2].min(axis=1)
#     # 需要交换一下
#     # kpts = kpts[:, :, [1,0,2]]
#     return kpts


# input image: /Users/yubo/github/data/forest/seq1_1/images/01/000000.jpg
# output example:
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

