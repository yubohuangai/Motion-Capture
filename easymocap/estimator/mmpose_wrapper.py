'''
Docstring for easymocap.estimator.mmpose_wrapper
'''
import os
import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer
from tqdm import tqdm
from glob import glob
from os.path import join
from .wrapper_base import check_result, save_annot
from ..annotator.file_utils import read_json
import contextlib
import io


COCO17_IN_BODY25 = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
BODY25_IN_HALPE = [0, 18, 6, 8, 10, 5, 7, 9, 10, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 20, 22, 24, 21, 23, 25]
HALPE_TO_BODY25 = {
    0: 0,
    1: 16,
    2: 15,
    3: 18,
    4: 17,
    5: 5,
    6: 2,
    7: 6,
    8: 3,
    9: 7,
    10: 4,
    11: 12,
    12: 9,
    13: 13,
    14: 10,
    15: 14,
    16: 11,
    # HALPE index 17 is intentionally dropped
    18: 1,
    19: 8,
    20: 19,
    21: 22,
    22: 20,
    23: 23,
    24: 21,
    25: 24
}
pairs = [[1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14], [1, 0], [0,15], [15,17], [0,16], [16,18], [14,19], [19,20], [14,21], [11,22], [22,23], [11,24]]

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


def halpe2body25(points2d):
    """
    Convert HALPE26 (N,26,3) → BODY25 (N,25,3)
    """
    assert points2d.shape[1] == 26

    N = points2d.shape[0]
    kpts = np.zeros((N, 25, 3), dtype=points2d.dtype)

    for h_idx, b_idx in HALPE_TO_BODY25.items():
        kpts[:, b_idx, :] = points2d[:, h_idx, :]

    return kpts


class MMPoseDetector:

    def __init__(self, model_cfg, model_weights, config_name, to_openpose=True):
        """
        model_name: str, name of the MMPose model config (e.g., 'TopDownHRNet')
        to_openpose: whether to convert COCO17 keypoints to BODY25
        show: visualize results
        """
        self.to_openpose = to_openpose
        # self.inferencer = MMPoseInferencer(
        #     pose2d=config_name,
        #     device="cuda"
        # )
        self.inferencer = MMPoseInferencer(
            pose2d=model_cfg,
            pose2d_weights=model_weights,
            device="cuda"
        )

    def predict_crop(self, image, bbox):
        """
        image: full image (H, W, 3)
        bbox: [x1, y1, x2, y2] or [x1, y1, x2, y2, score]
        """
        x1, y1, x2, y2 = map(int, bbox[:4])

        # Clamp to image bounds
        h, w = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]

        results = self.inferencer(crop)
        output = next(results)

        persons = output['predictions'][0]
        if len(persons) == 0:
            return None

        # Take the best person (top-down usually returns one)
        person = persons[0]

        kpts = np.array(person['keypoints'])
        if 'keypoint_scores' in person:
            conf = np.array(person['keypoint_scores'])
        else:
            conf = np.ones((kpts.shape[0],), dtype=kpts.dtype)

        if kpts.shape[1] == 2:
            kpts = np.concatenate([kpts, conf[:, None]], axis=1)
        else:
            kpts[:, 2] = conf

        if self.to_openpose:
            kpts = halpe2body25(kpts[None])[0]

        # map back to full image if cropped
        if bbox is not None:
            kpts[:, 0] += x1
            kpts[:, 1] += y1

        return kpts.tolist()


    # def predict(self, image):
    #     """Run MMPose inference on a single image"""
    #     results = self.inferencer(image)
    #     output = next(results)
    #     persons = output['predictions'][0]
    #     kpts25_all = []
    #     for pid, person in enumerate(persons):
    #         kpts17 = np.array(person['keypoints'])
    #         if 'keypoint_scores' in person:
    #             conf = np.array(person['keypoint_scores'])
    #         else:
    #             conf = np.ones((kpts17.shape[0],), dtype=kpts17.dtype)
    #
    #         if kpts17.shape[1] == 2:
    #             kpts17 = np.concatenate([kpts17, conf[:,None]], axis=1)
    #         else:
    #             kpts17[:,2] = conf
    #
    #         kpts25 = coco17tobody25(kpts17[None])[0]
    #         kpts25_all.append(kpts25.tolist())
    #
    #     return kpts25_all


def extract_2d(image_root, annot_root, config, to_openpose=False):
    config.pop('force')
    ext = config.pop('ext')
    detector = MMPoseDetector(
        model_cfg=config['pose2d'],
        model_weights=config['pose2d_weights'],
        config_name=config['config_name'],
        to_openpose=to_openpose
    )
    imgnames = sorted(glob(join(image_root, '*' + ext)))
    for imgname in tqdm(imgnames, desc='{:10s}'.format(os.path.basename(annot_root))):
        base = os.path.basename(imgname).replace(ext, '')
        annotname = join(annot_root, base + '.json')
        annots = read_json(annotname)
        detections = np.array([data['bbox'] for data in annots['annots']])
        image = cv2.imread(imgname)

        # If no bbox detected, create a dummy bbox for whole image
        if detections.shape[0] == 0:
            full_bbox = [0, 0, image.shape[1], image.shape[0]]
            kpts_all = detector.predict_crop(image, full_bbox)
            if kpts_all is not None:
                # create a new annotation for the person
                new_annot = {
                    'personID': 0,
                    'bbox': full_bbox + [1.0],  # assume score=1.0
                    'keypoints': kpts_all,
                    'isKeyframe': True
                }
                annots['annots'].append(new_annot)

        else:
            # process detected bboxes
            for i in range(detections.shape[0]):
                annot_ = annots['annots'][i]
                bbox = annot_['bbox']

                kpts = detector.predict_crop(image, bbox)
                if kpts is None:
                    continue

                annot_['keypoints'] = kpts

        save_annot(annotname, annots)
