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


COCO17_IN_BODY25 = [0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]
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


class MMPoseDetector:

    def __init__(self, model_cfg, model_weights, config_name, to_openpose=True):
        """
        model_name: str, name of the MMPose model config (e.g., 'TopDownHRNet')
        to_openpose: whether to convert COCO17 keypoints to BODY25
        show: visualize results
        """
        self.to_openpose = to_openpose
        self.inferencer = MMPoseInferencer(
            pose2d=config_name,
            device="cuda"
        )
        # self.inferencer = MMPoseInferencer(
        #     pose2d=model_cfg,
        #     pose2d_weights=model_weights,
        #     device="cuda"
        # )
        
    def predict(self, image):
        """Run MMPose inference on a single image"""
        results = self.inferencer(image)
        output = next(results)
        persons = output['predictions'][0]
        kpts25_all = []
        for pid, person in enumerate(persons):
            kpts17 = np.array(person['keypoints'])
            if 'keypoint_scores' in person:
                conf = np.array(person['keypoint_scores'])
            else:
                conf = np.ones((kpts17.shape[0],), dtype=kpts17.dtype)

            if kpts17.shape[1] == 2:
                kpts17 = np.concatenate([kpts17, conf[:,None]], axis=1)
            else:
                kpts17[:,2] = conf

            kpts25 = coco17tobody25(kpts17[None])[0] 
            kpts25_all.append(kpts25.tolist())

        return kpts25_all


def extract_2d(image_root, annot_root, config, to_openpose=True):
    config.pop('force')
    ext = config.pop('ext')
    detector = MMPoseDetector(model_cfg=config['pose2d'], model_weights=config['pose2d_weights'], config_name=config['config_name'], to_openpose=to_openpose)
    imgnames = sorted(glob(join(image_root, '*'+ext)))
    for imgname in tqdm(imgnames, desc='{:10s}'.format(os.path.basename(annot_root))):
        base = os.path.basename(imgname).replace(ext, '')
        annotname = join(annot_root, base+'.json')
        annots = read_json(annotname)
        detections = np.array([data['bbox'] for data in annots['annots']])
        image = cv2.imread(imgname)
        kpts2d = detector.predict(image)
        for i in range(detections.shape[0]):
            annot_ = annots['annots'][i]
            if i >= len(kpts2d):
                # Skip if MMPose didn't detect this person
                # but keep previous annotation structure
                continue
            annot_['keypoints'] = kpts2d[i]
        save_annot(annotname, annots)