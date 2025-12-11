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
import torch


COCO17_IN_BODY25 = [0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]
pairs = [[1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14], [1, 0], [0,15], [15,17], [0,16], [16,18], [14,19], [19,20], [14,21], [11,22], [22,23], [11,24]]
def coco17tobody25(points2d):
    kpts = np.zeros((points2d.shape[0], 25, 3))
    kpts[:, COCO17_IN_BODY25, :2] = points2d[:, :, :2]
    kpts[:, COCO17_IN_BODY25, 2:3] = points2d[:, :, 2:3]
    # pelvis
    kpts[:, 8, :2] = kpts[:, [9, 12], :2].mean(axis=1)
    kpts[:, 8, 2] = kpts[:, [9, 12], 2].min(axis=1)
    # neck
    kpts[:, 1, :2] = kpts[:, [2, 5], :2].mean(axis=1)
    kpts[:, 1, 2] = kpts[:, [2, 5], 2].min(axis=1)
    # 需要交换一下
    # kpts = kpts[:, :, [1,0,2]]
    return kpts


def bbox_from_keypoints(keypoints, rescale=1.2, detection_thresh=0.05, MIN_PIXEL=5):
    """Get center and scale for bounding box from openpose detections."""
    valid = keypoints[:,-1] > detection_thresh
    if valid.sum() < 3:
        return [0, 0, 100, 100, 0]
    valid_keypoints = keypoints[valid][:,:-1]
    center = (valid_keypoints.max(axis=0) + valid_keypoints.min(axis=0))/2
    bbox_size = valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)
    # adjust bounding box tightness
    if bbox_size[0] < MIN_PIXEL or bbox_size[1] < MIN_PIXEL:
        return [0, 0, 100, 100, 0]
    bbox_size = bbox_size * rescale
    bbox = [
        center[0] - bbox_size[0]/2, 
        center[1] - bbox_size[1]/2,
        center[0] + bbox_size[0]/2, 
        center[1] + bbox_size[1]/2,
        keypoints[valid, 2].mean()
    ]
    return bbox


class MMPoseDetector:

    def __init__(self, model_cfg='/mnt/yubo/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-384x288.py', model_weights='data/models/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth', to_openpose=True):
        """
        model_name: str, name of the MMPose model config (e.g., 'TopDownHRNet')
        to_openpose: whether to convert COCO17 keypoints to BODY25
        show: visualize results
        """
        self.to_openpose = to_openpose
        self.inferencer = MMPoseInferencer(
            pose2d=model_cfg,
            pose2d_weights=model_weights,
            device="cuda"
        )
        
    def process_image(self, image):
        """Run MMPose inference on a single image"""
        results = self.inferencer(image)
        if len(results) == 0:
            # no person detected
            keypoints = np.zeros((1, 17, 3))  # COCO17
        else:
            # take the first person detected
            keypoints = results[0]['keypoints']  # shape: (17, 3)

        keypoints = keypoints[np.newaxis, ...]  # make batch dim
        if self.to_openpose:
            kpts25 = coco17tobody25(keypoints)
        else:
            kpts25 = keypoints
        bbox = bbox_from_keypoints(kpts25[0])
        return kpts25[0], bbox

    def __call__(self, images):
        """
        images: list of images (H,W,3)
        returns: list of annotations dicts
        """
        annots_all = []
        for nv, image_ in enumerate(images):
            image_height, image_width, _ = image_.shape
            image = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            kpts, bbox = self.process_image(image_)
            data = {
                'personID': 0,
            }
            annots = {
                'filename': '{}/run.jpg'.format(nv),
                'height': image_height,
                'width': image_width,
                'annots': [
                    data
                ],
                'isKeyframe': False
            }
            annots_all.append(annots)
        return annots_all


def extract_2d(image_root, annot_root, config, to_openpose=True):
    """Extract 2D keypoints for all images in a folder"""
    if check_result(image_root, annot_root):
        return 0
    device = torch.device('cuda') \
            if torch.cuda.is_available() else torch.device('cpu')
    ext = config.pop('ext')
    detector = MMPoseDetector()
    imgnames = sorted(glob(join(image_root, '*'+ext)))
    for imgname in tqdm(imgnames, desc='{:10s}'.format(os.path.basename(annot_root))):
        base = os.path.basename(imgname).replace(ext, '')
        annotname = join(annot_root, base+'.json')
        image = cv2.imread(imgname)
        annots = detector([image])[0]
        annots['filename'] = os.sep.join(imgname.split(os.sep)[-2:])
        save_annot(annotname, annots)