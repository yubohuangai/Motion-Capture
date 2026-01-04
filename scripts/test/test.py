'''
Created on Jan 26, 2018
FilePath: test_omnicv.py
'''

import cv2
import numpy as np
import math
import time
import sys
from omnicv import fisheyeImgConv


equiRect = cv2.imread("/Users/yubo/data/s2/insta360_1/test/images/seq1/000000.jpg")  # 5,760 * 2,880
outShape = [1080, 1080]
mapper = fisheyeImgConv()
# Converting equirectangular to fisheye using Unified Camera model (UCM)
# fisheye = mapper.equirect2Fisheye_UCM(equiRect,outShape=outShape,xi=0.5)
# cv2.imshow("UCM Model Output",fisheye)

# Converting equirectangular to fisheye using Double Sphere (DS) model
fisheye = mapper.equirect2Fisheye_DS(equiRect,outShape=outShape,f=100,a_=0.5,xi_=0,angles=[0,0,0])
cv2.imshow("DS Model Output",fisheye)

# Converting equirectangular to fisheye using Field Of Vide (FOV) model
# fisheye = mapper.equirect2Fisheye_FOV(equiRect,outShape=outShape,f=40,w_=0.5,angles=[0,0,0])
# cv2.imshow("FOV model Output",fisheye)



cv2.waitKey(0)
