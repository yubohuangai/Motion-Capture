'''
https://kaustubh-sadekar.github.io/OmniCV-Lib/Equirectangular-to-fisheye.html
angles – List of camera [roll,pitch,yaw]
'''

import cv2
import numpy as np
import math
import time
import sys
from omnicv import fisheyeImgConv

# Import equirectangular image
equiRect = cv2.imread('/Users/yubo/data/s2/insta360_1/test/images/seq1/000000.jpg')

# Defining output shape
outShape = [1080, 1080]
f = 270
# Creating mapper object
mapper = fisheyeImgConv()

# Converting equirectangular to fisheye using Unified Camera model (UCM)
# fisheye = mapper.equirect2Fisheye_UCM(equiRect,outShape=outShape,xi=0.5)
# cv2.imshow("UCM Model Output",fisheye)
# cv2.waitKey(0)

# Converting equirectangular to fisheye using Extended UCM model
# fisheye = mapper.equirect2Fisheye_EUCM(equiRect, outShape=outShape, f=f, a_=0, b_=0, angles=[0, 0, 0])
# cv2.imshow("EUCM Model Output", fisheye)
# cv2.waitKey(0)

# Converting equirectangular to fisheye using Field Of Vide (FOV) model
# fisheye = mapper.equirect2Fisheye_FOV(equiRect, outShape=outShape, f=f, w_=0.5, angles=[0, 0, 0])
# cv2.imshow("FOV model Output", fisheye)
# cv2.waitKey(0)
#
# # Converting equirectangular to fisheye using Double Sphere (DS) model
fisheye = mapper.equirect2Fisheye_DS(equiRect, outShape=outShape, f=f, a_=0.5, xi_=0.8, angles=[0, 0, 180])
cv2.imshow("DS Model Output", fisheye)
cv2.waitKey(0)
#
# # Changing the distortion coefficient for (UCM)
# fisheye = mapper.equirect2Fisheye(equiRect, outShape=outShape, xi=0.2)
# cv2.imshow("fisheye", fisheye)
# cv2.waitKey(0)
#
# # Rotate the sphere
# fisheye = mapper.equirect2Fisheye(equiRect, outShape=outShape, angles=[0, 0, 180])
# cv2.imshow("fisheye", fisheye)
# cv2.waitKey(0)
