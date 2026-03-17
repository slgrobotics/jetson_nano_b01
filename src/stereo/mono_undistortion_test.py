#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import os

data = np.load("stereo_calibration.npz")
K1, D1 = data["K1"], data["D1"]
K2, D2 = data["K2"], data["D2"]
w, h = int(data["image_width"]), int(data["image_height"])

left_images = sorted(glob.glob(os.path.join("stereo_pairs", "left", "*.png")))
right_images = sorted(glob.glob(os.path.join("stereo_pairs", "right", "*.png")))

imgL = cv2.imread(left_images[0])
imgR = cv2.imread(right_images[0])

mapLx, mapLy = cv2.initUndistortRectifyMap(K1, D1, None, K1, (w, h), cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(K2, D2, None, K2, (w, h), cv2.CV_32FC1)

uL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
uR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)

cv2.imshow("mono undistort left", uL)
cv2.imshow("mono undistort right", uR)
cv2.waitKey(0)
cv2.destroyAllWindows()
