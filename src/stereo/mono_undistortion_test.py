#!/usr/bin/env python3

"""
Monocular undistortion sanity check for stereo cameras.

This script loads intrinsic calibration parameters for both cameras and
applies undistortion to a sample stereo image pair.

The resulting images are displayed to visually verify that lens distortion
has been correctly removed (e.g., straight lines appear straight, no warping).

Key features:
- Loads calibration data from saved .npz file
- Applies OpenCV undistortion maps to left and right images
- Displays side-by-side results for visual inspection

Intended use:
- Quick validation of intrinsic calibration quality
- Detecting incorrect calibration parameters or resolution mismatch
- Verifying camera models before stereo rectification
"""

import cv2
import numpy as np
import glob
import os

from config import Calib

try:
    calib = np.load(Calib.CALIBRATION_FILE)
except FileNotFoundError:
    raise RuntimeError(f"Calibration file '{Calib.CALIBRATION_FILE}' not found")

K1, D1 = calib["K1"], calib["D1"]
K2, D2 = calib["K2"], calib["D2"]
w, h = int(calib["image_width"]), int(calib["image_height"])

left_images = sorted(glob.glob(os.path.join("stereo_pairs", "left", Calib.IMAGE_EXT)))
right_images = sorted(glob.glob(os.path.join("stereo_pairs", "right", Calib.IMAGE_EXT)))

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
