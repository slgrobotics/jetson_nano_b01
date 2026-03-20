#!/usr/bin/env python3

"""
Stereo camera calibration and rectification pipeline.

This script processes a dataset of stereo image pairs to compute intrinsic
parameters for each camera and extrinsic parameters between them.

It detects chessboard corners in left/right images, filters valid pairs,
and performs:
- Monocular calibration for each camera
- Stereo calibration to estimate relative pose (R, T)
- Stereo rectification and projection matrix computation
- Generation of undistortion and rectification maps

Only image pairs with successful corner detection in both views are used,
improving calibration robustness.

The resulting calibration data (intrinsics, distortion, extrinsics,
rectification transforms, and remap grids) is saved to a .npz file for
later use in disparity and 3D reconstruction.

Key features:
- Automatic filtering of invalid stereo pairs
- Subpixel corner refinement for accuracy
- Full calibration + rectification pipeline
- Visual feedback for accepted pairs

Intended use:
- Producing stereo calibration files for depth estimation pipelines
- Preparing rectification maps for real-time disparity computation
- Ensuring accurate geometric alignment between stereo cameras
"""

import cv2
import glob
import numpy as np
import os

from config import Stereo, Calib

#
# Stereo vision on Nano:
# - https://chatgpt.com/s/t_69b88fead95c8191be1cacb3edff4ea2  - general advice
# - https://chatgpt.com/s/t_69b890a7d5e08191b447848349d0178b  - minimal three-script starter pack
# 
# Calibration board generator: https://markhedleyjones.com/projects/calibration-checkerboard-collection
#                              https://markhedleyjones.com/media/projects/calibration-checkerboard-collection/Checkerboard-A4-30mm-8x6.pdf
# 

"""
Capture a new set with:
 - board close, medium, and farther
 - strong tilt left/right/up/down
 - board near all four corners
 - fewer nearly identical poses
 - no blur
 - no reflections
 - rigid flat board
A set of 15 very diverse images is often better than 23 repetitive ones.
"""

def main():

    print(f"IP: Looking for image pairs in '{Calib.PAIR_DIR}'")

    left_images = sorted(glob.glob(os.path.join(Calib.PAIR_DIR, "left", Calib.IMAGE_EXT)))
    right_images = sorted(glob.glob(os.path.join(Calib.PAIR_DIR, "right", Calib.IMAGE_EXT)))

    if len(left_images) == 0 or len(right_images) == 0:
        raise RuntimeError("No stereo images found")

    if len(left_images) != len(right_images):
        raise RuntimeError("Left/right image count mismatch")

    # 3D points in chessboard coordinate system
    objp = np.zeros((Stereo.CHESSBOARD_SIZE[0] * Stereo.CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:Stereo.CHESSBOARD_SIZE[0], 0:Stereo.CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= Stereo.SQUARE_SIZE

    objpoints = []
    imgpointsL = []
    imgpointsR = []

    image_size = None

    print(f"OK: Found {len(left_images)} candidate stereo pairs")
    print("...analyzing and displaying pairs...", flush=True)

    criteria_subpix = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    good_pairs = 0

    for left_path, right_path in zip(left_images, right_images):
        imgL = cv2.imread(left_path)
        imgR = cv2.imread(right_path)

        if imgL is None or imgR is None:
            print(f"FYI: Skipping unreadable pair:\n  {left_path}\n  {right_path}")
            continue

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        image_size = grayL.shape[::-1]

        retL, cornersL = cv2.findChessboardCorners(grayL, Stereo.CHESSBOARD_SIZE, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, Stereo.CHESSBOARD_SIZE, None)

        if retL and retR:
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria_subpix)
            cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria_subpix)

            objpoints.append(objp)
            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)
            good_pairs += 1

            visL = imgL.copy()
            visR = imgR.copy()
            cv2.drawChessboardCorners(visL, Stereo.CHESSBOARD_SIZE, cornersL, retL)
            cv2.drawChessboardCorners(visR, Stereo.CHESSBOARD_SIZE, cornersR, retR)
            preview = cv2.hconcat([visL, visR])
            cv2.imshow("FYI: Accepted Pair", preview)
            cv2.waitKey(150)
        else:
            print(f"FYI: Rejected pair:\n  {left_path}\n  {right_path}")

    cv2.destroyAllWindows()

    if good_pairs < 30:
        raise RuntimeError(f"Not enough good pairs for calibration: {good_pairs} - need at least 30")

    print(f"IP: Using {good_pairs} good stereo pairs")
    print("...thinking...", flush=True)

    # Calibrate each camera individually
    retL, K1, D1, rvecsL, tvecsL = cv2.calibrateCamera(
        objpoints, imgpointsL, image_size, None, None
    )
    retR, K2, D2, rvecsR, tvecsR = cv2.calibrateCamera(
        objpoints, imgpointsR, image_size, None, None
    )

    print(f"FYI: Mono reprojection errors:  left: {retL}  right: {retR}")
    print("...thinking...", flush=True)

    # Stereo calibration
    stereo_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-5,
    )

    flags = cv2.CALIB_FIX_INTRINSIC

    retStereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpointsL,
        imgpointsR,
        K1,
        D1,
        K2,
        D2,
        image_size,
        criteria=stereo_criteria,
        flags=flags,
    )

    print(f"FYI: Stereo reprojection error: {retStereo}")
    print(f"FYI: Baseline T (meters if Stereo.SQUARE_SIZE is meters): {T.ravel()}")
    print("...thinking...", flush=True)

    # Rectification
    RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T, alpha=0
    )

    mapLx, mapLy = cv2.initUndistortRectifyMap(
        K1, D1, RL, PL, image_size, cv2.CV_32FC1
    )
    mapRx, mapRy = cv2.initUndistortRectifyMap(
        K2, D2, RR, PR, image_size, cv2.CV_32FC1
    )

    np.savez(
        Calib.CALIBRATION_FILE,
        K1=K1, D1=D1,
        K2=K2, D2=D2,
        R=R, T=T,
        E=E, F=F,
        RL=RL, RR=RR,
        PL=PL, PR=PR,
        Q=Q,
        mapLx=mapLx, mapLy=mapLy,
        mapRx=mapRx, mapRy=mapRy,
        image_width=image_size[0],
        image_height=image_size[1],
    )

    print(f"OK: Saved calibration to {Calib.CALIBRATION_FILE}")


if __name__ == "__main__":
    main()
