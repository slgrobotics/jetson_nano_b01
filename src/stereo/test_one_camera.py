#!/usr/bin/env python3

"""
@brief
Minimal GStreamer camera open test for Jetson Nano.

This script attempts to initialize a CSI camera using the nvargus GStreamer
pipeline and reports whether the camera was successfully opened.

It uses resolution and frame rate parameters from the shared configuration.

Key features:
- Simple pipeline-based camera initialization
- Quick verification of camera availability and GStreamer setup

Intended use:
- Sanity check for CSI camera and nvargus pipeline
- Debugging camera access issues before running full applications
"""

#
# Stereo vision on Nano:
# - https://chatgpt.com/s/t_69b88fead95c8191be1cacb3edff4ea2  - general advice
# - https://chatgpt.com/s/t_69b890a7d5e08191b447848349d0178b  - minimal three-script starter pack
# 
# Calibration board generator: 
# - https://markhedleyjones.com/projects/calibration-checkerboard-collection
# - https://markhedleyjones.com/media/projects/calibration-checkerboard-collection/Checkerboard-A4-30mm-8x6.pdf
# 

import cv2
from config import Camera

pipe = (
    "nvarguscamerasrc sensor-id=0 ! "
    f"video/x-raw(memory:NVMM), width=(int){Camera.WIDTH}, height=(int){Camera.HEIGHT}, "
    f"format=(string)NV12, framerate=(fraction){Camera.FPS}/1 ! "
    "nvvidconv ! video/x-raw, format=(string)BGRx ! "
    "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
)
cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
print(cap.isOpened())
