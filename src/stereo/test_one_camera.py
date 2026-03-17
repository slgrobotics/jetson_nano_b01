#!/usr/bin/env python3

import cv2

#
# Stereo vision on Nano:
# - https://chatgpt.com/s/t_69b88fead95c8191be1cacb3edff4ea2  - general advice
# - https://chatgpt.com/s/t_69b890a7d5e08191b447848349d0178b  - minimal three-script starter pack
# 
# Calibration board generator: https://markhedleyjones.com/projects/calibration-checkerboard-collection
#                              https://markhedleyjones.com/media/projects/calibration-checkerboard-collection/Checkerboard-A4-70mm-3x2.pdf
# 

pipe = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, "
    "format=(string)NV12, framerate=(fraction)30/1 ! "
    "nvvidconv ! video/x-raw, format=(string)BGRx ! "
    "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
)
cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
print(cap.isOpened())
