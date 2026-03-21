#!/usr/bin/env python3

# =====================================================
# GStreamer-based camera driver for Jetson Nano CSI cameras.
#
# This module provides a lightweight wrapper around OpenCV VideoCapture using
# the nvargus GStreamer pipeline, enabling efficient access to CSI cameras.
#
# It supports opening individual cameras or synchronized stereo pairs with
# configurable resolution, frame rate, and flip settings.
#
# Key features:
# - Encapsulated GStreamer pipeline generation
# - Simple API for single or stereo camera initialization
# - Integration with shared configuration parameters
# - Error handling for camera availability
#
# Intended use:
# - Reusable camera interface across capture, calibration, and inference scripts
# - Consistent camera setup in stereo vision pipelines
# - Simplifying CSI camera access on embedded platforms
# =====================================================

import cv2

from config import Camera

class CameraDriver:
    @staticmethod
    def gstreamer_pipeline(sensor_id, width, height, fps, flip_method):
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
            f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=(string)BGR ! appsink drop=true sync=false max-buffers=1"
        )

    @staticmethod
    def open_camera(sensor_id, width, height, fps, flip_method=0):
        cap = cv2.VideoCapture(
            CameraDriver.gstreamer_pipeline(
                sensor_id=sensor_id,
                width=width,
                height=height,
                fps=fps,
                flip_method=flip_method,
            ),
            cv2.CAP_GSTREAMER,
        )
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera sensor-id={sensor_id}")
        return cap

    @staticmethod
    def open_stereo_cameras(width=Camera.WIDTH, height=Camera.HEIGHT, fps=Camera.FPS,
                            left_id=Camera.LEFT, right_id=Camera.RIGHT,
                            flip_method=0):
        capL = CameraDriver.open_camera(left_id, width, height, fps, flip_method)
        capR = CameraDriver.open_camera(right_id, width, height, fps, flip_method)
        return capL, capR
