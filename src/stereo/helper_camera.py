#!/usr/bin/env python3

import cv2


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
