#!/usr/bin/env python3

"""
@brief
Stereo image pair capture tool for calibration on Jetson Nano.

This script captures synchronized image pairs from two CSI cameras at fixed
time intervals and saves them to disk for stereo calibration.

During operation, a live preview is displayed. Before each capture, a short
visual cue (red center rectangle) is shown to prompt the user to remain still,
improving capture consistency and calibration accuracy.

Key features:
- Dual-camera synchronized capture using GStreamer (nvargus)
- Automatic timed acquisition of N stereo pairs
- Pre-capture stabilization cue (visual flash overlay)
- Frame flushing to reduce motion artifacts
- Organized output into left/right image folders

Intended use:
- Collecting high-quality stereo datasets for OpenCV calibration
- Ensuring varied board poses while minimizing motion blur
- Simple, repeatable capture workflow on embedded platforms

Move the checkerboard through many positions and angles, filling the frame, tilting and rotating it,
 and covering different depths—holding it steady during each capture.
"""

import cv2
import os
import time

from config import Camera, Stereo, Calib

#
# Stereo vision on Nano:
# - https://chatgpt.com/s/t_69b88fead95c8191be1cacb3edff4ea2  - general advice
# - https://chatgpt.com/s/t_69b890a7d5e08191b447848349d0178b  - minimal three-script starter pack
#
# Calibration board generator:
#  - https://markhedleyjones.com/projects/calibration-checkerboard-collection
#  - https://markhedleyjones.com/media/projects/calibration-checkerboard-collection/Checkerboard-A4-30mm-8x6.pdf
#


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


def open_camera(sensor_id, width, height, fps):
    cap = cv2.VideoCapture(
        gstreamer_pipeline(sensor_id=sensor_id, width=width, height=height, fps=fps, flip_method=0),
        cv2.CAP_GSTREAMER,
    )
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera sensor-id={sensor_id}")
    return cap


def flush_and_read(cap, n=4):
    for _ in range(n):
        cap.grab()
    return cap.read()


def draw_flash_border(img, color=(0, 0, 255), thickness=4):
    """
    Draw a centered rectangle sized to 1/5 of the image width and height.
    """
    out = img.copy()
    h, w = out.shape[:2]

    rw = w - thickness
    rh = h - thickness

    x0 = (w - rw) // 2
    y0 = (h - rh) // 2
    x1 = x0 + rw
    y1 = y0 + rh

    cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
    return out


def main():
    out_dir = Calib.PAIR_DIR
    left_dir = os.path.join(out_dir, "left")
    right_dir = os.path.join(out_dir, "right")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    capL = open_camera(Camera.LEFT, Camera.WIDTH, Camera.HEIGHT, Camera.FPS)
    capR = open_camera(Camera.RIGHT, Camera.WIDTH, Camera.HEIGHT, Camera.FPS)

    print(f"Starting auto capture: {Calib.NUM_PAIRS} pairs, interval {Calib.INTERVAL_SEC}s")
    print("Hold board still before each capture...\n")

    # Warm up cameras
    for _ in range(10):
        capL.read()
        capR.read()

    pair_idx = 0

    while pair_idx < Calib.NUM_PAIRS:
        t_start = time.time()

        # Preview loop during waiting interval
        while True:
            okL, left = capL.read()
            okR, right = capR.read()

            if okL and okR:
                # flip horizontally for preview, easier to see yourself in the mirror
                left_mirror = cv2.flip(left, 1)
                right_mirror = cv2.flip(right, 1)

                preview = cv2.hconcat([left_mirror, right_mirror])
                cv2.putText(
                    preview,
                    f"Next capture in {max(0, Calib.INTERVAL_SEC - (time.time() - t_start)):.1f}s | idx={pair_idx}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Stereo Preview", preview)

            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                print("Aborted by user")
                capL.release()
                capR.release()
                cv2.destroyAllWindows()
                return

            if time.time() - t_start >= Calib.INTERVAL_SEC:
                break

        print(f"[{pair_idx:03d}] Flashing warning rectangle... stand still")

        # Flash warning rectangle before sampling
        flash_start = time.time()
        while time.time() - flash_start < Calib.FLASH_SEC:
            okL, left = capL.read()
            okR, right = capR.read()

            if okL and okR:
                # flip horizontally for preview, easier to see yourself in the mirror
                left_mirror = cv2.flip(left, 1)
                right_mirror = cv2.flip(right, 1)

                left_flash = draw_flash_border(left_mirror)
                right_flash = draw_flash_border(right_mirror)
                preview = cv2.hconcat([left_flash, right_flash])

                cv2.putText(
                    preview,
                    f"Capturing {pair_idx} of {Calib.NUM_PAIRS}... HOLD STILL",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Stereo Preview", preview)

            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                print("Aborted by user")
                capL.release()
                capR.release()
                cv2.destroyAllWindows()
                return

        print(f"[{pair_idx:03d}] Capturing now")

        # Flush + fresh capture
        okL, left = flush_and_read(capL, Calib.FLUSH_FRAMES)
        okR, right = flush_and_read(capR, Calib.FLUSH_FRAMES)

        if not okL or not okR:
            print("Capture failed, skipping")
            continue

        left_path = os.path.join(left_dir, f"left_{pair_idx:04d}.png")
        right_path = os.path.join(right_dir, f"right_{pair_idx:04d}.png")

        cv2.imwrite(left_path, left)
        cv2.imwrite(right_path, right)

        print(f"Saved pair {pair_idx:04d}")

        pair_idx += 1

    print("\nDone capturing.")

    capL.release()
    capR.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
