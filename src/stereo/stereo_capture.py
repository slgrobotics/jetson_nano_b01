#!/usr/bin/env python3
import cv2
import os
import time

#
# Stereo vision on Nano:
# - https://chatgpt.com/s/t_69b88fead95c8191be1cacb3edff4ea2  - general advice
# - https://chatgpt.com/s/t_69b890a7d5e08191b447848349d0178b  - minimal three-script starter pack
#
# Calibration board generator: https://markhedleyjones.com/projects/calibration-checkerboard-collection
#                              https://markhedleyjones.com/media/projects/calibration-checkerboard-collection/Checkerboard-A4-70mm-3x2.pdf
#


# ===== settings =====
NUM_PAIRS = 50
INTERVAL_SEC = 2.0
FLUSH_FRAMES = 4
# ====================


def gstreamer_pipeline(sensor_id=0, width=1280, height=720, fps=30, flip_method=0):
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
        gstreamer_pipeline(sensor_id=sensor_id, width=width, height=height, fps=fps),
        cv2.CAP_GSTREAMER,
    )
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera sensor-id={sensor_id}")
    return cap


def flush_and_read(cap, n=4):
    for _ in range(n):
        cap.grab()
    return cap.read()


def main():
    width = 1280
    height = 720
    fps = 30

    out_dir = "stereo_pairs"
    left_dir = os.path.join(out_dir, "left")
    right_dir = os.path.join(out_dir, "right")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    capL = open_camera(0, width, height, fps)
    capR = open_camera(1, width, height, fps)

    print(f"Starting auto capture: {NUM_PAIRS} pairs, interval {INTERVAL_SEC}s")
    print("Hold board still before each capture...\n")

    # Warm up cameras
    for _ in range(10):
        capL.read()
        capR.read()

    pair_idx = 0

    while pair_idx < NUM_PAIRS:
        t_start = time.time()

        # Preview loop during waiting interval
        while True:
            okL, left = capL.read()
            okR, right = capR.read()

            if okL and okR:
                preview = cv2.hconcat([left, right])
                cv2.putText(
                    preview,
                    f"Next capture in {max(0, INTERVAL_SEC - (time.time() - t_start)):.1f}s | idx={pair_idx}",
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

            if time.time() - t_start >= INTERVAL_SEC:
                break

        print(f"[{pair_idx:03d}] Capturing... hold still")

        # Flush + fresh capture
        okL, left = flush_and_read(capL, FLUSH_FRAMES)
        okR, right = flush_and_read(capR, FLUSH_FRAMES)

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
