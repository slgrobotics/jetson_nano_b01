#!/usr/bin/env python3
import cv2
import numpy as np


def gstreamer_pipeline(sensor_id=0, width=1280, height=720, fps=30, flip_method=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink drop=true sync=false"
    )


def open_camera(sensor_id, width, height, fps):
    cap = cv2.VideoCapture(
        gstreamer_pipeline(sensor_id=sensor_id, width=width, height=height, fps=fps),
        cv2.CAP_GSTREAMER,
    )
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera sensor-id={sensor_id}")
    return cap


def draw_horizontal_lines(img, step=40):
    out = img.copy()
    h, w = out.shape[:2]
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w - 1, y), (0, 255, 0), 1)
    return out


def main():
    calib = np.load("stereo_calibration.npz")

    mapLx = calib["mapLx"]
    mapLy = calib["mapLy"]
    mapRx = calib["mapRx"]
    mapRy = calib["mapRy"]

    width = int(calib["image_width"])
    height = int(calib["image_height"])
    fps = 30

    # Try swapping these if disparity is mostly invalid
    capL = open_camera(0, width, height, fps)
    capR = open_camera(1, width, height, fps)

    min_disp = 0
    num_disp = 16 * 6  # 96, must be multiple of 16
    block_size = 7

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 1 * block_size * block_size,
        P2=32 * 1 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    while True:
        okL, left = capL.read()
        okR, right = capR.read()
        if not okL or not okR:
            continue

        left_rect = cv2.remap(left, mapLx, mapLy, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, mapRx, mapRy, cv2.INTER_LINEAR)

        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        valid = disparity > min_disp
        if np.any(valid):
            msg = (
                f"valid px={np.count_nonzero(valid)} "
                f"min={np.min(disparity[valid]):.2f} "
                f"max={np.max(disparity[valid]):.2f} "
                f"mean={np.mean(disparity[valid]):.2f}"
            )
        else:
            msg = "No valid disparity pixels. Check rectification or swap left/right cameras."

        rect_preview = cv2.hconcat([
            draw_horizontal_lines(left_rect, 40),
            draw_horizontal_lines(right_rect, 40)
        ])
        cv2.putText(rect_preview, msg, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        disp_vis = (disparity - min_disp) / num_disp
        disp_vis = np.clip(disp_vis, 0, 1)
        disp_vis = (disp_vis * 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

        cv2.imshow("Rectified Pair", rect_preview)
        cv2.imshow("Disparity", disp_color)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

