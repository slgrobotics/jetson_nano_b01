#!/usr/bin/env python3
import cv2
import numpy as np
import time

"""
Now the "disparity" picture started making sense, I want to improve code a bit.
 First, I want to have an option by keyboard stroke (space) to toggle the left/right camera preview. 
 This way I will estimate the FPS in disparity only mode.
 Next, I want to overlay numbers on disparity screen, showing distances in centimeters to the closest object in a cell.
 Say, the screen is divided by 10x10 cells for that purpose.

 See https://chatgpt.com/s/t_69b97c6ac4188191929289332c6f6c83
"""

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


def draw_horizontal_lines(img, step=40):
    out = img.copy()
    h, w = out.shape[:2]
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w - 1, y), (0, 255, 0), 1)
    return out


def draw_grid(img, rows=10, cols=10, color=(255, 255, 255), thickness=1):
    out = img.copy()
    h, w = out.shape[:2]

    for r in range(1, rows):
        y = int(r * h / rows)
        cv2.line(out, (0, y), (w - 1, y), color, thickness)

    for c in range(1, cols):
        x = int(c * w / cols)
        cv2.line(out, (x, 0), (x, h - 1), color, thickness)

    return out


def estimate_depth_cm_from_disparity(disparity_px, focal_px, baseline_m):
    """
    Z = f * B / d
    Returns depth in centimeters, or None if disparity is invalid.
    """
    if disparity_px <= 0:
        return None
    z_m = (focal_px * baseline_m) / disparity_px
    return z_m * 100.0


def overlay_cell_distances(
    disp_color,
    disparity,
    focal_px,
    baseline_m,
    rows=10,
    cols=10,
    min_valid_disp=1.0,
    max_depth_cm=999,
):
    """
    For each cell, find the largest valid disparity (= closest object),
    convert to distance, and overlay the distance in cm.
    """
    out = draw_grid(disp_color, rows=rows, cols=cols, color=(255, 255, 255), thickness=1)
    h, w = disparity.shape[:2]

    for r in range(rows):
        y0 = int(r * h / rows)
        y1 = int((r + 1) * h / rows)

        for c in range(cols):
            x0 = int(c * w / cols)
            x1 = int((c + 1) * w / cols)

            cell = disparity[y0:y1, x0:x1]
            valid = cell > min_valid_disp

            if not np.any(valid):
                text = "--"
            else:
                closest_disp = float(np.max(cell[valid]))   # largest disparity = closest
                depth_cm = estimate_depth_cm_from_disparity(closest_disp, focal_px, baseline_m)

                if depth_cm is None or not np.isfinite(depth_cm):
                    text = "--"
                else:
                    depth_cm = min(depth_cm, max_depth_cm)
                    text = f"{int(round(depth_cm))}"

            cx = x0 + (x1 - x0) // 2
            cy = y0 + (y1 - y0) // 2

            # black outline
            cv2.putText(
                out, text, (cx - 18, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA
            )
            # white text
            cv2.putText(
                out, text, (cx - 18, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
            )

    return out


def main():
    calib = np.load("stereo_calibration.npz")

    mapLx = calib["mapLx"]
    mapLy = calib["mapLy"]
    mapRx = calib["mapRx"]
    mapRy = calib["mapRy"]

    # Use projection matrix / calibration to estimate focal length and baseline
    PL = calib["PL"]
    T = calib["T"]

    width = int(calib["image_width"])
    height = int(calib["image_height"])
    fps = 30

    # Stereo geometry
    focal_px = float(PL[0, 0])         # rectified focal length in pixels
    baseline_m = float(np.linalg.norm(T))

    print(f"Using focal length: {focal_px:.2f} px")
    print(f"Using baseline    : {baseline_m * 100.0:.2f} cm")

    capL = open_camera(0, width, height, fps)
    capR = open_camera(1, width, height, fps)

    min_disp = 0
    num_disp = 16 * 6   # 96, multiple of 16
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

    show_preview = True

    # FPS measurement
    last_time = time.time()
    fps_filtered = 0.0

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

        # Disparity visualization
        disp_vis = (disparity - min_disp) / num_disp
        disp_vis = np.clip(disp_vis, 0, 1)
        disp_vis = (disp_vis * 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

        # Overlay 10x10 closest-object distances in cm
        disp_color = overlay_cell_distances(
            disp_color,
            disparity,
            focal_px=focal_px,
            baseline_m=baseline_m,
            rows=10,
            cols=10,
            min_valid_disp=1.0,
            max_depth_cm=999,
        )

        # FPS update
        now = time.time()
        dt = now - last_time
        last_time = now
        fps_now = 1.0 / dt if dt > 0 else 0.0
        fps_filtered = 0.9 * fps_filtered + 0.1 * fps_now if fps_filtered > 0 else fps_now

        mode_text = "preview: ON" if show_preview else "preview: OFF"

        cv2.putText(
            disp_color,
            f"FPS {fps_filtered:.1f} | {mode_text} | space=toggle preview | q=quit",
            (20, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            disp_color,
            f"f={focal_px:.0f}px B={baseline_m*100:.1f}cm",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            disp_color,
            msg,
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if show_preview:
            rect_preview = cv2.hconcat([
                draw_horizontal_lines(left_rect, 40),
                draw_horizontal_lines(right_rect, 40)
            ])
            cv2.putText(
                rect_preview,
                msg,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Rectified Pair", rect_preview)
        else:
            try:
                cv2.destroyWindow("Rectified Pair")
            except cv2.error:
                pass

        cv2.imshow("Disparity", disp_color)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord(" "):
            show_preview = not show_preview

    capL.release()
    capR.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
