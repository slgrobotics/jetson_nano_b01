#!/usr/bin/env python3
import cv2
import numpy as np
import socket
import struct
import time

"""
@brief
UDP-based stereo perception server for Jetson Nano.

This script captures synchronized frames from two CSI cameras, applies stereo
rectification using precomputed calibration, computes a disparity map via
StereoSGBM, and converts it into a sparse 3D point representation.

The image is divided into a fixed grid (e.g., 10x10 cells). For each cell,
a representative 3D point is selected based on high-percentile disparity
(closest obstacle), filtered for validity and range, and converted into a
ROS-compatible coordinate frame.

Each frame is serialized into a compact binary packet and transmitted over
UDP to a remote client (e.g., a ROS 2 node), which can reconstruct and publish
the data as a PointCloud2 message.

Key characteristics:
- Low-latency, connectionless UDP streaming (no delivery guarantees)
- Sparse, obstacle-focused point cloud (one point per grid cell)
- Designed for constrained platforms (Jetson Nano)
- Suitable for real-time perception prototyping and ROS 2 integration

Intended use:
- Lightweight stereo depth server feeding external ROS 2 processing
- Obstacle detection and navigation experiments
- Rapid iteration without full ROS stack on embedded device

See https://chatgpt.com/s/t_69b986e63f508191a5de89865356377f

Receiver expectations:
 - The ROS 2 node later should:
 - recvfrom()
 - unpack header
 - unpack point_count point records
 - publish PointCloud2

The cloud will be sparse but immediately useful.

"""

# =========================
# Configuration
# =========================
UDP_IP = "192.168.1.100"   # dev machine (ROS2 node) IP
UDP_PORT = 5005

GRID_ROWS = 10
GRID_COLS = 10

MIN_VALID_DISP = 1.0
MAX_RANGE_M = 5.0
MIN_CONFIDENCE = 0.02   # valid fraction in cell

SHOW_PREVIEW = False
SHOW_DISPLAY = True   # display ON by default

HEADER_STRUCT = struct.Struct("<4sBBBBIQHH")
HEADER_MAGIC = b"SPC2"
HEADER_VERSION = 1

POINT_STRUCT = struct.Struct("<ffffHH")


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


def make_raw_disparity_view(disparity, min_disp, num_disp):
    disp_vis = (disparity - min_disp) / num_disp
    disp_vis = np.clip(disp_vis, 0, 1)
    disp_vis = (disp_vis * 255).astype(np.uint8)
    return cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)


def estimate_depth_cm_from_disparity(disparity_px, focal_px, baseline_m):
    if disparity_px <= 0:
        return None
    z_m = (focal_px * baseline_m) / disparity_px
    return z_m * 100.0


def overlay_cell_distances(
    img,
    disparity,
    focal_px,
    baseline_m,
    rows=10,
    cols=10,
    min_valid_disp=1.0,
    max_depth_cm=999,
):
    out = draw_grid(img, rows=rows, cols=cols, color=(255, 255, 255), thickness=1)
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
                closest_disp = float(np.max(cell[valid]))
                depth_cm = estimate_depth_cm_from_disparity(
                    closest_disp, focal_px, baseline_m
                )

                if depth_cm is None or not np.isfinite(depth_cm):
                    text = "--"
                else:
                    depth_cm = min(depth_cm, max_depth_cm)
                    text = f"{int(round(depth_cm))}"

            cx = x0 + (x1 - x0) // 2
            cy = y0 + (y1 - y0) // 2

            cv2.putText(
                out, text, (cx - 18, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                out, text, (cx - 18, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
            )

    return out


def cam_to_ros(x_cam, y_cam, z_cam):
    """
    OpenCV stereo camera coordinates:
      x right, y down, z forward

    ROS-like convention:
      x forward, y left, z up
    """
    x_ros = z_cam
    y_ros = -x_cam
    z_ros = -y_cam
    return x_ros, y_ros, z_ros


def extract_sparse_points(disparity, points_3d, rows, cols, min_valid_disp, max_range_m):
    """
    One representative point per cell.

    Strategy:
    - valid disparity mask in cell
    - pick 95th percentile disparity
    - choose actual pixel nearest that target disparity
    - emit XYZ + confidence + grid row/col
    """
    h, w = disparity.shape[:2]
    points = []

    for r in range(rows):
        y0 = int(r * h / rows)
        y1 = int((r + 1) * h / rows)

        for c in range(cols):
            x0 = int(c * w / cols)
            x1 = int((c + 1) * w / cols)

            cell_disp = disparity[y0:y1, x0:x1]
            valid_mask = np.isfinite(cell_disp) & (cell_disp > min_valid_disp)

            valid_fraction = float(np.count_nonzero(valid_mask)) / float(cell_disp.size)
            if valid_fraction < MIN_CONFIDENCE:
                continue

            valid_values = cell_disp[valid_mask]
            target_disp = float(np.percentile(valid_values, 95))

            ys, xs = np.where(valid_mask)
            disp_candidates = cell_disp[ys, xs]
            best_idx = int(np.argmin(np.abs(disp_candidates - target_disp)))

            py = y0 + int(ys[best_idx])
            px = x0 + int(xs[best_idx])

            xyz = points_3d[py, px]
            x_cam, y_cam, z_cam = float(xyz[0]), float(xyz[1]), float(xyz[2])

            if not np.isfinite(x_cam) or not np.isfinite(y_cam) or not np.isfinite(z_cam):
                continue
            if z_cam <= 0.0 or z_cam > max_range_m:
                continue

            x_ros, y_ros, z_ros = cam_to_ros(x_cam, y_cam, z_cam)
            points.append((x_ros, y_ros, z_ros, valid_fraction, r, c))

    return points


def pack_packet(seq, rows, cols, points):
    stamp_ns = int(time.time() * 1e9)  # Python 3.6 compatible

    header = HEADER_STRUCT.pack(
        HEADER_MAGIC,
        HEADER_VERSION,
        rows,
        cols,
        0,              # reserved
        seq,
        stamp_ns,
        len(points),
        0,              # reserved2
    )

    payload = bytearray(header)
    for x, y, z, confidence, row, col in points:
        payload += POINT_STRUCT.pack(x, y, z, confidence, row, col)

    return bytes(payload)


def main():
    calib = np.load("stereo_calibration.npz")

    mapLx = calib["mapLx"]
    mapLy = calib["mapLy"]
    mapRx = calib["mapRx"]
    mapRy = calib["mapRy"]
    Q = calib["Q"]
    PL = calib["PL"]
    T = calib["T"]

    width = int(calib["image_width"])
    height = int(calib["image_height"])
    fps = 30

    focal_px = float(PL[0, 0])
    baseline_m = float(np.linalg.norm(T))

    capL = open_camera(0, width, height, fps)
    capR = open_camera(1, width, height, fps)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    min_disp = 0
    num_disp = 16 * 6
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

    seq = 0
    last_time = time.time()
    fps_filtered = 0.0
    show_display = SHOW_DISPLAY

    try:
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
            points_3d = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=False)

            points = extract_sparse_points(
                disparity,
                points_3d,
                rows=GRID_ROWS,
                cols=GRID_COLS,
                min_valid_disp=MIN_VALID_DISP,
                max_range_m=MAX_RANGE_M,
            )

            packet = pack_packet(seq, GRID_ROWS, GRID_COLS, points)
            sock.sendto(packet, (UDP_IP, UDP_PORT))

            now = time.time()
            dt = now - last_time
            last_time = now
            fps_now = 1.0 / dt if dt > 0 else 0.0
            fps_filtered = 0.9 * fps_filtered + 0.1 * fps_now if fps_filtered > 0 else fps_now

            print(
                f"seq={seq} points={len(points)} udp_bytes={len(packet)} fps={fps_filtered:.2f}",
                end="\r",
                flush=True,
            )

            if SHOW_PREVIEW:
                rect_preview = cv2.hconcat([
                    draw_horizontal_lines(left_rect, 40),
                    draw_horizontal_lines(right_rect, 40)
                ])
                cv2.imshow("Rectified Pair", rect_preview)

            if show_display:
                disp_view = make_raw_disparity_view(disparity, min_disp, num_disp)
                disp_view = overlay_cell_distances(
                    disp_view,
                    disparity,
                    focal_px=focal_px,
                    baseline_m=baseline_m,
                    rows=GRID_ROWS,
                    cols=GRID_COLS,
                    min_valid_disp=MIN_VALID_DISP,
                    max_depth_cm=int(MAX_RANGE_M * 100.0),
                )

                cv2.putText(
                    disp_view,
                    f"seq={seq} points={len(points)} fps={fps_filtered:.2f}",
                    (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    disp_view,
                    "space=toggle display | q=quit",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    disp_view,
                    f"f={focal_px:.0f}px B={baseline_m*100:.1f}cm",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("Disparity", disp_view)
            else:
                try:
                    cv2.destroyWindow("Disparity")
                except cv2.error:
                    pass

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord(" "):
                show_display = not show_display

            seq += 1

    finally:
        capL.release()
        capR.release()
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
