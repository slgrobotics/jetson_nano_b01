#!/usr/bin/env python3

# =====================================================
# UDP-based stereo perception server for Jetson Nano.
#
#    cd ~/jetson_nano_b01/src/stereo
#    ./disparity_streamer.py --udp-ip 192.168.1.100 --udp-port 5005 [--no-display --show-preview --grid-size 10 --min-confidence 0.02]
#
# This script captures synchronized frames from two CSI cameras, applies stereo
# rectification using precomputed calibration, computes a disparity map via
# StereoSGBM, and converts it into a sparse 3D point representation.
#
# The image is divided into a fixed grid (e.g., 10x10 cells). For each cell,
# a representative 3D point is selected based on high-percentile disparity
# (closest obstacle), filtered for validity and range, and converted into a
# ROS-compatible coordinate frame.
#
# Each frame is serialized into a compact binary packet and transmitted over
# UDP to a remote client (e.g., a ROS 2 node), which can reconstruct and publish
# the data as a PointCloud2 message.
#
# Key characteristics:
# - Low-latency, connectionless UDP streaming (no delivery guarantees)
# - Sparse, obstacle-focused point cloud (one point per grid cell)
# - Designed for constrained platforms (Jetson Nano)
# - Suitable for real-time perception prototyping and ROS 2 integration
#
# Intended use:
# - Lightweight stereo depth server feeding external ROS 2 processing
# - Obstacle detection and navigation experiments
# - Rapid iteration without full ROS stack on embedded device
#
# Receiver (ROS2 node) expectations:
# - recvfrom()
# - unpack header
# - unpack point_count point records
# - publish PointCloud2
# =====================================================

import argparse
import socket
import struct
import time
import os
import sys
import threading

import cv2
import numpy as np

from helper_tcp_server import LatestFrameBuffer, resize_to_fit, encode_jpeg
from config import Stereo, Streamer, Calib
from helper_camera import CameraDriver

# Add the parent directory of this file to Python’s import search path.
# This is a bit hacky, but it's the simplest thing to do:
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tcp_helpers import recv_message, send_json, send_json_with_jpeg

# ==============================================
# Protocol configuration - must match ROS2 node
# ==============================================

HEADER_STRUCT = struct.Struct("<4sBBBBIQHH")
HEADER_MAGIC = b"SPC2"
HEADER_VERSION = 1

POINT_STRUCT = struct.Struct("<ffffHH")


def parse_args():
    parser = argparse.ArgumentParser(description="Stereo UDP disparity server")

    parser.add_argument(
        "--udp-ip",
        type=str,
        default="192.168.1.100",
        help="Destination IP address for UDP packets",
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=5005,
        help="Destination UDP port",
    )
    parser.add_argument(
        "--show-display",
        dest="show_display",
        action="store_true",
        default=True,
        help="Enable disparity display (default: enabled)",
    )
    parser.add_argument(
        "--no-display",
        dest="show_display",
        action="store_false",
        help="Disable disparity display",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        default=False,
        help="Show rectified stereo preview (default: off)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=10,
        help="Grid size NxN for sparse sampling (default: 10)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.02,
        help="Minimum valid pixel fraction per grid cell (default: 0.02)",
    )

    parser.add_argument(
        "--close-cutout-factor",
        type=float,
        default=1.0,
        help="Larger -> keep closer objects, larger disparity search range. Smaller -> near cutoff moves farther away.",
    )

    parser.add_argument(
        "--far-smoothing-factor",
        type=float,
        default=1.0,
        help="Larger -> smoother disparity, less far objects detail. Smaller -> sharper detail, more noise, more far detail.",
    )

    parser.add_argument(
        "--tcp-image-port",
        type=int,
        default=5006,
        help="TCP port for on-demand JPEG image serving",
    )

    parser.add_argument(
        "--disable-tcp-image-server",
        dest="enable_tcp_image_server",
        action="store_false",
        default=True,
        help="Disable TCP server for on-demand JPEG preview images",
    )

    return parser.parse_args()


def draw_preview_grid(img, step=40):
    out = img.copy()
    h, w = out.shape[:2]

    # horizontal lines
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w - 1, y), (0, 255, 0), 1)

    # vertical center line
    cx = w // 2
    cv2.line(out, (cx, 0), (cx, h - 1), (0, 255, 0), 1)

    return out


def draw_overlay_grid(img, rows=10, cols=10, color=(255, 255, 255), thickness=1):
    out = img.copy()
    h, w = out.shape[:2]

    for r in range(1, rows):
        y = int(r * h / rows)
        cv2.line(out, (0, y), (w - 1, y), color, thickness)

    for c in range(1, cols):
        x = int(c * w / cols)
        cv2.line(out, (x, 0), (x, h - 1), color, thickness)

    return out


def make_valid_disparity_mask(disparity, min_valid_disp, invalid_left_cols, invalid_right_cols=0):
    valid = np.isfinite(disparity) & (disparity > min_valid_disp)

    if invalid_left_cols > 0:
        valid[:, :invalid_left_cols] = False

    if invalid_right_cols > 0:
        valid[:, -invalid_right_cols:] = False

    return valid

def derive_sgbm_params(
    close_cutout_factor: float = 1.0,
    far_smoothing_factor: float = 1.0,
):
    # ==============================================
    # Human-friendly mapping to StereoSGBM parameters.
    #
    # close_cutout_factor:
    #     Larger -> keep closer objects, larger disparity search range.
    #     Smaller -> near cutoff moves farther away.
    #
    # far_smoothing_factor:
    #     Larger -> smoother disparity, less detail.
    #     Smaller -> sharper detail, more noise.
    # ==============================================

    # Clamp to sane ranges
    close_cutout_factor = max(0.5, min(2.0, close_cutout_factor))
    far_smoothing_factor = max(0.5, min(2.0, far_smoothing_factor))

    # Baseline working settings
    base_min_disp = 1
    base_num_disp = 16 * 8
    base_block_size = 9

    # Near-range preservation:
    # Larger factor -> larger num_disp
    num_disp_steps = round(8 * close_cutout_factor)   # around 4..16
    num_disp_steps = max(4, min(16, num_disp_steps))
    num_disp = 16 * num_disp_steps

    # Slightly bias min_disp upward when focusing on near field
    min_disp = max(0, int(round(base_min_disp * close_cutout_factor)))

    # Smoothing/detail tradeoff:
    # 0.5 -> block 5
    # 1.0 -> block 9
    # 2.0 -> block 15
    block_size = int(round(9 * far_smoothing_factor))
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(5, min(15, block_size))

    return min_disp, num_disp, block_size


def make_raw_disparity_view(disparity, min_disp, num_disp, valid_mask=None):
    disp_vis = (disparity - min_disp) / float(num_disp)
    disp_vis = np.clip(disp_vis, 0, 1)
    disp_vis = (disp_vis * 255).astype(np.uint8)
    color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    if valid_mask is not None:
        color[~valid_mask] = 0

    return color


def estimate_depth_cm_from_disparity(disparity_px, focal_px, baseline_m):
    if disparity_px <= 0:
        return None
    z_m = (focal_px * baseline_m) / disparity_px
    return z_m * 100.0


def make_depth_heatmap_view(disparity, focal_px, baseline_m, valid_mask, max_range_m):
    # =====================================================
    # Build a true depth heatmap from disparity.
    # Near = high intensity after inversion, far = low intensity.
    # Invalid pixels are shown as black.
    # =====================================================

    depth_m = np.zeros_like(disparity, dtype=np.float32)
    depth_m[valid_mask] = (focal_px * baseline_m) / disparity[valid_mask]

    viz = np.zeros_like(depth_m, dtype=np.float32)
    valid_depth = valid_mask & (depth_m > 0.0) & (depth_m <= max_range_m)

    if np.any(valid_depth):
        clipped = np.clip(depth_m[valid_depth], 0.0, max_range_m)
        inv = 1.0 - (clipped / max_range_m)
        viz[valid_depth] = inv

    viz_u8 = (viz * 255).astype(np.uint8)
    color = cv2.applyColorMap(viz_u8, cv2.COLORMAP_JET)
    color[~valid_depth] = 0
    return color


def overlay_cell_distances(
    img,
    disparity,
    valid_mask,
    focal_px,
    baseline_m,
    rows=10,
    cols=10,
    max_depth_cm=999,
):
    out = draw_overlay_grid(img, rows=rows, cols=cols, color=(255, 255, 255), thickness=1)
    h, w = disparity.shape[:2]

    for r in range(rows):
        y0 = int(r * h / rows)
        y1 = int((r + 1) * h / rows)

        for c in range(cols):
            x0 = int(c * w / cols)
            x1 = int((c + 1) * w / cols)

            cell = disparity[y0:y1, x0:x1]
            cell_valid = valid_mask[y0:y1, x0:x1]

            if not np.any(cell_valid):
                text = "--"
            else:
                closest_disp = float(np.percentile(cell[cell_valid], 95))
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
    # =====================================================
    # OpenCV stereo camera coordinates:
    #   x right, y down, z forward

    # ROS-like convention:
    #   x forward, y left, z up
    # =====================================================
    x_ros = z_cam
    y_ros = -x_cam
    z_ros = -y_cam
    return x_ros, y_ros, z_ros


def extract_sparse_points(
    disparity,
    points_3d,
    valid_mask,
    rows,
    cols,
    max_range_m,
    min_confidence,
):
    # =====================================================
    # One representative point per cell.

    # Strategy:
    # - valid disparity mask in cell
    # - pick 95th percentile disparity
    # - choose actual pixel nearest that target disparity
    # - emit XYZ + confidence + grid row/col
    # =====================================================
    
    h, w = disparity.shape[:2]
    points = []

    for r in range(rows):
        y0 = int(r * h / rows)
        y1 = int((r + 1) * h / rows)

        for c in range(cols):
            x0 = int(c * w / cols)
            x1 = int((c + 1) * w / cols)

            cell_disp = disparity[y0:y1, x0:x1]
            cell_valid = valid_mask[y0:y1, x0:x1]

            valid_fraction = float(np.count_nonzero(cell_valid)) / float(cell_disp.size)
            if valid_fraction < min_confidence:
                continue

            valid_values = cell_disp[cell_valid]
            target_disp = float(np.percentile(valid_values, 95))

            ys, xs = np.where(cell_valid)
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


def pack_packet(seq, rows, cols, points, stamp_ns):

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


# TCP/IP helpers:

def handle_tcp_client(conn, addr, frame_buffer):
    print(f"\nTCP image client connected: {addr}")

    try:
        while True:
            req = recv_message(conn)

            request_jpeg = bool(req.get("request_jpeg", True))
            max_width = int(req.get("max_width", 320))
            max_height = int(req.get("max_height", 180))
            jpeg_quality = int(req.get("jpeg_quality", 60))

            if not request_jpeg:
                send_json(conn, {
                    "ok": True,
                    "has_jpeg": False,
                    "message": "request_jpeg=false",
                })
                continue

            frame, seq, timestamp_ns = frame_buffer.get()
            if frame is None:
                send_json(conn, {
                    "ok": False,
                    "has_jpeg": False,
                    "error": "no_frame_available",
                })
                continue

            out = resize_to_fit(frame, max_width, max_height)
            jpg_bytes = encode_jpeg(out, jpeg_quality)
            h, w = out.shape[:2]

            send_json_with_jpeg(conn, {
                "ok": True,
                "has_jpeg": True,
                "seq": seq,
                "timestamp_ns": timestamp_ns,
                "width": w,
                "height": h,
                "jpeg_quality": jpeg_quality,
            }, jpg_bytes)

    except Exception as e:
        print(f"\nTCP image client disconnected {addr}: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

def tcp_image_server_loop(bind_host, bind_port, frame_buffer, stop_event):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((bind_host, bind_port))
    srv.listen(2)
    srv.settimeout(1.0)

    print(f"TCP image server listening on {bind_host}:{bind_port}")

    try:
        while not stop_event.is_set():
            try:
                conn, addr = srv.accept()
                conn.settimeout(5.0)
            except socket.timeout:
                continue

            t = threading.Thread(
                target=handle_tcp_client,
                args=(conn, addr, frame_buffer),
                daemon=True,
            )
            t.start()
    finally:
        srv.close()




def main():

    args = parse_args()

    min_confidence = args.min_confidence

    if not (0.0 <= min_confidence <= 1.0):
        raise ValueError("--min-confidence must be in [0.0, 1.0]")

    if args.grid_size <= 8:
        raise ValueError("--grid-size must be > 8")

    close_cutout_factor = args.close_cutout_factor
    far_smoothing_factor = args.far_smoothing_factor

    udp_ip = args.udp_ip
    udp_port = args.udp_port
    show_display = args.show_display
    show_preview = args.show_preview
    grid_rows = args.grid_size
    grid_cols = args.grid_size

    print(f"UDP target        : {udp_ip}:{udp_port}")
    print(f"Display enabled   : {show_display}")
    print(f"Preview enabled   : {show_preview}")
    print(f"PointCloud2 grid  : {grid_rows}x{grid_cols}")
    print(f"Min confidence    : {min_confidence:.3f}")

    tcp_image_port = args.tcp_image_port
    enable_tcp_image_server = args.enable_tcp_image_server

    print(f"TCP image server  : {enable_tcp_image_server}")
    if enable_tcp_image_server:
        print(f"TCP image port    : {tcp_image_port}")

    # =====================================================
    # min_confidence:
    #     0.05–0.1 → cleaner but may drop distant objects
    #     <0.01 → fills more grid cells, but noisy / unstable
    #     sweet spot ≈ 0.02–0.05
    # =====================================================

    try:
        calib = np.load(Calib.CALIBRATION_FILE)
    except FileNotFoundError:
        raise RuntimeError(f"Calibration file '{Calib.CALIBRATION_FILE}' not found")

    mapLx = calib["mapLx"]
    mapLy = calib["mapLy"]
    mapRx = calib["mapRx"]
    mapRy = calib["mapRy"]
    Q = calib["Q"]
    PL = calib["PL"]
    T = calib["T"]

    width = int(calib["image_width"])
    height = int(calib["image_height"])

    focal_px = float(PL[0, 0])
    baseline_m = float(np.linalg.norm(T))

    capL, capR = CameraDriver.open_stereo_cameras(width, height)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # min_disp = minimum disparity the matcher will search
    # the algorithm searches disparities in: [min_disp, min_disp + num_disp]
    #min_disp = 1    # min_disp = 0: full range, includes far; min_disp > 0: ignore far, focus near
    #block_size = 9  # matching window size (odd number); larger = smoother, less detail

    # smaller num_disp means the nearest measurable depth moves farther away
    # larger num_disp means the matcher can represent closer objects
    #num_disp = 16 * 6  # closest objects cutoff at 0.9 meters
    #num_disp = 16 * 8  # closest objects cutoff at 0.5 meters

    min_disp, num_disp, block_size = derive_sgbm_params(
        close_cutout_factor,
        far_smoothing_factor
    )

    print(f"min_disp          : {min_disp}")
    print(f"num_disp          : {num_disp}")
    print(f"block_size        : {block_size}")

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

    frame_buffer = LatestFrameBuffer()
    stop_event = threading.Event()
    tcp_thread = None

    if enable_tcp_image_server:
        tcp_thread = threading.Thread(
            target=tcp_image_server_loop,
            args=("0.0.0.0", tcp_image_port, frame_buffer, stop_event),
            daemon=True,
        )
        tcp_thread.start()

    seq = 0
    last_time = time.time()
    fps_filtered = 0.0
    show_heatmap = Streamer.START_IN_HEATMAP_MODE

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

            invalid_left_cols = num_disp
            invalid_right_cols = block_size // 2

            valid_mask = make_valid_disparity_mask(
                disparity,
                min_valid_disp=Stereo.MIN_VALID_DISP,
                invalid_left_cols=invalid_left_cols,
                invalid_right_cols=invalid_right_cols,
            )

            points_3d = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=False)

            points = extract_sparse_points(
                disparity,
                points_3d,
                valid_mask=valid_mask,
                rows=grid_rows,
                cols=grid_cols,
                max_range_m=Streamer.MAX_RANGE_M,
                min_confidence=min_confidence,
            )

            now = time.time()
            timestamp_ns = int(now * 1e9)

            packet = pack_packet(seq, grid_rows, grid_cols, points, timestamp_ns)
            sock.sendto(packet, (udp_ip, udp_port))

            frame_buffer.update(left_rect, seq, timestamp_ns)

            dt = now - last_time
            last_time = now
            fps_now = 1.0 / dt if dt > 0 else 0.0
            fps_filtered = 0.9 * fps_filtered + 0.1 * fps_now if fps_filtered > 0 else fps_now

            print(
                f"seq={seq} points={len(points)} udp_bytes={len(packet)} fps={fps_filtered:.2f}",
                end="\r",
                flush=True,
            )

            if show_preview:
                rect_preview = cv2.hconcat([
                    draw_preview_grid(left_rect, 40),
                    draw_preview_grid(right_rect, 40)
                ])
                cv2.imshow("Rectified Pair", rect_preview)
            else:
                try:
                    cv2.destroyWindow("Rectified Pair")
                except cv2.error:
                    pass

            if show_display:
                if show_heatmap:
                    disp_view = make_depth_heatmap_view(
                        disparity,
                        focal_px=focal_px,
                        baseline_m=baseline_m,
                        valid_mask=valid_mask,
                        max_range_m=Streamer.MAX_RANGE_M,
                    )
                    mode_text = "heatmap"
                else:
                    disp_view = make_raw_disparity_view(
                        disparity,
                        min_disp,
                        num_disp,
                        valid_mask=valid_mask,
                    )
                    mode_text = "disparity"

                disp_view = overlay_cell_distances(
                    disp_view,
                    disparity,
                    valid_mask=valid_mask,
                    focal_px=focal_px,
                    baseline_m=baseline_m,
                    rows=grid_rows,
                    cols=grid_cols,
                    max_depth_cm=int(Streamer.MAX_RANGE_M * 100.0),
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
                    f"mode={mode_text} | space=display on/off | any other key=toggle mode | q=quit",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
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
            elif key != 255:
                show_heatmap = not show_heatmap

            seq += 1

    finally:
        stop_event.set()
        if tcp_thread is not None:
            tcp_thread.join(timeout=2.0)
        capL.release()
        capR.release()
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
