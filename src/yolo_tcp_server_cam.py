#!/usr/bin/env python3

"""
#
# Run it in the Nano container.
# If your container is on host networking, the RPi5/ROS2 can connect to the Nano IP on port 5001.
# See https://chatgpt.com/s/t_69ab72e92950819191c249c64f5adc5b
#
python3 yolo_tcp_server.py \
  --model /code/src/dt-duckpack-yolo/packages/yolo_node/best.engine \
  --imgsz 480 \
  --warmup 3 \
  --host 0.0.0.0 \
  --port 5001 \
  --request-timeout 30 \
  --quiet
"""

import argparse
import json
import socket
import struct
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

from argus_stdout_grabber import ArgusStdoutGrabber
from inference_worker import InferenceWorker, InferenceJob
from tcp_helpers import decode_image, recv_message, send_json


class TCPInferenceServer:
    def __init__(self, host: str, port: int, grabber: ArgusStdoutGrabber, worker: InferenceWorker, request_timeout_s: float = 30.0, quiet: bool = False):
        self.host = host
        self.port = port
        self.grabber = grabber
        self.worker = worker
        self.request_timeout_s = request_timeout_s
        self.stop_evt = threading.Event()
        self.quiet = quiet
        self.server_sock: Optional[socket.socket] = None
    

    def serve_forever(self) -> None:
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(8)
        print(f"Listening on {self.host}:{self.port}  quiet: {self.quiet}", flush=True)

        try:
            while not self.stop_evt.is_set():
                client_sock, addr = self.server_sock.accept()
                if not self.quiet:
                    print(f"Client connected: {addr}", flush=True)
                t = threading.Thread(target=self.handle_client, args=(client_sock, addr), daemon=True)
                t.start()
        finally:
            if self.server_sock:
                self.server_sock.close()

    def shutdown(self) -> None:
        self.stop_evt.set()
        if self.server_sock:
            try:
                self.server_sock.close()
            except Exception:
                pass

    def handle_client(self, sock: socket.socket, addr) -> None:
        sock.settimeout(self.request_timeout_s)
        try:
            while not self.stop_evt.is_set():
                try:
                    header = recv_message(sock)
                except socket.timeout:
                    continue
                except ConnectionError:
                    break
                except Exception as e:
                    send_json(sock, {"ok": False, "error": f"bad_request: {e}"})
                    break

                try:
                    frame_from_client = decode_image(header)
                    frame_from_camera = self.grabber.read_frame()
                    frame = frame_from_camera # frame_from_client if frame_from_client is not None else frame_from_camera
                except Exception as e:
                    send_json(sock, {"ok": False, "error": f"decode_error: {e}"})
                    continue

                frame_id = header.get("frame_id")
                timestamp_ns = int(header.get("timestamp_ns", 0))
                received_ns = time.time_ns()

                job = InferenceJob(
                    frame_id=frame_id,
                    timestamp_ns=timestamp_ns,
                    received_ns=received_ns,
                    frame=frame,
                )

                self.worker.submit(job)

                if not job.done_evt.wait(timeout=self.request_timeout_s):
                    send_json(
                        sock,
                        {
                            "ok": False,
                            "frame_id": frame_id,
                            "timestamp_ns": timestamp_ns,
                            "server_received_ns": received_ns,
                            "error": "inference_timeout",
                        },
                    )
                    continue

                send_json(sock, job.result or {"ok": False, "error": "unknown_error"})

        except Exception as e:
            print(f"Client handler error {addr}: {e}", flush=True)
        finally:
            try:
                sock.close()
            except Exception:
                pass
            if not self.quiet:
                print(f"Client disconnected: {addr}", flush=True)


def main():
    """
    # quick pipeline test:
    grabber = ArgusStdoutGrabber(0, 640, 480, 5)
    grabber.start()

    while True:
        frame = grabber.read_frame()
        if frame is not None:
            print(frame.shape)
        else:
            print("got None frame")
    """

    ap = argparse.ArgumentParser()
    # Camera related args:
    ap.add_argument("--sensor-id", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--capture-fps", type=int, default=5)
    # Server related args:
    ap.add_argument("--model", required=True, help="Path to YOLO .engine/.pt/.onnx")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--request-timeout", type=float, default=30.0)
    ap.add_argument("--quiet", action="store_true", help="Suppress non-error logs")  # When the flag appears → True, otherwise False
    args = ap.parse_args()

    # Now start camera pipeline
    grabber = ArgusStdoutGrabber(args.sensor_id, args.width, args.height, args.capture_fps)
    grabber.start()

    # make sure the camera is working before loading the model:
    while True:
        frame = grabber.read_frame()
        if frame is not None:
            print(frame.shape)
            break
        else:
            print("got None frame")

    worker = InferenceWorker(
        model_path=args.model,
        imgsz=args.imgsz,
        warmup=max(0, args.warmup),
        quiet=args.quiet,
    )

    server = TCPInferenceServer(
        host=args.host,
        port=args.port,
        grabber=grabber,
        worker=worker,
        request_timeout_s=args.request_timeout,
        quiet=args.quiet,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down...", flush=True)
        server.shutdown()
        worker.shutdown()
        print("Done.", flush=True)


if __name__ == "__main__":
    main()

