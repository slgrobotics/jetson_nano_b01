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
import socket
import threading
import time
from typing import Optional

import cv2

from argus_stdout_grabber import ArgusStdoutGrabber, LatestFrame
from inference_worker import InferenceWorker, InferenceJob
from tcp_helpers import decode_image, recv_message, send_json, send_json_with_jpeg


class TCPInferenceServer:
    def __init__(self, host: str, port: int, grabber: Optional[ArgusStdoutGrabber], worker: InferenceWorker, request_timeout_s: float = 30.0, quiet: bool = False):
        self.host = host
        self.port = port
        self.grabber = grabber
        self.worker = worker
        self.request_timeout_s = request_timeout_s
        self.stop_evt = threading.Event()
        self.quiet = quiet
        self.server_sock: Optional[socket.socket] = None

        # These are used when server-side camera capture is enabled:
        self.latest_frame = LatestFrame()  # Shared latest-frame buffer
        self.camera_thread: Optional[threading.Thread] = None
    
    def camera_loop(self) -> None:
        if self.grabber is None:
            return
        
        while not self.stop_evt.is_set():
            try:
                frame = self.grabber.read_frame()  # called in camera_thread
                if frame is not None:
                    self.latest_frame.set(frame)
            except Exception as e:
                if not self.quiet:
                    print(f"Camera loop error: {e}", flush=True)
                time.sleep(0.05)

    def serve_forever(self) -> None:
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(8)
        print(f"Listening on {self.host}:{self.port}  quiet: {self.quiet}", flush=True)

        if self.grabber is not None:
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()

        try:
            while not self.stop_evt.is_set():
                try:
                    client_sock, addr = self.server_sock.accept()
                except OSError:
                    if self.stop_evt.is_set():
                        break
                    raise

                if not self.quiet:
                    print(f"Client connected: {addr}", flush=True)
                t = threading.Thread(target=self.handle_client, args=(client_sock, addr), daemon=True)
                t.start()
        finally:
            if self.server_sock is not None:
                self.server_sock.close()

    def shutdown(self) -> None:
        self.stop_evt.set()
        if self.server_sock is not None:
            try:
                self.server_sock.close()
            except Exception:
                pass

    def handle_client(self, sock: socket.socket, addr) -> None:
        """
        Handles a single client connection, processing incoming frames and sending back inference results.
        This method runs in a separate thread for each client.
        It reads frames either from the local camera (if grabber is set) or from the client socket,
          submits them to the inference worker, and sends back the results.
        It also handles timeouts and errors gracefully.

        Responses:
         - ok=True,  has_jpeg=True  -> server-camera success, JSON followed by JPEG image for control and overlays
         - ok=True,  has_jpeg=False -> client-image success, JSON only (client already has the image for overlays)
         - ok=False, has_jpeg=False -> any error, JSON only
        """
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
                    send_json(sock, {"ok": False, "has_jpeg": False, "error": f"bad_request: {e}"})
                    break

                try:
                    if self.grabber is not None:
                        frame = self.latest_frame.get()  # from local camera
                        if frame is None:
                            send_json(sock, {"ok": False, "has_jpeg": False, "error": "no_camera_frame"})
                            continue
                    else:
                        frame = decode_image(header)  # from TCP/IP client
                        if frame is None:
                            send_json(sock, {"ok": False, "has_jpeg": False, "error": "no_client_frame"})
                            continue

                except Exception as e:
                    send_json(sock, {"ok": False, "has_jpeg": False, "error": f"decode_error: {e}"})
                    continue

                frame_id = header.get("frame_id")
                timestamp_ns = int(header.get("timestamp_ns", 0))
                received_ns = time.time_ns()
                # Client can request a JPEG for visualization, but does it only if using server camera feed
                raw_request_jpeg = header.get("request_jpeg", False)
                request_jpeg = raw_request_jpeg is True

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
                            "has_jpeg": False,
                            "frame_id": frame_id,
                            "timestamp_ns": timestamp_ns,
                            "server_received_ns": received_ns,
                            "error": "inference_timeout",
                        },
                    )
                    continue

                response = job.result
                if response is None:
                    send_json(sock, {"ok": False, "has_jpeg": False, "error": "missing_result"})
                    continue

                response = dict(response)  # defensive copy to modify

                """
                if not self.quiet:
                    print(
                        f"frame_id={frame_id} request_jpeg={request_jpeg} "
                        f"ok={response.get('ok', False)} use_server_cam={self.grabber is not None}",
                        flush=True,
                    )
                """

                if self.grabber is not None and response.get("ok", False) and request_jpeg:   # Return dict["ok"] if the "ok" key exists (hopefully True), otherwise return the default "False"
                    ok, encoded = cv2.imencode(
                        ".jpg",
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 80],
                    )
                    if not ok:
                        send_json(sock, {"ok": False, "has_jpeg": False, "error": "jpeg_encode_failed"})
                        continue

                    response["has_jpeg"] = True
                    send_json_with_jpeg(sock, response, encoded.tobytes())
                else:
                    response["has_jpeg"] = False
                    send_json(sock, response)

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

    ap = argparse.ArgumentParser()
    # Camera related args:
    ap.add_argument("--use_server_cam",  action="store_true", help="Use the server camera feed instead of client frames")  # When the flag appears → True, otherwise False
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

    use_server_cam = args.use_server_cam

    if use_server_cam:
        print("Using server camera feed instead of client frames")
        # Now start camera pipeline
        grabber = ArgusStdoutGrabber(args.sensor_id, args.width, args.height, args.capture_fps)
        grabber.start()

        # make sure the camera is working before loading the model:
        for _ in range(10):
            frame = grabber.read_frame()  # called in main thread
            if frame is not None:
                print(f"Local camera works, frame shape: {frame.shape}")
                break
            time.sleep(0.2)
        else:
            raise RuntimeError("Camera failed to produce frames")
    else:
        print("Not using server camera feed, will rely on client frames")
        grabber = None

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
        if grabber is not None:
            grabber.stop()
        server.shutdown()
        worker.shutdown()
        print("Done.", flush=True)


if __name__ == "__main__":
    main()

