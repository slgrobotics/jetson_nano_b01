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
import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO


def now_ns() -> int:
    return time.time_ns()


def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> Dict[str, Any]:
    hdr_len_bytes = recv_exact(sock, 4)
    hdr_len = struct.unpack(">I", hdr_len_bytes)[0]
    hdr_bytes = recv_exact(sock, hdr_len)
    header = json.loads(hdr_bytes.decode("utf-8"))

    payload_size = int(header.get("payload_size", 0))
    payload = recv_exact(sock, payload_size) if payload_size > 0 else b""
    header["_payload"] = payload
    return header


def send_json(sock: socket.socket, obj: Dict[str, Any]) -> None:
    data = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    sock.sendall(struct.pack(">I", len(data)))
    sock.sendall(data)


def decode_image(header: Dict[str, Any]) -> np.ndarray:
    encoding = header.get("encoding", "jpeg").lower()
    payload = header["_payload"]

    if encoding in ("jpeg", "jpg"):
        arr = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode JPEG payload")
        return frame

    raise ValueError(f"Unsupported encoding: {encoding}")


@dataclass
class InferenceJob:
    frame_id: Any
    timestamp_ns: int
    received_ns: int
    frame: np.ndarray
    done_evt: threading.Event = field(default_factory=threading.Event)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class InferenceWorker:
    def __init__(self, model_path: str, imgsz: int, warmup: int, quiet: bool = False):
        self.model_path = model_path
        self.imgsz = imgsz
        self.warmup = warmup
        self.quiet = quiet
        self.job_q: "queue.Queue[InferenceJob]" = queue.Queue()
        self.stop_evt = threading.Event()

        print(f"Loading YOLO model: {self.model_path}", flush=True)
        self.model = YOLO(self.model_path, task="detect")
        self.model_name = self._derive_model_name()
        self.ultralytics_version = ultralytics.__version__

        print(f"...warming model with {self.warmup} blank images...", flush=True)
        self._warm_model()

        print("Model classes:", flush=True)
        for i, name in self.model.names.items():
            print(f"{i}: {name}", flush=True)

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _derive_model_name(self) -> str:
        # Best-effort human-readable model name
        try:
            return getattr(self.model, "ckpt_path", None) or self.model_path
        except Exception:
            return self.model_path

    def _warm_model(self) -> None:
        if self.warmup <= 0:
            return
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        t0 = time.time()
        for _ in range(self.warmup):
            _ = self.model.predict(dummy, imgsz=self.imgsz, verbose=False)
        dt = time.time() - t0
        print(f"Warmed model with {self.warmup} dummy inference(s) in {dt:.2f}s", flush=True)

    def submit(self, job: InferenceJob) -> None:
        self.job_q.put(job)

    def shutdown(self) -> None:
        self.stop_evt.set()
        try:
            self.job_q.put_nowait(
                InferenceJob(frame_id=-1, timestamp_ns=0, received_ns=0, frame=np.zeros((1, 1, 3), dtype=np.uint8))
            )
        except Exception:
            pass
        self.thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self.stop_evt.is_set():
            try:
                job = self.job_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if self.stop_evt.is_set():
                break

            infer_start_ns = now_ns()
            queue_delay_ms = (infer_start_ns - job.received_ns) / 1e6

            try:
                ti0 = time.time()
                results = self.model.predict(job.frame, imgsz=self.imgsz, verbose=False)
                ti1 = time.time()
                infer_ms = (ti1 - ti0) * 1000.0
                infer_end_ns = now_ns()

                detections: List[Dict[str, Any]] = []

                if len(results):
                    r = results[0]
                    for box in r.boxes:
                        trk_id = None
                        if getattr(box, "id", None) is not None:
                            try:
                                trk_id = int(box.id[0])
                            except Exception:
                                trk_id = None

                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = self.model.names[cls]

                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx, cy, w, h = box.xywh[0].tolist()

                        detections.append(
                            {
                                "class_id": cls,
                                "label": label,
                                "confidence": conf,
                                "track_id": trk_id,
                                "bbox_xyxy": [x1, y1, x2, y2],
                                "bbox_xywh": [cx, cy, w, h],
                            }
                        )

                job.result = {
                    "ok": True,
                    "frame_id": job.frame_id,
                    "timestamp_ns": job.timestamp_ns,
                    "server_received_ns": job.received_ns,
                    "server_infer_start_ns": infer_start_ns,
                    "server_infer_end_ns": infer_end_ns,
                    "queue_delay_ms": queue_delay_ms,
                    "infer_ms": infer_ms,
                    "model_name": self.model_name,
                    "model_path": self.model_path,
                    "ultralytics_version": self.ultralytics_version,
                    "imgsz": self.imgsz,
                    "detections": detections,
                }

            except Exception as e:
                job.error = str(e)
                job.result = {
                    "ok": False,
                    "frame_id": job.frame_id,
                    "timestamp_ns": job.timestamp_ns,
                    "server_received_ns": job.received_ns,
                    "server_infer_start_ns": infer_start_ns,
                    "queue_delay_ms": queue_delay_ms,
                    "error": str(e),
                    "model_name": self.model_name,
                    "model_path": self.model_path,
                    "ultralytics_version": self.ultralytics_version,
                    "imgsz": self.imgsz,
                }
            finally:
                job.done_evt.set()


class TCPInferenceServer:
    def __init__(self, host: str, port: int, worker: InferenceWorker, request_timeout_s: float = 30.0, quiet: bool = False):
        self.host = host
        self.port = port
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
                    frame = decode_image(header)
                except Exception as e:
                    send_json(sock, {"ok": False, "error": f"decode_error: {e}"})
                    continue

                frame_id = header.get("frame_id")
                timestamp_ns = int(header.get("timestamp_ns", 0))
                received_ns = now_ns()

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to YOLO .engine/.pt/.onnx")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--request-timeout", type=float, default=30.0)
    ap.add_argument("--quiet", action="store_true", help="Suppress non-error logs")
    args = ap.parse_args()

    worker = InferenceWorker(
        model_path=args.model,
        imgsz=args.imgsz,
        warmup=max(0, args.warmup),
        quiet=args.quiet,
    )

    server = TCPInferenceServer(
        host=args.host,
        port=args.port,
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

