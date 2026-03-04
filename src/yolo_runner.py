#!/usr/bin/env python3
import argparse
import os
import queue
import shlex
import signal
import subprocess
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


class ArgusStdoutGrabber:
    def __init__(self, sensor_id: int, width: int, height: int, fps: int):
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self.fps = fps

        # NOTE: BGR bytes via videoconvert (works but can be CPU heavy).
        self.bytes_per_frame = self.width * self.height * 3
        self.proc: Optional[subprocess.Popen] = None

    def _cmd(self):
        pipeline = (
            f"nvarguscamerasrc sensor-id={self.sensor_id} ! "
            f"video/x-raw(memory:NVMM),width={self.width},height={self.height},framerate={self.fps}/1,format=NV12 ! "
            f"nvvidconv ! video/x-raw,format=BGRx,width={self.width},height={self.height} ! "
            f"videoconvert ! video/x-raw,format=BGR,width={self.width},height={self.height} ! "
            f"fdsink fd=1 sync=false"
        )
        return ["gst-launch-1.0", "-q"] + shlex.split(pipeline)

    def start(self):
        if self.proc and self.proc.poll() is None:
            return
        self.proc = subprocess.Popen(
            self._cmd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.SIGINT)
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()

    def restart_if_needed(self):
        if self.proc is None:
            self.start()
            return

        rc = self.proc.poll()
        if rc is None:
            return

        try:
            err = self.proc.stderr.read().decode("utf-8", errors="replace")
            tail = "\n".join(err.strip().splitlines()[-30:])
        except Exception:
            tail = "<could not read stderr>"

        print(f"[grabber] gst-launch exited rc={rc}\n[grabber] stderr tail:\n{tail}\n---", flush=True)

        if "CameraProvider" in tail or "Cannot create camera provider" in tail:
            time.sleep(5.0)
        else:
            time.sleep(0.5)

        self.start()

    def read_frame_blocking(self) -> Optional[np.ndarray]:
        self.restart_if_needed()
        if not self.proc or not self.proc.stdout:
            return None

        data = self.proc.stdout.read(self.bytes_per_frame)
        if not data or len(data) < self.bytes_per_frame:
            return None

        return np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width, 3))


def warm_model(model: YOLO, imgsz: int, n: int = 1) -> None:
    """
    Warm up TensorRT / CUDA by running a few dummy inferences.
    Your engine is fixed at 480, so imgsz must match that.
    """
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    t0 = time.time()
    for i in range(n):
        _ = model.predict(dummy, imgsz=imgsz, verbose=False)
    dt = time.time() - t0
    print(f"Warmed model with {n} dummy inference(s) in {dt:.2f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sensor-id", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--capture-fps", type=int, default=5)
    ap.add_argument("--max-yolo-hz", type=float, default=1.0)
    ap.add_argument("--imgsz", type=int, default=480)  # fixed by your .engine
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--save-every", type=int, default=10)
    args = ap.parse_args()

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading YOLO model: {args.model}", flush=True)
    model = YOLO(args.model, task="detect")

    # Warm first (before starting Argus)
    warm_model(model, imgsz=args.imgsz, n=max(0, args.warmup))

    # Now start camera pipeline
    grabber = ArgusStdoutGrabber(args.sensor_id, args.width, args.height, args.capture_fps)
    grabber.start()

    latest_q: "queue.Queue[Tuple[np.ndarray, float]]" = queue.Queue(maxsize=1)
    stop_evt = threading.Event()

    def capture_loop():
        while not stop_evt.is_set():
            frame = grabber.read_frame_blocking()
            if frame is None:
                time.sleep(0.05)
                continue

            # keep latest only
            try:
                while True:
                    latest_q.get_nowait()
            except queue.Empty:
                time.sleep(0.05)
                pass
            try:
                latest_q.put_nowait((frame, time.time()))
            except queue.Full:
                time.sleep(0.05)
                pass

    threading.Thread(target=capture_loop, daemon=True).start()

    min_dt = (1.0 / args.max_yolo_hz) if args.max_yolo_hz and args.max_yolo_hz > 0 else 0.0
    last_t = 0.0

    n_inf = 0
    t0 = time.time()
    idx = 0

    print(
        f"Capturing {args.width}x{args.height}@{args.capture_fps} | "
        f"YOLO max_hz={args.max_yolo_hz} imgsz={args.imgsz}",
        flush=True
    )

    try:
        while True:
            try:
                frame, _ts = latest_q.get(timeout=1.0)
            except queue.Empty:
                time.sleep(0.05)
                continue

            now = time.time()
            if min_dt > 0 and (now - last_t) < min_dt:
                time.sleep(0.05)
                continue
            last_t = now

            results = model.predict(frame, imgsz=args.imgsz, verbose=False)

            if args.out_dir:
                annotated = results[0].plot() if len(results) else frame
                idx += 1
                if idx % max(1, args.save_every) == 0:
                    frame_name = "frame_annotated.jpg"
                    #frame_name = f"frame_{idx:06d}.jpg"
                    print(f"saving frame, idx={idx}  name={frame_name}", flush=True)
                    cv2.imwrite(os.path.join(args.out_dir, frame_name), annotated)

            n_inf += 1
            dt = now - t0
            if dt >= 1.0:
                print(f"infer_fps={n_inf/dt:5.2f}  len(results)={len(results)}", flush=True)
                n_inf = 0
                t0 = now

    except KeyboardInterrupt:
        pass
    finally:
        print("...stopping capure loop...", flush=True)
        stop_evt.set()
        print("...stopping grabber...", flush=True)
        grabber.stop()
        print("Done.", flush=True)


if __name__ == "__main__":
    main()


