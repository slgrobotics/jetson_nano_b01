import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

import ultralytics
from ultralytics import YOLO
from .argus_stdout_grabber import ArgusStdoutGrabber

##
# InferenceWorker: A background worker thread that runs YOLO inference on submitted frames.
#
# It loads the YOLO model, warms it up with dummy inferences, and then waits for jobs to be submitted via a thread-safe queue.
# Each job contains a frame and metadata. The worker runs inference on the frame, captures timing information, and stores the results back in the job object before signaling that it's done.
# This allows the main server thread to submit frames for inference and wait for the results without blocking the server's ability to handle incoming requests.
#

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

            infer_start_ns = time.time_ns()
            queue_delay_ms = (infer_start_ns - job.received_ns) / 1e6

            try:
                ti0 = time.time()
                results = self.model.predict(job.frame, imgsz=self.imgsz, verbose=False)
                ti1 = time.time()
                infer_ms = (ti1 - ti0) * 1000.0
                infer_end_ns = time.time_ns()

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
