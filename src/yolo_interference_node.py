#!/usr/bin/env python3
import subprocess
import threading
import queue
import time
import os
import signal

import numpy as np
import cv2

import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from sensor_msgs.msg import CompressedImage as ROSCompressedImage
from ultralytics import YOLO

from dt_robot_utils import get_robot_name

import shlex

# ------------------------------------------
#
# This is a ROS1 (bionic) test node inspired by:
#    https://github.com/masterhapero/dt-duckpack-yolov11/blob/ente-yolo/packages/yolo_node/src/yolo_interference_node.py
#
# For a modern ROS2 node see https://github.com/slgrobotics/ros2_jetson_nano_inference package
#
# ------------------------------------------

class ArgusStdoutGrabber:
    def __init__(self, sensor_id, width, height, fps):
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self.fps = fps
        self.bytes_per_frame = self.width * self.height * 3
        self.proc = None

    def _cmd(self):
        # Build argv explicitly; no shell.
        pipeline = (
            f"nvarguscamerasrc sensor-id={self.sensor_id} ! "
            f"video/x-raw(memory:NVMM),width={self.width},height={self.height},framerate={self.fps}/1,format=NV12 ! "
            f"nvvidconv ! video/x-raw,format=BGRx,width={self.width},height={self.height} ! "
            f"videoconvert ! video/x-raw,format=BGR,width={self.width},height={self.height} ! "
            f"fdsink fd=1 sync=false"
        )
        # gst-launch expects the pipeline as tokens; shlex.split is safer than .split()
        return ["gst-launch-1.0", "-q"] + shlex.split(pipeline)

    def start(self):
        self.proc = subprocess.Popen(
            self._cmd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,   # <-- keep stderr!
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

        # Process exited: dump a little stderr for diagnosis
        try:
            err = self.proc.stderr.read().decode("utf-8", errors="replace")
            tail = "\n".join(err.strip().splitlines()[-20:])
        except Exception:
            tail = "<could not read stderr>"

        print(f"[grabber] gst-launch exited rc={rc}\n[grabber] stderr tail:\n{tail}\n---")

        time.sleep(0.5)
        self.start()

    def read_frame(self):
        self.restart_if_needed()
        if not self.proc or not self.proc.stdout:
            return None

        data = self.proc.stdout.read(self.bytes_per_frame)
        if not data or len(data) < self.bytes_per_frame:
            return None

        import numpy as np
        frame = np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width, 3))
        return frame


class YoloNode(DTROS):
    # =====================================================
    # Grabs frames directly from CSI (Argus) and publishes YOLO-annotated JPEG frames.

    # Publisher:
    #     ~yolo/compressed (sensor_msgs/CompressedImage)
    # =====================================================
    def __init__(self):
        super(YoloNode, self).__init__(
            node_name="yolo",
            node_type=NodeType.DRIVER,
            help="Grabs CSI frames via nvarguscamerasrc and runs YOLO inference",
        )

        self._robot_name = get_robot_name()

        # Params
        self.sensor_id = rospy.get_param("~sensor_id", 0)
        self.width = rospy.get_param("~width", 640)
        self.height = rospy.get_param("~height", 480)
        self.capture_fps = rospy.get_param("~capture_fps", 3)
        self.max_yolo_hz = float(rospy.get_param("~max_yolo_hz", 5.0))

        # Throttle: maximum YOLO inferences per second (set 0 to disable)
        self.max_yolo_hz = float(rospy.get_param("~max_yolo_hz", 5.0))

        self.pub_img = rospy.Publisher(
            "~yolo/compressed",
            ROSCompressedImage,
            queue_size=1,
            dt_topic_type=TopicType.DRIVER,
            dt_help="JPEG compressed images with YOLO overlays",
        )

        model_path = rospy.get_param(
            "~model_path",
            "/code/src/dt-duckpack-yolo/packages/yolo_node/best.engine",
        )
        rospy.loginfo(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path, task="detect")

        # Latest-only queue (drops backlog)
        self.frame_queue = queue.Queue(maxsize=1)

        # Camera grabber
        self.grabber = ArgusStdoutGrabber(
            sensor_id=self.sensor_id,
            width=self.width,
            height=self.height,
            fps=self.capture_fps,
        )

        # Threads
        self.grab_thread = threading.Thread(target=self.grab_loop, daemon=True)
        self.proc_thread = threading.Thread(target=self.process_loop, daemon=True)

        self.grabber.start()
        self.grab_thread.start()
        self.proc_thread.start()

        self.loginfo(
            f"Initialized. CSI sensor_id={self.sensor_id} {self.width}x{self.height}@{self.capture_fps} "
            f"max_yolo_hz={self.max_yolo_hz}"
        )

    def shutdown(self):
        try:
            self.grabber.stop()
        except Exception:
            pass

    def _queue_latest(self, frame, stamp):
        # keep latest only
        try:
            while not self.frame_queue.empty():
                _ = self.frame_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self.frame_queue.put_nowait({"frame": frame, "timestamp": stamp})
        except queue.Full:
            pass

    def grab_loop(self):
        # Grab frames continuously, overwrite latest
        while not rospy.is_shutdown():
            frame = self.grabber.read_frame()
            if frame is None:
                rospy.logerr("Camera stream ended (gst-launch exited?)")
                time.sleep(0.5)
                continue

            self._queue_latest(frame, rospy.Time.now())

    def process_loop(self):
        last_infer_t = 0.0
        min_dt = (1.0 / self.max_yolo_hz) if self.max_yolo_hz and self.max_yolo_hz > 0 else 0.0

        while not rospy.is_shutdown():
            try:
                item = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Throttle inference rate (skip frames if YOLO is too slow / you want lower rate)
            now = time.time()
            if min_dt > 0 and (now - last_infer_t) < min_dt:
                # Too soon: just drop this frame and keep going
                continue
            last_infer_t = now

            frame = item["frame"]
            stamp = item["timestamp"]

            try:
                processed = frame.copy()

                results = self.model.predict(frame, imgsz=480, verbose=False)

                # Draw detections
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf)
                        cls = int(box.cls)

                        cv2.rectangle(processed, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            processed,
                            f"{cls}:{conf:.2f}",
                            (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                # Publish compressed image
                msg = ROSCompressedImage()
                msg.header.stamp = stamp
                msg.header.frame_id = "camera"
                msg.format = "jpeg"
                ok, enc = cv2.imencode(".jpg", processed)
                if ok:
                    msg.data = enc.tobytes()
                    self.pub_img.publish(msg)

            except Exception as e:
                rospy.logerr(f"YOLO processing error: {e}")

    def spin(self):
        try:
            rospy.spin()
        finally:
            self.shutdown()


if __name__ == "__main__":
    node = YoloNode()
    node.spin()

