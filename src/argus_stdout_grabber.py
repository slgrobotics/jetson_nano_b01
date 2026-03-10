import shlex
import signal
import subprocess
import select
import time
import numpy as np

##
# ArgusStdoutGrabber: Captures camera frames by running a gst-launch pipeline that outputs raw BGRx frames to stdout.
#
# This is a workaround to get frames from the camera without using OpenCV's VideoCapture, which can be unstable on Jetson Nano.
# It runs gst-launch-1.0 with a pipeline that captures from the camera, converts to BGRx format, and writes to stdout.
# The grabber reads from the subprocess's stdout, decodes the raw bytes into frames, and provides a method to read frames with a timeout.
# It also monitors the subprocess and restarts it if it crashes or stalls.
#

class ArgusStdoutGrabber:
    def __init__(self, sensor_id: int, width: int, height: int, fps: int):
        self.sensor_id = sensor_id
        self.w = width
        self.h = height
        self.fps = fps
        self.bytes_per_frame = self.w * self.h * 4  # BGRx
        self.proc = None

    def _cmd(self):
        pipeline = (
            f"nvarguscamerasrc sensor-id={self.sensor_id} ! "
            f"video/x-raw(memory:NVMM),width={self.w},height={self.h},framerate={self.fps}/1,format=NV12 ! "
            f"nvvidconv ! video/x-raw,format=BGRx,width={self.w},height={self.h} ! "
            f"fdsink fd=1 sync=false"
        )
        return ["gst-launch-1.0", "-q"] + shlex.split(pipeline)

    def start(self):
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

    def _stderr_tail(self, n=30):
        if not self.proc or not self.proc.stderr:
            time.sleep(0.05)
            return ""
        try:
            data = self.proc.stderr.read().decode("utf-8", errors="replace")
            return "\n".join(data.strip().splitlines()[-n:])
        except Exception:
            time.sleep(0.05)
            return "<could not read stderr>"

    def restart_if_needed(self):
        if self.proc is None:
            self.start()
            return
        rc = self.proc.poll()
        if rc is None:
            return
        tail = self._stderr_tail()
        print(f"[grabber] gst-launch exited rc={rc}\n[grabber] stderr tail:\n{tail}\n---", flush=True)
        time.sleep(1.0)
        self.start()

    def read_frame(self, timeout_s=2.0):
        """
        Returns BGR frame (h, w, 3) or None on timeout / stream trouble.
        Never blocks forever.
        """
        self.restart_if_needed()
        if not self.proc or not self.proc.stdout:
            time.sleep(0.05)
            return None

        # Wait until some data is available to read (avoid hanging forever)
        r, _, _ = select.select([self.proc.stdout], [], [], timeout_s)
        if not r:
            # No data; likely stalled. Check if process died.
            if self.proc.poll() is not None:
                return None
            return None

        data = self.proc.stdout.read(self.bytes_per_frame)
        if not data or len(data) < self.bytes_per_frame:
            return None

        arr = np.frombuffer(data, dtype=np.uint8).reshape((self.h, self.w, 4))
        bgr = arr[:, :, :3]
        # bgr = arr[:, :, :3].copy()  # copy so buffer isn't tied to pipe bytes - if observing instability, try this to decouple from pipe buffer
        return bgr


