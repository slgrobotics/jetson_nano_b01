#!/usr/bin/env python3

#
# https://chatgpt.com/s/t_69a5c15b08748191adc9f6de66a13fa3
#

import re
import subprocess
import sys
import signal

PAT = re.compile(
    r"rendered:\s*(\d+),\s*dropped:\s*(\d+),\s*current:\s*([0-9.]+),\s*average:\s*([0-9.]+)"
)

def main():
    sensor_id = 0
    w, h = 1280, 720
    fps = 60

    cmd = [
        "gst-launch-1.0",
        "nvarguscamerasrc", f"sensor-id={sensor_id}",
        "!",
        f"video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1",
        "!",
        "fpsdisplaysink",
        "video-sink=fakesink",
        "sync=false",
        "text-overlay=false",
        "-v",
    ]

    print("Running:", " ".join(cmd))
    print("Press Ctrl+C to stop.\n")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        for line in proc.stdout:
            m = PAT.search(line)
            if m:
                rendered, dropped, current, avg = m.groups()
                print(f"rendered={rendered} dropped={dropped} fps_current={current} fps_avg={avg}")
    except KeyboardInterrupt:
        # Forward Ctrl+C to gst-launch so it prints stats and exits cleanly
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass
    finally:
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

    return proc.returncode or 0

if __name__ == "__main__":
    sys.exit(main())


