#!/bin/bash

#
# see https://github.com/slgrobotics/jetson_nano_b01
#

python3 yolo_runner.py --model /code/src/dt-duckpack-yolo/packages/yolo_node/best.engine  --sensor-id 0 --width 640 --height 480 --capture-fps 5 --max-yolo-hz 5 --imgsz 480  --warmup 3 --out-dir "." --save-every 10


