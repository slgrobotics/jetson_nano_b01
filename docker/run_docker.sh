#!/bin/bash

#
# see https://github.com/slgrobotics/jetson_nano_b01
#

#
# The container will have access to X11 screen on host (Jetson Nano).
# Note the -e DISPLAY=:0.0 line in "docker run" below.
# Open a terminal on Nano's Desktop and check the $DISPLAY environment variable:
#
#    echo $DISPLAY
#

docker run -it --rm \
  --net=host \
  --runtime nvidia \
  --privileged \
  --ipc=host \
  --cpuset-cpus="0-2" --cpus="2.7" \
  --memory="3200m" --memory-swap="3200m" \
  --name duckpack \
  -v /home/jetson/jetson_nano_b01:/code/src/dt-duckpack-yolo/shared \
  -v /tmp/argus_socket:/tmp/argus_socket \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=:0.0 \
  duckpack \
  bash -lc "source /opt/ros/noetic/setup.bash && exec bash"

