#!/bin/bash
set -e

echo "Waiting for nvargus-daemon and /tmp/argus_socket..."

# must start in '/home/jetson/jetson_nano_b01/src/stereo' directory

for i in {1..60}; do
    if systemctl is-active --quiet nvargus-daemon && [ -S /tmp/argus_socket ]; then
        echo "Argus is ready"
        # Edith the ROS2 node host name (or IP address) and other arguments below:
        exec python3 disparity_server.py --udp-ip <yourhost>.local --udp-port 5005 --no-display --grid-size 10
    fi
    sleep 2
done

echo "Argus did not become ready in time"
exit 1

