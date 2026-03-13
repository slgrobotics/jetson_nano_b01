#!/bin/bash
set -e

echo "Waiting for nvargus-daemon and /tmp/argus_socket..."

for i in {1..60}; do
    if systemctl is-active --quiet nvargus-daemon && [ -S /tmp/argus_socket ]; then
        echo "Argus is ready"
        exec /usr/bin/docker start -a duckpack
    fi
    sleep 2
done

echo "Argus did not become ready in time"
exit 1

