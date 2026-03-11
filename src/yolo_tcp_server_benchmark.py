#!/usr/bin/env python3

"""

This is just for validating yolo_tcp_server.py from:
 - another machine or 
 - from inside the Nano container.

Measures the round-trip latency including:
 - JPEG encoding (client)
 - network transfer
 - JPEG decoding (Nano)
 - YOLO inference
 - JSON serialization
 - network transfer back

So it approximates the real robot runtime latency, not just raw inference.

See https://chatgpt.com/s/t_69ac38f6cb3881919810a636f657f0e0

"""

import json
import socket
import struct
import time
import cv2

SERVER_HOST = "127.0.0.1"  # Use the actual IP address of the Nano (not container) if testing from another machine.
SERVER_PORT = 5001
REQUESTS = 20

# If True, the client sends an empty payload and the server captures from its camera.
# Otherwise, the client sends a JPEG frame (IMAGE_PATH).
# Make sure the server is running with the same setting for yolo_tcp_server_cam.py --use-server-cam true|false
USE_SERVER_CAM = False

IMAGE_PATH = "../media/duckies_2_480x480.jpg"


def recv_exact(sock, n):
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_request(sock, frame_id, jpg_bytes, use_server_cam=False):
    payload = b"" if use_server_cam else jpg_bytes

    header = {
        "frame_id": frame_id,
        "timestamp_ns": time.time_ns(),
        "encoding": "jpeg",
        "payload_size": len(payload),
    }

    hdr = json.dumps(header).encode("utf-8")

    sock.sendall(struct.pack(">I", len(hdr)))
    sock.sendall(hdr)
    if payload:
        sock.sendall(payload)


def recv_response(sock):
    hdr_len = struct.unpack(">I", recv_exact(sock, 4))[0]
    data = recv_exact(sock, hdr_len)
    response = json.loads(data.decode("utf-8"))

    jpeg_bytes = None
    if response.get("has_jpeg", False):
        jpeg_len = struct.unpack(">I", recv_exact(sock, 4))[0]
        jpeg_bytes = recv_exact(sock, jpeg_len)

    return response, jpeg_bytes


def main():
    print(f"Loading image from {IMAGE_PATH}...")

    img = cv2.imread(IMAGE_PATH)
    ok, enc = cv2.imencode(".jpg", img)
    jpg = enc.tobytes()

    print(f"Connecting to {SERVER_HOST}:{SERVER_PORT} and sending {REQUESTS} requests...")

    timings = []
    jpeg_sizes = []
    last_resp = None

    with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=10) as sock:
        sock.settimeout(10.0)

        for i in range(REQUESTS):
            t0 = time.time()

            send_request(sock, i, jpg, use_server_cam=USE_SERVER_CAM)
            resp, jpeg_bytes = recv_response(sock)

            t1 = time.time()

            timings.append((t1 - t0) * 1000.0)
            last_resp = resp

            if jpeg_bytes is not None:
                jpeg_sizes.append(len(jpeg_bytes))

    print("Last response:")
    print(json.dumps(last_resp, indent=2))

    if jpeg_sizes:
        avg_jpeg_kb = sum(jpeg_sizes) / len(jpeg_sizes) / 1024.0
        print(f"Returned JPEGs: {len(jpeg_sizes)}  average size: {avg_jpeg_kb:.1f} KB")

    for i, t in enumerate(timings):
        print(f"req {i+1:02d}: {t:6.1f} ms")

    avg_start = REQUESTS // 4
    avg_ms = sum(timings[avg_start:]) / len(timings[avg_start:])

    print(f"\nAverage latency over requests {avg_start+1}..{REQUESTS}: {avg_ms:.1f} ms\n")


if __name__ == "__main__":
    main()
