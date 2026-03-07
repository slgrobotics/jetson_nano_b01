#!/usr/bin/env python3

"""

This is just for validating yolo_tcp_server.py from:
 - another machine or 
 - from inside the Nano container.

See https://chatgpt.com/s/t_69ab72e92950819191c249c64f5adc5b

"""

import json
import socket
import struct
import time
import cv2

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

def send_request(sock, frame_id, jpg_bytes):
    header = {
        "frame_id": frame_id,
        "timestamp_ns": time.time_ns(),
        "encoding": "jpeg",
        "payload_size": len(jpg_bytes),
    }
    hdr = json.dumps(header).encode("utf-8")
    sock.sendall(struct.pack(">I", len(hdr)))
    sock.sendall(hdr)
    sock.sendall(jpg_bytes)

def recv_response(sock):
    n = struct.unpack(">I", recv_exact(sock, 4))[0]
    data = recv_exact(sock, n)
    return json.loads(data.decode("utf-8"))

img = cv2.imread("../media/duckies_1_480x480.jpg")
ok, enc = cv2.imencode(".jpg", img)
jpg = enc.tobytes()

with socket.create_connection(("127.0.0.1", 5001), timeout=10) as sock:
    send_request(sock, 1, jpg)
    resp = recv_response(sock)
    print(json.dumps(resp, indent=2))

