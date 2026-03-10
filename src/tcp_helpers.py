import json
import socket
import struct
from typing import Any, Dict, Optional

import cv2
import numpy as np

##
# Helper functions for TCP communication and image decoding.
#
# These are used by the server code to send/receive messages and decode images.
#

def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> Dict[str, Any]:
    hdr_len_bytes = recv_exact(sock, 4)
    hdr_len = struct.unpack(">I", hdr_len_bytes)[0]
    hdr_bytes = recv_exact(sock, hdr_len)
    header = json.loads(hdr_bytes.decode("utf-8"))

    payload_size = int(header.get("payload_size", 0))
    payload = recv_exact(sock, payload_size) if payload_size > 0 else b""
    header["_payload"] = payload
    return header


def send_json(sock: socket.socket, obj: Dict[str, Any]) -> None:
    data = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    sock.sendall(struct.pack(">I", len(data)))
    sock.sendall(data)


def decode_image(header: Dict[str, Any]) -> np.ndarray:
    encoding = header.get("encoding", "jpeg").lower()
    payload = header["_payload"]

    if encoding in ("jpeg", "jpg"):
        arr = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode JPEG payload")
        return frame

    raise ValueError(f"Unsupported encoding: {encoding}")


