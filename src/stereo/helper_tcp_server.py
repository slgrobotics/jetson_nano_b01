#!/usr/bin/env python3

# =====================================================
# Utility helpers for thread-safe frame sharing and JPEG encoding.

# This module provides:
# - A synchronized buffer for sharing the latest camera frame between threads
# - Image resizing with aspect-ratio preservation
# - JPEG encoding with configurable quality

# It is designed to support real-time pipelines where a producer thread
# (e.g., stereo processing loop) updates frames, and consumer threads
# (e.g., TCP servers) retrieve and serve them without blocking.

# Intended use:
# - Sharing latest camera frames between processing and networking threads
# - Preparing resized images for network transmission
# - Encoding frames into JPEG for efficient streaming
# =====================================================

import cv2

class LatestFrameBuffer:
    # =====================================================
    # Thread-safe container for sharing the latest image frame.
    #
    # This class stores the most recent frame along with its sequence number
    # and timestamp. It uses a mutex to ensure safe concurrent access between
    # producer and consumer threads.
    #
    # Key features:
    # - Atomic update of frame + metadata
    # - Safe retrieval without race conditions
    # - Defensive copying to avoid data corruption
    #
    # Intended use:
    # - Passing frames from capture/processing loops to networking threads
    # - Serving the latest available image to clients on demand
    # =====================================================

    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.seq = -1
        self.timestamp_ns = 0

    def update(self, frame, seq, timestamp_ns):
        with self.lock:
            self.frame = frame.copy()
            self.seq = seq
            self.timestamp_ns = timestamp_ns

    def get(self):
        with self.lock:
            if self.frame is None:
                return None, -1, 0
            return self.frame.copy(), self.seq, self.timestamp_ns
        

def resize_to_fit(img, max_width, max_height):
    # =====================================================
    # Resize an image to fit within given dimensions while preserving aspect ratio.
    #
    # The image is scaled down (never up) so that both width and height are
    # within the specified limits. If the image already fits, it is returned unchanged.
    #
    # Intended use:
    # - Preparing images for bandwidth-limited transmission
    # - Generating preview frames for remote visualization
    # =====================================================

    h, w = img.shape[:2]

    if max_width <= 0 or max_height <= 0:
        return img

    scale = min(float(max_width) / float(w), float(max_height) / float(h), 1.0)
    if scale >= 1.0:
        return img

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def encode_jpeg(img, quality=60):
    # =====================================================
    # Encode an image into JPEG format with configurable quality.
    #
    # The image is compressed using OpenCV's JPEG encoder. The quality parameter
    # controls compression level and output size.
    #
    # Raises an exception if encoding fails.
    #
    # Intended use:
    # - Preparing images for network transmission (e.g., TCP streaming)
    # - Reducing bandwidth usage for remote visualization
    # =====================================================

    quality = int(max(1, min(100, quality)))
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return enc.tobytes()