#!/usr/bin/env python3

"""
@brief
Interactive stereo dataset validator and cleanup tool.

This script loads stereo image pairs (left/right), detects calibration
checkerboard corners in each image, and provides visual feedback with
annotated previews.

Users can review each pair and:
- Keep valid pairs
- Manually delete bad pairs
- Automatically remove pairs where detection fails

The tool tracks detection statistics and summarizes dataset quality,
helping ensure only reliable image pairs are used for stereo calibration.

Key features:
- Side-by-side visualization with detection overlays
- Per-image pass/fail annotation
- Interactive and automatic dataset pruning
- Summary of detection success and failures

Intended use:
- Cleaning stereo datasets before calibration
- Removing frames with missed or poor detections
- Improving calibration accuracy and robustness
"""

import cv2
import glob
import os

from config import Stereo, Calib


def detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, Stereo.CHESSBOARD_SIZE, None)

    vis = img.copy()

    if found:
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        cv2.drawChessboardCorners(vis, Stereo.CHESSBOARD_SIZE, corners, found)

    return found, vis


def annotate(img, label, ok):
    out = img.copy()
    color = (0, 255, 0) if ok else (0, 0, 255)
    text = f"{label} | {'FOUND' if ok else 'NOT FOUND'}"

    cv2.putText(
        out,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    return out


def load_pairs():
    left_files = sorted(glob.glob(os.path.join(Calib.PAIR_DIR, "left", Calib.IMAGE_EXT)))
    right_files = sorted(glob.glob(os.path.join(Calib.PAIR_DIR, "right", Calib.IMAGE_EXT)))

    if len(left_files) != len(right_files):
        print("Warning: left/right image count mismatch")

    pair_count = min(len(left_files), len(right_files))
    return [(left_files[i], right_files[i]) for i in range(pair_count)]


def delete_pair(left_path, right_path):
    deleted_any = False

    for path in (left_path, right_path):
        if os.path.exists(path):
            os.remove(path)
            print(f"Deleted: {path}")
            deleted_any = True
        else:
            print(f"File already missing: {path}")

    return deleted_any


def main():
    pairs = load_pairs()

    if not pairs:
        print("No stereo pairs found.")
        return

    scanned = 0
    left_fail = 0
    right_fail = 0
    both_ok = 0
    deleted_count = 0

    print(f"Scanning {len(pairs)} stereo pairs")
    print(f"Pattern: {Stereo.CHESSBOARD_SIZE}")
    print("Controls:")
    print("  d = delete current pair")
    print("  q or ESC = quit")
    print("  any other key = keep and continue")
    print()

    i = 0
    while i < len(pairs):
        left_path, right_path = pairs[i]

        imgL = cv2.imread(left_path)
        imgR = cv2.imread(right_path)

        if imgL is None or imgR is None:
            print(f"Unreadable pair:")
            print(f"  {left_path}")
            print(f"  {right_path}")
            i += 1
            continue

        foundL, visL = detect(imgL)
        foundR, visR = detect(imgR)

        scanned += 1
        if not foundL:
            left_fail += 1
        if not foundR:
            right_fail += 1
        if foundL and foundR:
            both_ok += 1

        pair_ok = foundL and foundR

        visL = annotate(visL, os.path.basename(left_path), foundL)
        visR = annotate(visR, os.path.basename(right_path), foundR)

        preview = cv2.hconcat([visL, visR])

        status_text = (
            f"pair {i:03d}/{len(pairs)-1:03d} | "
            f"L={'OK' if foundL else 'FAIL'} "
            f"R={'OK' if foundR else 'FAIL'} | "
            f"d=delete q=quit"
        )

        cv2.putText(
            preview,
            status_text,
            (20, preview.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Stereo pair check", preview)

        print(
            f"pair {i:03d} | "
            f"L={'OK' if foundL else 'FAIL'} "
            f"R={'OK' if foundR else 'FAIL'}"
        )

        if Calib.DELETE_BAD_AUTOMATICALLY and not pair_ok:
            print("Auto-deleting bad pair")
            if delete_pair(left_path, right_path):
                deleted_count += 1
            i += 1
            continue

        key = cv2.waitKey(int(Calib.INTERVAL_SEC * 1000)) & 0xFF

        if key in (27, ord("q")):
            print("Stopped by user")
            break
        elif key == ord("d"):
            if delete_pair(left_path, right_path):
                deleted_count += 1
        else:
            pass

        i += 1

    cv2.destroyAllWindows()

    print("\n===== summary =====")
    print(f"pairs scanned : {scanned}")
    print(f"both OK       : {both_ok}")
    print(f"left failures : {left_fail}")
    print(f"right failures: {right_fail}")
    print(f"any failure   : {scanned - both_ok}")
    print(f"deleted pairs : {deleted_count}")


if __name__ == "__main__":
    main()
