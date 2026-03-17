#!/usr/bin/env python3
import cv2
import glob
import os
import time

# ====== easy-to-edit globals ======
CHESSBOARD_SIZE = (8, 6)   # inner corners: (across, down)
PAIR_DIR = "stereo_pairs"
PAUSE_SECONDS = 2.0
IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
# =================================


def find_images(folder):
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


def detect_and_draw(img, chessboard_size):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    vis = img.copy()
    if found:
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            ),
        )
        cv2.drawChessboardCorners(vis, chessboard_size, corners, found)

    return found, vis


def annotate(img, text, ok):
    out = img.copy()
    color = (0, 255, 0) if ok else (0, 0, 255)
    cv2.putText(
        out,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )
    return out


def process_folder(label, folder):
    paths = find_images(folder)
    if not paths:
        print(f"{label}: no images found in {folder}")
        return 0, 0

    misses = 0
    total = 0

    print(f"\n=== {label} ===")
    print(f"Scanning {len(paths)} images in: {folder}")

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Unreadable image: {path}")
            misses += 1
            total += 1
            continue

        found, vis = detect_and_draw(img, CHESSBOARD_SIZE)
        total += 1
        if not found:
            misses += 1

        basename = os.path.basename(path)
        status = "FOUND" if found else "NOT FOUND"
        print(f"{label}: {basename}: {status}")

        text = f"{label}: {basename} | {status} | pattern={CHESSBOARD_SIZE}"
        vis = annotate(vis, text, found)

        cv2.imshow(f"{label} chessboard check", vis)
        key = cv2.waitKey(int(PAUSE_SECONDS * 1000)) & 0xFF
        if key in (27, ord("q")):
            print("Stopped by user.")
            break

    cv2.destroyWindow(f"{label} chessboard check")
    return total, misses


def main():
    left_dir = os.path.join(PAIR_DIR, "left")
    right_dir = os.path.join(PAIR_DIR, "right")

    left_total, left_misses = process_folder("LEFT", left_dir)
    right_total, right_misses = process_folder("RIGHT", right_dir)

    print("\n=== summary ===")
    print(f"Pattern checked: {CHESSBOARD_SIZE}")
    print(f"LEFT : total={left_total}, misses={left_misses}, found={left_total - left_misses}")
    print(f"RIGHT: total={right_total}, misses={right_misses}, found={right_total - right_misses}")
    print(f"ALL  : total={left_total + right_total}, misses={left_misses + right_misses}, found={(left_total - left_misses) + (right_total - right_misses)}")
    print("\nPress any key in an OpenCV window or close windows to exit.")


if __name__ == "__main__":
    main()
