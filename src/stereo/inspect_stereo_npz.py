#!/usr/bin/env python3
import numpy as np
import cv2
import glob
import os

NPZ = "stereo_calibration.npz"
PAIR_DIR = "stereo_pairs"


def stats(name, a):
    a = np.asarray(a)
    finite = np.isfinite(a)
    print(f"{name}: shape={a.shape} dtype={a.dtype}")
    print(f"  finite: {np.count_nonzero(finite)}/{a.size}")
    if np.any(finite):
        print(f"  min={np.nanmin(a):.6f} max={np.nanmax(a):.6f}")
    else:
        print("  no finite values")


def main():
    data = np.load(NPZ)

    print("=== keys ===")
    print(sorted(data.files))
    print()

    K1 = data["K1"]
    D1 = data["D1"]
    K2 = data["K2"]
    D2 = data["D2"]
    R = data["R"]
    T = data["T"]
    RL = data["RL"]
    RR = data["RR"]
    PL = data["PL"]
    PR = data["PR"]
    mapLx = data["mapLx"]
    mapLy = data["mapLy"]
    mapRx = data["mapRx"]
    mapRy = data["mapRy"]
    w = int(data["image_width"])
    h = int(data["image_height"])

    print(f"image size from npz: width={w} height={h}\n")

    for name, arr in [
        ("K1", K1), ("D1", D1),
        ("K2", K2), ("D2", D2),
        ("R", R), ("T", T),
        ("RL", RL), ("RR", RR),
        ("PL", PL), ("PR", PR),
        ("mapLx", mapLx), ("mapLy", mapLy),
        ("mapRx", mapRx), ("mapRy", mapRy),
    ]:
        stats(name, arr)
        print()

    print("=== map sanity ===")
    for name, m, lo, hi in [
        ("mapLx", mapLx, 0, w - 1),
        ("mapLy", mapLy, 0, h - 1),
        ("mapRx", mapRx, 0, w - 1),
        ("mapRy", mapRy, 0, h - 1),
    ]:
        inside = (m >= lo) & (m <= hi) & np.isfinite(m)
        frac = np.count_nonzero(inside) / m.size
        print(f"{name}: inside-image fraction = {frac:.3f}")

    print()
    print("=== distortion sanity ===")
    print("D1 =", D1.ravel())
    print("D2 =", D2.ravel())
    print()
    print("=== translation / baseline ===")
    print("T =", T.ravel())
    print("baseline magnitude =", float(np.linalg.norm(T)))

    left_images = sorted(glob.glob(os.path.join(PAIR_DIR, "left", "*.png")))
    right_images = sorted(glob.glob(os.path.join(PAIR_DIR, "right", "*.png")))

    if not left_images or not right_images:
        print("\nNo saved image pairs found for offline remap test.")
        return

    imgL = cv2.imread(left_images[0])
    imgR = cv2.imread(right_images[0])

    if imgL is None or imgR is None:
        print("\nCould not read first stereo pair.")
        return

    print("\n=== offline remap test on first saved pair ===")
    print("left image shape:", imgL.shape)
    print("right image shape:", imgR.shape)

    rectL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)

    # Add guide lines
    for y in range(0, h, 40):
        cv2.line(rectL, (0, y), (w - 1, y), (0, 255, 0), 1)
        cv2.line(rectR, (0, y), (w - 1, y), (0, 255, 0), 1)

    both = cv2.hconcat([rectL, rectR])
    cv2.imshow("Offline rectified first pair", both)
    print("Press any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
