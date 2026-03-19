# config.py

class Camera:    # parameters related to camera
    WIDTH = 640
    HEIGHT = 360
    FPS = 30
    LEFT = 0
    RIGHT = 1

class Stereo:    # parameters related to stereo algorithms
    CHESSBOARD_SIZE = (8, 6)  # inner corners (across, down)
    SQUARE_SIZE = 0.02821     # meters  =(28.2285714 + 28.1916666667) / 2 mm
    # Depth filtering:
    MIN_VALID_DISP = 1.0
    MAX_RANGE_M = 5.0
    MIN_CONFIDENCE = 0.02

class Calib:     # parameters used during calibration
    PAIR_DIR = f"stereo_pairs_{Camera.WIDTH}x{Camera.HEIGHT}"
    GRID_SIZE = 10
    NUM_PAIRS = 50
    INTERVAL_SEC = 2.0
    FLUSH_FRAMES = 4
    FLASH_SEC = 0.5
    IMAGE_EXT = "*.png"
    IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    DELETE_BAD_AUTOMATICALLY = True   # set True to auto-delete pairs where either side fails

 