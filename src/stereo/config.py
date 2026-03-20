# config.py

class Camera:    # parameters related to camera
    """
    Camera configuration parameters.

    Defines resolution, frame rate, and device identifiers for left and right
    CSI cameras. These settings are used across capture, calibration, and
    streaming components to ensure consistent image dimensions and timing.
    """
    WIDTH = 640
    HEIGHT = 360
    FPS = 30
    LEFT = 0
    RIGHT = 1
    CAMERA_STEREO_BASE = 0.060  # meters, Waveshare stereo camera for Jetson Nano

class Stereo:    # parameters related to stereo algorithms
    """
    Stereo vision and depth estimation parameters.

    Contains checkerboard geometry used for calibration as well as filtering
    thresholds for disparity and 3D point extraction. These values directly
    affect calibration accuracy and depth computation quality.

    Note:
    CHESSBOARD_SIZE and SQUARE_SIZE must match your physical calibration board.
    """
    CHESSBOARD_SIZE = (8, 6)   # inner corners (across, down)
    #SQUARE_SIZE = 0.02821     # meters, small board
    SQUARE_SIZE = 0.06290      # meters, large board
    # Depth filtering (reasonable defaults):
    MIN_VALID_DISP = 1.0
    MAX_RANGE_M = 5.0
    MIN_CONFIDENCE = 0.02

class Calib:     # parameters used during calibration
    """
    Dataset capture and calibration workflow parameters.

    Defines storage locations, capture timing, and dataset management options
    used during stereo image acquisition and calibration. Includes settings for
    automatic cleanup of invalid pairs and output file naming.

    Ensures consistent dataset structure and reproducible calibration runs.
    """
    PAIR_DIR = f"stereo_pairs_{Camera.WIDTH}x{Camera.HEIGHT}"
    GRID_SIZE = 10
    NUM_PAIRS = 50
    INTERVAL_SEC = 2.0
    FLUSH_FRAMES = 4
    FLASH_SEC = 0.5
    IMAGE_EXT = "*.png"
    IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    DELETE_BAD_AUTOMATICALLY = True   # set True to auto-delete pairs where either side fails
    CALIBRATION_FILE = "stereo_calibration.npz"  # produced during calibration

class Streamer:
    """
    Runtime streaming and visualization parameters.

    Controls behavior of the stereo depth streaming pipeline, including maximum
    depth range and initial display mode (raw disparity or depth heatmap).

    These settings influence visualization and downstream perception output.
    """
    MAX_RANGE_M = 5.0
    START_IN_HEATMAP_MODE = False  # False = disparity, True = depth heatmap

