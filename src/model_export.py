
#
# The following Python code will download model, invoke Ultralytics AutoUpdate and save the ".engine" file in the current directory.
# it takes several minutes to complete.
#
# See https://github.com/slgrobotics/jetson_nano_b01/blob/main/README.md#exporting-models-as-engine
#

from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.export(
    format="engine",
    imgsz=480,      # match what you already use
    device=0,       # GPU
    half=True       # FP16 for Nano
)

