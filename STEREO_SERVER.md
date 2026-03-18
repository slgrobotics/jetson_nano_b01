Back to [Main Project Home](https://github.com/slgrobotics/articubot_one/wiki)

If you are looking for **ROS 2 compatible Inference Server** - read [this guide](https://github.com/slgrobotics/jetson_nano_b01/blob/main/README.md#inference-tcpip-server).

## ROS2 Stereo Camera "appliance" on Jetson Nano B01

The machine and OS (and other experiments) are described [here](https://github.com/slgrobotics/articubot_one/wiki/Ollama-on-Jetson-Nano).

**Note:**
- I use Waveshare Binocular Camera [Module](https://www.newegg.com/p/3C6-00U7-00PK8?item=9SIC4CTKR07923),
Dual IMX219 8 Megapixels. It supports stereo vision (depth vision).
- The software described here should work on RPi5 (once access to two cameras is provided) - I plan to test it some day.

### Purpose and Credits

The goal (if there was a goal) was to put the old Nano to work in some of my modern [projects](https://github.com/slgrobotics/articubot_one/wiki).

### Clone the repository

On the *host* (Jetson Nano) do:
```
cd
git clone https://github.com/slgrobotics/jetson_nano_b01.git
```

## Camera pipeline tests:

**test without X11 display, no output:**
```
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! fakesink -v
(should not crash)
```

**test brings left camera stream to X11 Display:**
```
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM),width=1640,height=1232,framerate=30/1' ! nvvidconv ! autovideosink
```

**test brings right camera stream to X11 Display:**
```
gst-launch-1.0 nvarguscamerasrc sensor-id=1 ! 'video/x-raw(memory:NVMM),width=1640,height=1232,framerate=30/1' ! nvvidconv ! autovideosink
```

**FPS benchmarks:**
```
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1' ! \
fpsdisplaysink video-sink=fakesink sync=false text-overlay=false -v

(~57–60 FPS average  0 drops   Stable pipeline   NVMM-only = GPU path)

gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1' ! \
nvvidconv ! 'video/x-raw,format=BGRx' ! videoconvert ! fpsdisplaysink video-sink=fakesink sync=false text-overlay=false -v

(720p → BGR conversion pipeline: Average FPS ≈ 56–57  Some jitter - 45–65 instantaneous, No drops)

gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1' ! \
nvvidconv ! 'video/x-raw(memory:NVMM),width=640,height=480' ! nvvidconv ! \
'video/x-raw,format=BGRx' ! videoconvert ! fpsdisplaysink video-sink=fakesink sync=false text-overlay=false -v

(1280×720 → resize in NVMM to 640x480 → BGR pipeline  ~58 FPS average  0 drops  Stable)
```

### Important: Stereo Camera calibration

You need to enable Desktop mode on your nano (`sudo init 5`)

There are multiple Python scripts in `~/jetson_nano_b01/src/stereo` directory:
```
jetson@jetson:~/jetson_nano_b01/src/stereo$ ls -l
Checkerboard-A4-30mm-8x6.pdf - printable board from https://markhedleyjones.com/projects/calibration-checkerboard-collection
check_stereo_pairs.py*
disparity_server.py*         - the streamer to send UDP packets to ROS2 node
find_chessboard_corners.py*
inspect_stereo_npz.py*
mono_undistortion_test.py*
stereo_calibrate.py*         - calibration processor, uses "stereo_pairs/" dataset
stereo_calibration.npz       - processed calibration file (calibration result)
stereo_capture_keybd.py*     - calibration dataset gatherer (one pair per "s" key press)
stereo_capture.py*           - calibration dataset gatherer (50 pairs, 2 sec interval)
stereo_disparity_numbers.py*
stereo_disparity.py*
stereo_pairs/                - calibration dataset is accumulated here
test_one_camera.py*
```

### Stereo Camera UDP Streamer

**Note:** the streamer described here is intended to work with [ROS2 Inference package](https://github.com/slgrobotics/ros2_jetson_nano_inference) - 
see [this launch file](https://github.com/slgrobotics/ros2_jetson_nano_inference/blob/main/launch/ros2_disparity_client.launch.py).

**Tip:** Setting up a lean headless appliance:
- use a 32 GB "high endurance" SD card and follow OS installation [process](https://github.com/slgrobotics/articubot_one/wiki/Ollama-on-Jetson-Nano)
- do not change swapping from default (except, maybe, *swappiness*)
- test camera pipelines
- make sure you calibrate cameras properly first
- after setting up and calibrting cameras switch to non-desktop operation, use SSH
- follow steps below

<img width="460" alt="jetson_cam" src="https://github.com/user-attachments/assets/d8231e9a-3182-45b0-bf4a-05a64b7aed67" />

**Note:**
- Jetson Nano takes 5.2V and consumes on average 1.0..1.2A when running camera server. It may very shortly peak to 8–10 W (4-5A).
Therefore a 15W/3A "*USB charger*" power supply is not a good option, use a good 5A-10A DC-DC [converter](https://www.amazon.com/dp/B08B4M1LXM) for a 12V power source.
- as an extra precaution, I use a 1000 uF and 0.1 uf capacitors and a ferrite [clip](https://www.amazon.com/dp/B07CWCSNW9) on the DC-DC converter output line.
- power input: 5.5mm x 2.1mm Power Adapter [Connectors](https://www.amazon.com/dp/B07C7VSRBG)
- fix max clocks for CPU: `sudo jetson_clocks`
- edit `/etc/sysctl.conf`  - add `vm.swappiness=10` to increase swapping threshold (Nano swaps to RAM by default, that's fine)

### Headless operation

When working with the ROS2 [disparity client](https://github.com/slgrobotics/ros2_jetson_nano_inference/blob/main/launch/ros2_disparity_client.launch.py),
Jetson Nano should behave as a maintenance-free intelligent camera rather than as a general-purpose computer.

Here is how to run the server in headless mode.

Disable the GUI to save RAM and speed up boot:
```
sudo systemctl set-default multi-user.target
```

Re-enable the GUI if needed:
```
sudo systemctl set-default graphical.target
```

To avoid crashes caused by the slow startup of *nvargus-daemon*, the streamr is started by *systemd* after a readiness check (alternatively, after a fixed delay).

The repository contains two required files. Make sure they are present:
- a startup script: `~/jetson_nano_b01/src/stereo/start-stereo.sh`
- a service file: `~/jetson_nano_b01/src/stereo/stereo.service`

You need to edit the startup script - change the ROS2 node host name (or IP address) and other arguments there.

Deploy the service file:
```
sudo cp ~/jetson_nano_b01/src/stereo/stereo.service /etc/systemd/system/.
```

Reload systemd and enable the service:
```
sudo systemctl daemon-reload
sudo systemctl enable stereo
sudo systemctl start stereo
```

Check the service status:
```
sudo systemctl status stereo
```

After reboot, the service should start the streamer automatically, and the ROS2 client should start receiving UDP packages within 2–3 minutes after power-up.

The Nano’s *Power* button initiates a graceful shutdown.

### Configuring WiFi

See [this section](https://github.com/slgrobotics/jetson_nano_b01?tab=readme-ov-file#configuring-wifi).

-------------------------

Back to [Main Project Home](https://github.com/slgrobotics/articubot_one/wiki)
