## experiments with Jetson Nano B01

This is a collection of files and docs related to my experiments with NVIDIA Jetson Nano Bo1 (2020 DevKit with two camera connectors).

The machine and OS (and other experiments) are described [here](https://github.com/slgrobotics/articubot_one/wiki/Ollama-on-Jetson-Nano).

I used a Docker file, code and info from the following work of Sampsa Ranta:
- https://github.com/masterhapero/dt-duckpack-yolov11
- https://hub.docker.com/r/masterhapero/dt-duckpack-yolov11/tags

The goal (if there was a goal) was to put the old Nano to work in some of my modern [projects](https://github.com/slgrobotics/articubot_one/wiki).

An alternative approach is described [here](https://github.com/slgrobotics/articubot_one/wiki/Jetson-Nano%3A-%22jetson%E2%80%90inference%22-container).

**Note:** I use Waveshare Binocular Camera [Module](https://www.newegg.com/p/3C6-00U7-00PK8?item=9SIC4CTKR07923),
Dual IMX219 8 Megapixels. It supports stereo vision (depth vision). Other CSI-connected cameras compatible with Nano B01 might work.

### Clone the repository

On the *host* (Jetson Nano) do:
```
cd
git clone https://github.com/slgrobotics/jetson_nano_b01.git
```

### Create Docker image

```
cd ~/jetson_nano_b01/docker
docker build -t duckpack .
```
It will take a while, the base image will be pulled and the container OS will be upgraded with the latest Ubuntu 18.04 packages.

Confirm:
```
jetson@jetson:~/masterhapero$ docker image ls
REPOSITORY    TAG      IMAGE ID       CREATED    SIZE
duckpack      latest   54c022f33208   ...        6.09GB
```

Prepare a shared directory to keep files between container runs:
```
mkdir -p /home/jetson/masterhapero/catkin_ws
```

The container will have access to X11 screen on host (Jetson Nano). Note the `-e DISPLAY=:0.0` line in "docker run" below. Open a terminal on Nano's Desktop and check the $DISPLAY environment variable:
```
echo $DISPLAY
```

Run it: allow 3 of 4 CPUs, limit RAM use to 3.2GB (of 4GB), no swap allowed - container will quit if RAM is full
```
docker run -it --rm \
  --net=host \
  --runtime nvidia \
  --privileged \
  --ipc=host \
  --cpuset-cpus="0-2" --cpus="2.7" \
  --memory="3200m" --memory-swap="3200m" \
  --name duckpack \
  -v /home/jetson/masterhapero/catkin_ws:/code/src/dt-duckpack-yolo/catkin_ws \
  -v /tmp/argus_socket:/tmp/argus_socket \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=:0.0 \
  duckpack \
  bash -lc "source /opt/ros/noetic/setup.bash && exec bash"
```

You can run extra shells in host's terminals:
```
docker exec -it duckpack bash -lc "source /opt/ros/noetic/setup.bash && exec bash"
```

**Tip:** Stuck pipeline - run on host:  `sudo systemctl restart nvargus-daemon`

## Tests (inside the container):

**test without X11 display:**
```
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! fakesink -v
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
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1' ! fpsdisplaysink video-sink=fakesink sync=false text-overlay=false -v
(~57–60 FPS average  0 drops   Stable pipeline   NVMM-only = GPU path)

gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1' ! nvvidconv ! 'video/x-raw,format=BGRx' ! videoconvert ! fpsdisplaysink video-sink=fakesink sync=false text-overlay=false -v
(720p → BGR conversion pipeline: Average FPS ≈ 56–57  Some jitter - 45–65 instantaneous, No drops)

gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1' ! nvvidconv ! 'video/x-raw(memory:NVMM),width=640,height=480' ! nvvidconv ! \
'video/x-raw,format=BGRx' ! videoconvert ! fpsdisplaysink video-sink=fakesink sync=false text-overlay=false -v
(1280×720 → resize in NVMM to 640x480 → BGR pipeline  ~58 FPS average  0 drops  Stable)
```




-------------------------

Back to [Main Project Home](https://github.com/slgrobotics/articubot_one/wiki)
