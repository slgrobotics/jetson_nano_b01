## Experiments with Jetson Nano B01

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
docker image ls

REPOSITORY    TAG      IMAGE ID       CREATED    SIZE
duckpack      latest   54c022f33208   ...        6.09GB
```

We will use a shared directory to persist files between container runs:
```
(host) /home/jetson/jetson_nano_b01  ==> /code/src/dt-duckpack-yolo/shared (container)
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
  -v /home/jetson/jetson_nano_b01:/code/src/dt-duckpack-yolo/shared \
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

Make sure the "*shared*" volume is visible from the container shell:
```
root@jetson:/code/src/dt-duckpack-yolo# cd shared
root@jetson:/code/src/dt-duckpack-yolo/shared# ll
total 56
drwxrwxr-x 8 1000 1000  4096 Mar  4 15:59 .git/
-rw-rw-r-- 1 1000 1000  4688 Mar  4 15:59 .gitignore
-rw-rw-r-- 1 1000 1000 16726 Mar  4 15:59 LICENSE
-rw-rw-r-- 1 1000 1000  4366 Mar  4 15:59 README.md
drwxrwxr-x 2 1000 1000  4096 Mar  4 15:59 docker/
drwxrwxr-x 2 1000 1000  4096 Mar  4 15:59 src/
```

## Tests (inside the container):

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

## Experiments

To explore Nano's GPU we use files in the "shared" folder, cloned from the repository.

**Warning:** Review the "[Core Technical Barriers](https://github.com/slgrobotics/articubot_one/wiki/Ollama-on-Jetson-Nano#core-technical-barriers)" guide. Results may vary. Sanity not included. ;-)

Switch to the "*shared*" directory:
```
root@jetson:/code/src/dt-duckpack-yolo# cd shared/src
root@jetson:/code/src/dt-duckpack-yolo/shared/src# ll
total 28
drwxrwxr-x 2 1000 1000 4096 Mar  4 15:59 ./
drwxrwxr-x 5 1000 1000 4096 Mar  4 15:59 ../
-rw-rw-r-- 1 1000 1000 1542 Mar  4 15:59 test.py
-rw-rw-r-- 1 1000 1000 8050 Mar  4 15:59 yolo_interference_node.py
-rw-rw-r-- 1 1000 1000 6399 Mar  4 15:59 yolo_runner.py
```

Make sure the "frame grabber" test works:
```
root@jetson:/code/src/dt-duckpack-yolo/shared/src# python3 test.py 
Running: gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1 ! fpsdisplaysink video-sink=fakesink sync=false text-overlay=false -v
Press Ctrl+C to stop.

rendered=32 dropped=0 fps_current=62.51 fps_avg=62.51
rendered=62 dropped=0 fps_current=59.95 fps_avg=61.25
rendered=93 dropped=0 fps_current=60.10 fps_avg=60.86
```

Try YOLO runner ("*--model*" is required, other arguments shown may differ from defaults, see source):
```
python3 yolo_runner.py --model /code/src/dt-duckpack-yolo/packages/yolo_node/best.engine \
 --sensor-id 0 --width 640 --height 480 --capture-fps 5 --max-yolo-hz 5 --imgsz 480 \
 --warmup 3 --out-dir "." --save-every 10

Loading YOLO model: /code/src/dt-duckpack-yolo/packages/yolo_node/best.engine
Loading /code/src/dt-duckpack-yolo/packages/yolo_node/best.engine for TensorRT inference...
[TRT] [I] [MemUsageChange] Init CUDA: CPU +229, GPU +0, now: CPU 301, GPU 2518 (MiB)
[TRT] [I] Loaded engine size: 10 MiB
[TRT] [W] Using an engine plan file across different models of devices is not recommended
          and is likely to affect performance or even cause errors.
[TRT] [I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +158, GPU +244, now: CPU 478, GPU 2791 (MiB)
[TRT] [I] [MemUsageChange] Init cuDNN: CPU +241, GPU +349, now: CPU 719, GPU 3140 (MiB)
[TRT] [I] [MemUsageChange] TensorRT-managed allocation in engine deserialization:
                            CPU +0, GPU +8, now: CPU 0, GPU 8 (MiB)
[TRT] [I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +1, GPU +0, now: CPU 709, GPU 3130 (MiB)
[TRT] [I] [MemUsageChange] Init cuDNN: CPU +0, GPU +0, now: CPU 709, GPU 3130 (MiB)
[TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation:
                            CPU +0, GPU +19, now: CPU 0, GPU 27 (MiB)
Warmed model with 3 dummy inference(s) in 34.44s
Capturing 640x480@5 | YOLO max_hz=5.0 imgsz=480
infer_fps= 0.35
infer_fps= 0.37
...
infer_fps= 3.00  len(results)=1
saving frame, idx=210  name=frame_annotated.jpg
infer_fps= 2.50  len(results)=1
```

Loading and "*warming up*" the model takes less than two minutes. The video pipeline building is postponed till the model is fully operational, otherwise the pipeline will crash.

You can open File manager on Jetson's Desktop, switch to `/home/jetson/jetson_nano_b01/src` directory and double-click on the "*frame_annotated.img*" to see it updated in real time. The viewer automatically refreshes image when it is updated.

Watch GPU usage on host using `jtop` (it wasn't high in my case).

**Tip:** if the image grabber pipeline is stuck:
- run on host `sudo systemctl restart nvargus-daemon`
- restart the container


-------------------------

Back to [Main Project Home](https://github.com/slgrobotics/articubot_one/wiki)
