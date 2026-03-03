# syntax=docker/dockerfile:1
FROM masterhapero/dt-duckpack-yolov11:ente-arm64v8

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN set -eux; \
    apt-key del F42ED6FBAB17C654 || true; \
    curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -; \
    rm -f /etc/apt/sources.list.d/cuda.list; \
    apt-get update && apt-get -y upgrade; \
    apt-get install -y --no-install-recommends \
        v4l-utils \
        libyaml-cpp-dev \
        libturbojpeg0-dev \
        libogg-dev \
        libtheora-dev \
        pkg-config \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libboost-all-dev; \
    python3 -m pip install --no-cache-dir --upgrade pip; \
    python3 -m pip install --no-cache-dir pycryptodomex;


#    rm -rf /var/lib/apt/lists/*

WORKDIR /code/src/dt-duckpack-yolo
CMD ["bash"]


