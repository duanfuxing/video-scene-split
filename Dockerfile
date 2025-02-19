FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    wget \
    cmake \
    git \
    vim \
    libtool \
    autoconf \
    automake \
    pkg-config \
    yasm \
    nasm \
    zlib1g-dev \
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libfdk-aac-dev \
    libsdl2-dev \
    libass-dev \
    libva-dev \
    libvdpau-dev \
    libxcb1-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    nvidia-driver-libs-headless \
    && rm -rf /var/lib/apt/lists/*

# 下载并安装nv-codec-headers 注意显卡驱动版本
RUN git clone https://github.com/FFmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && \
    git checkout n12.2.72.0 && \
    make install PREFIX=/usr/local

# 下载并编译支持NVIDIA硬件加速的FFmpeg
RUN git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg_source && \
    cd ffmpeg_source && \
    PKG_CONFIG_PATH=/usr/local/lib/pkgconfig ./configure \
    --enable-cuda \
    --enable-cuda-nvcc \
    --enable-cuvid \
    --enable-nvenc \
    --extra-cflags=-I/usr/local/cuda/include \
    --extra-ldflags=-L/usr/local/cuda/lib64 \
    --enable-libnpp \
    --enable-gpl \
    --enable-libx264 \
    --enable-libx265 \
    --enable-nonfree && \
    make -j$(nproc) && \
    make install

# 验证FFmpeg是否支持NVIDIA编码器
RUN ffmpeg -encoders | grep nvenc

# 设置工作目录
WORKDIR /app

# 安装Python依赖
RUN pip3 install --no-cache-dir ffmpeg-python \
    opencv-python \
    numpy \
    flask \
    gunicorn \
    tensorflow \
    pillow \
    tqdm \
    moviepy

# 设置环境变量
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility,graphics
ENV NVIDIA_REQUIRE_CUDA="cuda>=12.0"

# 暴露API端口
EXPOSE 9000

# 启动命令
CMD ["python3", "/app/server/api_server.py"]