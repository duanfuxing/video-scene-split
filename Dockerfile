FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# 设置代理
ENV http_proxy=http://172.19.26.199:7890 \
    https_proxy=http://172.19.26.199:7890 \
    HTTP_PROXY=http://172.19.26.199:7890 \
    HTTPS_PROXY=http://172.19.26.199:7890

# 修改镜像源
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list

# 安装系统依赖
ENV DEBIAN_FRONTEND=noninteractive
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
    nvidia-driver-525-server \
    libreadline-dev \
    libbz2-dev \
    libsm6 \
    libxrender1 \
    libxext-dev \
    libgomp1 \
    liblzma-dev \
    libgl1-mesa-glx \
    libprotobuf-dev \
    protobuf-compiler \
    libglib2.0-0 \
    mpich \
    openmpi-bin \
    libopenmpi-dev \
    gcc \
    g++ \
    make \
    zlib1g \
    openssl \
    libsqlite3-dev \
    libssl-dev \
    libffi-dev \
    unzip \
    pciutils \
    net-tools \
    libblas-dev \
    gfortran \
    libblas3 \
    libopenblas-dev \
    git-lfs \
    libswresample-dev \
    libfreetype6-dev \
    libtheora-dev \
    libvorbis-dev \
    texinfo \
    libmp3lame-dev \
    libopus-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 下载并安装nv-codec-headers 注意显卡驱动版本
RUN git clone https://github.com/FFmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && \
    git checkout n12.2.72.0 && \
    make install PREFIX=/usr/local

# 设置CUDA环境变量
ENV PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# 下载并编译支持NVIDIA硬件加速的FFmpeg
RUN cd /tmp && \
    wget https://ffmpeg.org/releases/ffmpeg-5.1.tar.gz && \
    tar xvf ffmpeg-5.1.tar.gz && \
    cd ffmpeg-5.1 && \
    PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/local/cuda/lib64/pkgconfig" \
    ./configure \
    --prefix=/usr/local \
    --pkg-config-flags="--static" \
    --disable-debug \
    --disable-doc \
    --disable-ffplay \
    --enable-shared \
    --enable-gpl \
    --enable-nonfree \
    --enable-libfdk-aac \
    --enable-libx264 \
    --enable-libx265 \
    --enable-cuda-nvcc \
    --enable-cuda \
    --enable-cuvid \
    --enable-nvenc \
    --enable-libnpp \
    --enable-pthreads \
    --extra-cflags="-I/usr/local/cuda/include -I/usr/local/include" \
    --extra-ldflags="-L/usr/local/cuda/lib64 -L/usr/local/lib" \
    --extra-libs="-lpthread -lm" && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && \
    rm -rf ffmpeg* && \
    ldconfig

# 验证FFmpeg是否支持NVIDIA编码器
RUN ffmpeg -encoders | grep nvenc

# 设置工作目录
WORKDIR /app

# 安装Python依赖，并使用阿里云PyPI镜像加速
RUN pip3 install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple ffmpeg-python \
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