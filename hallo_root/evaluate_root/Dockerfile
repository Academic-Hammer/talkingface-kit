# 使用 NVIDIA 提供的 CUDA 镜像作为基础镜像
# FROM nvidia/cuda:12.7-base-ubuntu22.04
# FROM nvidia/cuda:12.7-devel-ubuntu22.04 
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# 设置环境变量，避免交互式安装
ENV DEBIAN_FRONTEND=noninteractive

# 安装 Python 环境和其他依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    build-essential \
    cmake \
    g++ \
    libopenblas-dev \
    libopencv-dev \
    liblapack-dev \
    libboost-all-dev \
    git \
    ffmpeg \
    && apt-get clean

# 设置工作目录
WORKDIR /app

# 复制项目文件到容器
COPY . /app/

# 安装 Python 包
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# 安装 InsightFace
RUN pip install insightface
