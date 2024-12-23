# 使用官方的 Python 镜像作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 复制当前目录下的所有文件到容器中的工作目录
COPY . /app/

# 安装必需的编译工具和依赖
RUN apt-get update && apt-get install -y --fix-missing \
    build-essential \
    cmake \
    g++ \
    libopenblas-dev \
    libopencv-dev \
    liblapack-dev \
    libboost-all-dev \
    git \
    python3-dev \
    python3-pip \
    && apt-get clean


# 手动安装insightface
RUN pip install insightface

# 安装项目的依赖
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
