# 使用 NVIDIA CUDA 基础镜像（包含 Ubuntu 22.04 和 CUDA 12.1）
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04 

# 设置环境变量，避免交互式安装
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-distutils \
    python3-pip \
    ffmpeg \
    git \
    build-essential \
    curl \
    wget \
    && apt-get clean

# 安装 Miniconda
RUN curl -o miniconda.sh -L "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda init bash

# 将 Conda 添加到环境变量
ENV PATH=/opt/conda/bin:$PATH

# 复制本地项目文件到容器
COPY . /app

# 设置工作目录
WORKDIR /app

# 创建 Conda 环境并安装依赖
RUN /bin/bash -c "conda create -n hallo python=3.10 && \
    source activate hallo && \
    pip install -r requirements.txt && \
    pip install ."

# 安装 huggingface_hub 0.25.2
RUN /opt/conda/bin/conda run -n hallo pip install huggingface_hub==0.25.2

# 克隆 Hugging Face 模型
RUN git clone https://huggingface.co/fudan-generative-ai/hallo pretrained_models

# 暴露端口（根据需要开放端口）
EXPOSE 8000

# 启动 bash 终端
CMD ["/bin/bash"]