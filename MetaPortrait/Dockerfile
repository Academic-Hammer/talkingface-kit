# 确保将Resources的7个文件复制到相应路径，具体操作参考配置文档
FROM nvidia/cuda:11.6.1-devel-ubuntu20.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH=/root/anaconda3/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    libgl1 \
    cmake \
    unzip \
    libx11-dev \
    libatlas-base-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /root/anaconda3 && \
    rm /miniconda.sh && \
    /root/anaconda3/bin/conda clean -a -y

# Initialize conda and add to PATH
RUN conda init bash && \
    echo "conda activate meta_portrait_base" >> ~/.bashrc

# Create working directory
WORKDIR /app

# 复制 environment.yml 并创建 Conda 环境
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate environment
SHELL ["conda", "run", "-n", "meta_portrait_base", "/bin/bash", "-c"]

# Install additional Python packages
RUN pip install  facenet-pytorch dlib

# Install PyTorch with CUDA 11.6 support
# RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install BasicSR and facexlib
WORKDIR /app/sr_model/Basicsr
COPY sr_model/Basicsr/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir facexlib

# Copy project files
WORKDIR /app
COPY . .

# Install Inpaint-Anything dependencies
WORKDIR /app/Inpaint-Anything
RUN python -m pip install -e segment_anything && \
    python -m pip install -r lama/requirements.txt

# Download and extract model files
WORKDIR /app/Inpaint-Anything/pretrained_models
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth && \
   unzip big-lama-20241220T131240Z-001.zip

# Return to app directory
WORKDIR /app

RUN pip install --upgrade opencv-python
RUN pip install --upgrade opencv-contrib-python

# Set up entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]