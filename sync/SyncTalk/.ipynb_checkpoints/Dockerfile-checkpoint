# 使用 Ubuntu 18.04 作为基础镜像
FROM ubuntu:18.04

# 设置环境变量，防止交互式安装时出现提示
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:$PATH"
ENV CUDA_HOME="/usr/local/cuda-11.3"
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH"

# 更新系统并安装必要工具
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    software-properties-common \
    portaudio19-dev \
    && apt-get clean

# 安装 CUDA 11.3 和 cuDNN 8.2
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && apt-get install -y \
    cuda-11-3 \
    libcudnn8=8.2.1.*-1+cuda11.3 \
    libcudnn8-dev=8.2.1.*-1+cuda11.3 \
    && apt-get clean

# 安装 Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy

# 创建 Conda 环境并安装 Python 3.8.8
RUN conda create -n synctalk python=3.8.8 -y && \
    conda clean -a -y

# 激活环境并安装 PyTorch 和依赖
RUN /bin/bash -c "source activate synctalk && \
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 && \
    pip install -r /app/requirements.txt && \
    pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1121/download.html && \
    pip install tensorflow-gpu==2.8.1"

# 复制项目文件到容器
WORKDIR /app
COPY . /app

# 激活环境并安装本地包
RUN /bin/bash -c "source activate synctalk && \
    pip install ./freqencoder && \
    pip install ./shencoder && \
    pip install ./gridencoder && \
    pip install ./raymarching"

# 确保脚本可执行
RUN chmod +x install.sh inference.sh main.sh

# 默认执行 main.sh
CMD ["/bin/bash"]
