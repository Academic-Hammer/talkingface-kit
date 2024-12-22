# 使用官方 Anaconda 镜像作为基础镜像
FROM continuumio/miniconda3

LABEL authors="hjch1"

# 设置工作目录
WORKDIR /app

# 复制项目文件到工作目录
COPY . .

SHELL ["/bin/bash", "-c"]

ARG cache1="resnet18-5c106cde.pth"
ARG cache2="pipnet_resnet18_10x68x32x256_300w.pth"
ARG folder="/root/.cache/torch/hub/checkpoints"

RUN mkdir -p ${folder} && \
    mv ${cache1} ${folder} && \
    mv ${cache2} ${folder}

RUN echo "deb http://repo.huaweicloud.com/debian/ bookworm main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb-src http://repo.huaweicloud.com/debian/ bookworm main contrib non-free" >> /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get clean

ARG pip_mirror="https://repo.huaweicloud.com/repository/pypi/simple"
ARG wheel_path="torch-1.8.1+cu111-cp39-cp39-linux_x86_64.whl"

RUN conda init bash && \
    source ~/.bashrc && \
    conda create -n diffdub python==3.9.0 -y && \
    conda activate diffdub && \
    pip install ${wheel_path} && \
    rm -f ${wheel_path} && \
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html  && \
    pip install -r requirements.txt -i ${pip_mirror} && \
    pip install --no-deps torchlm==0.1.6.10 -i ${pip_mirror} && \
    pip cache purge && \
    conda clean -a -y

RUN chmod +x entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]




