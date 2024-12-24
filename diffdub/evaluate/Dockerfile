FROM continuumio/miniconda3
LABEL authors="hjchcjcjh"

# 设置工作目录
WORKDIR /app

# 复制项目文件到工作目录
COPY . .

SHELL ["/bin/bash", "-c"]

# 更换源
RUN echo "deb http://repo.huaweicloud.com/debian/ bookworm main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb-src http://repo.huaweicloud.com/debian/ bookworm main contrib non-free" >> /etc/apt/sources.list

ARG cache1="alexnet-owt-4df8aa71.pth"
ARG cache2="inception_v3_google-1a9a5a14.pth"
ARG folder="/root/.cache/torch/hub/checkpoints"

# 权重文件
RUN mkdir -p ${folder} && \
    mv ${cache1} ${folder} && \
    mv ${cache2} ${folder}

# 安装依赖
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get install -y ffmpeg && \
    apt-get clean

ARG pip_mirror="https://repo.huaweicloud.com/repository/pypi/simple"
ARG wheel_path="torch-1.8.1+cu111-cp39-cp39-linux_x86_64.whl"

# 安装环境
RUN conda init bash && \
    source ~/.bashrc && \
    conda create -n evaluate python==3.9.0 -y && \
    conda activate evaluate && \
    pip install ${wheel_path} && \
    rm -f ${wheel_path} && \
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html  && \
    conda install -c conda-forge dlib -y && \
    pip install -r requirements.txt -i ${pip_mirror} && \
    pip cache purge && \
    conda clean -a -y


RUN chmod +x entrypoint.sh

# 使用 entrypoint.sh 脚本作为入口点
ENTRYPOINT ["/app/entrypoint.sh"]