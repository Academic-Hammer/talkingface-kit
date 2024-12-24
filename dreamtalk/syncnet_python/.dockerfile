FROM continuumio/anaconda3:latest

WORKDIR /syncnet_python

# 复制需要的文件
COPY requirements.txt /syncnet_python/

RUN apt-get update

# 确保依赖项不会缺失，不一定需要，只是个人习惯下一些依赖
RUN apt-get install build-essential g++ cmake ffmpeg libgl1 -y

RUN conda create -n SP python=3.9 -y

SHELL ["/bin/bash", "-c"]

RUN conda init bash && \
    source ~/.bashrc && \
    conda activate SP && \
    pip install -r requirements.txt

RUN pip install opencv-python-headless


COPY . /syncnet_python/

RUN sh download_model.sh

# 添加脚本
RUN chmod +x /syncnet_python/run_commands.sh

# 设置默认命令
ENTRYPOINT ["conda", "run", "-n", "SP", "bash", "/syncnet_python/run_commands.sh"]
