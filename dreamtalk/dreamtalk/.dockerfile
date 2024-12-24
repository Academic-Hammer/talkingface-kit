FROM continuumio/anaconda3:latest

RUN mkdir dreamtalk

# 设置工作目录
WORKDIR /dreamtalk

# 复制需要的文件
COPY environment.yml /dreamtalk/

RUN apt-get update

RUN apt-get install build-essential libgl1 dialog libssl-dev g++ cmake -y

RUN conda env create -f environment.yml

SHELL ["/bin/bash", "-c"]

RUN conda init bash && \
    source ~/.bashrc && \
    conda activate dt && \
    pip install torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install sox


COPY inference_for_demo_video.py /dreamtalk/
COPY generators /dreamtalk/generators
COPY core /dreamtalk/core
COPY configs /dreamtalk/configs
COPY media /dreamtalk/media


# 设置默认命令
ENTRYPOINT ["conda", "run", "-n", "dt", "python", "inference_for_demo_video.py"]


