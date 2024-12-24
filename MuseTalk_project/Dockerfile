FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

COPY ./Miniconda3-latest-Linux-x86_64.sh /workspace/Miniconda3-latest-Linux-x86_64.sh

RUN apt-get update && apt-get install -y wget bzip2 && apt-get clean && \
    bash /workspace/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm /workspace/Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda clean -a

ENV PATH="/opt/conda/bin:$PATH"

WORKDIR /workspace

COPY ./MuseTalk /workspace

COPY environments.yaml /workspace/environments.yaml

COPY requirements.txt /workspace/requirements.txt

RUN conda config --add channels nvidia && \
    conda config --set show_channel_urls yes && \
    conda config --set channel_priority false && \
    conda config --set remote_connect_timeout_secs 60 && \
    conda config --set remote_read_timeout_secs 120 && \
    conda config --set remote_max_retries 5


RUN conda env create -f /workspace/environments.yaml

RUN conda clean -a && \
    echo "conda activate muse" > ~/.bashrc

RUN conda init bash

SHELL ["conda", "run", "-n", "muse", "/bin/bash", "-c"]

ENV FFMPEG_PATH=/workspace/ffmpeg-4.4-amd64-static
ENV PATH="${FFMPEG_PATH}:$PATH"

RUN pip install omegaconf==2.3.0
RUN pip install numpy==1.23.5

RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get update && apt-get install -y libglib2.0-0
RUN pip install huggingface-hub==0.25.1
RUN mim install "mmcv==2.1.0"

COPY ./inference.py /workspace/scripts/inference.py

RUN pip install scikit-image==0.24.0
RUN pip install scenedetect
RUN pip install python_speech_features

CMD ["bash"]