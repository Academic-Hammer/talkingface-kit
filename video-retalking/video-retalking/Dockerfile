FROM continuumio/miniconda3

WORKDIR /app
COPY . /app

ENV PATH /opt/conda/envs/video_retalking/bin:$PATH
ENV CONDA_DEFAULT_ENV video_retalking

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 libgl1 libgl1-mesa-glx -y

# 创建Conda环境并安装所有依赖项
RUN conda create -n video_retalking python=3.10 \
    && /bin/bash -c "source activate video_retalking" \
    && echo "conda activate video_retalking" >> ~/.bashrc \
    && conda install -c anaconda cmake \
    && conda install -c conda-forge dlib \
    && pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118 \
    && pip install -r /app/requirements.txt

ENTRYPOINT ["conda", "run", "-n", "video_retalking", "python", "inference.py"]

# RUN conda create -n video_retalking python=3.10 \
#     && exec bash \
#     && source activate \
#     && conda activate video_retalking \
#     && conda install -c anaconda cmake \
#     && conda install -c conda-forge dlib
#     && pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118 \
#     && pip install -r /app/requirements.txt \

# RUN chmod u+x /app/start.sh
# ENTRYPOINT ["/bin/bash", "-c", "source activate video_retalking && python inference.py"]

# CMD "python" "inference.py" $PARAMS

# FROM app:new

# ENV PATH /opt/conda/envs/video_retalking/bin:$PATH
# ENV CONDA_DEFAULT_ENV video_retalking

# ENTRYPOINT ["conda", "run", "-n", "video_retalking", "python", "inference.py"]