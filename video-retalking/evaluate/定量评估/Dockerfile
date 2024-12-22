FROM python:3.10

WORKDIR /app
COPY . /app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 libgl1 libgl1-mesa-glx -y

RUN echo "export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/" >> ~/.bashrc
RUN pip install -r /app/requirements.txt

RUN chmod u+x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]