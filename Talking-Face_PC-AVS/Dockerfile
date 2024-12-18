#基于python的基础镜像
FROM python:3.6-slim

#代码添加到code文件夹
ADD ./Talking-Face_PC-AVS /code

#设置code文件夹是工作目录
WORKDIR /code

COPY requirements.txt /code/

RUN pip install --upgrade pip -i https://pypi.mirrors.ustc.edu.cn/simple/

RUN sed -i 's|http://deb.debian.org/debian|http://mirrors.aliyun.com/debian|g' /etc/apt/sources.list

RUN apt-get update

RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN apt-get install -y libglib2.0-0

RUN apt-get install -y libglib2.0-dev

RUN apt-get install -y libopencv-dev

RUN apt-get install -y build-essential

RUN apt-get install -y python3-dev

RUN apt-get install -y gcc

RUN apt-get install -y g++

RUN pip install numpy==1.19.5 -i https://pypi.mirrors.ustc.edu.cn/simple/

#conda安装环境
RUN pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/

CMD ["/bin/bash", "/code/experiments/demo_vox_new.sh"]
