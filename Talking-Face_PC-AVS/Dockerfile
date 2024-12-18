#基于python的基础镜像
FROM python:3.6

#代码添加到code文件夹
ADD ./Talking-Face_PC-AVS /code

#设置code文件夹是工作目录
WORKDIR /code

#conda安装环境
RUN conda env create -f environment.yml

CMD ["python", "/code/inference.py"]
