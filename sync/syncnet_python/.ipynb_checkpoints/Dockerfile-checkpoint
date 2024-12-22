# 使用官方的 Python 3.9.0 镜像作为基础镜像
FROM python:3.9.0

# 设置工作目录为 /app
WORKDIR /app

# 复制项目文件到容器中
COPY . /app

# 安装 requirements.txt 中列出的所有依赖
RUN pip install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN apt-get update && apt-get install -y libgl1-mesa-glx && apt-get install -y ffmpeg && apt-get clean

# 默认执行命令
CMD ["./evaluate.sh"]
