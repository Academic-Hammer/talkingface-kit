# 使用Python 3.8官方镜像
FROM python:3.8-slim-buster

# 设置工作目录
WORKDIR /app

# 安装系统依赖
# 这些是MakeItTalk项目运行所需的基本依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件并安装Python依赖
# 这样可以利用Docker的缓存机制，提高构建效率
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 仅复制必要的代码文件
COPY *.py .
COPY src/ src/
COPY util/ util/
COPY thirdparty/ thirdparty/

# 设置环境变量
# 确保Python输出不被缓冲，方便查看实时日志
ENV PYTHONUNBUFFERED=1

# 设置默认运行命令
# 可以根据需要修改为其他入口文件
CMD ["python", "main_end2end.py"]
