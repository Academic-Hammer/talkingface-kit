# 使用官方的 Python 镜像作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 复制当前目录下的所有文件到容器中的工作目录
COPY . /app/

# 安装项目的依赖
RUN pip install --no-cache-dir -r requirements.txt

# 设置容器启动时默认执行的命令
CMD ["python", "Evaluate/main.py"]
