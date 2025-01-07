# 使用支持 Conda 的基础镜像，假设 Conda 已经预装
FROM continuumio/miniconda3:latest

# 设置工作目录
WORKDIR /app

# 将当前目录下的所有文件复制到镜像的 /app 目录
COPY . /app/

# 创建虚拟环境
RUN conda create -n diffused-heads python=3.9 -y

# 激活虚拟环境并安装 pip 依赖和 Conda 依赖
RUN /bin/bash -c "source activate diffused-heads && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /app/requirements.txt && \
    conda install -n diffused-heads -c conda-forge ffmpeg openh264 -y"

# 使用 CMD 启动 Python 脚本，直接在 diffused-heads 环境中运行
ENTRYPOINT ["conda", "run", "-n", "diffused-heads", "python", "/app/sample.py"]

# 如果没有传递命令参数，默认显示帮助信息
CMD ["--help"]
