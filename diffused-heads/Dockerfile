# 使用支持 Conda 的基础镜像，假设 Conda 已经预装
FROM continuumio/miniconda3:latest

# 设置工作目录
WORKDIR /app

# 将当前目录下的所有文件复制到镜像的 /workspace 目录
COPY . /app/

# 创建虚拟环境
RUN conda create -n diffused-heads python=3.9 -y

# # 设置 HTTP 和 HTTPS 代理
# ENV http_proxy=http://127.0.0.1:2340
# ENV https_proxy=http://127.0.0.1:2340


# 激活虚拟环境并安装 pip 依赖和 Conda 依赖
# 注意：确保使用 /workspace/requirements.txt 作为 pip 安装的路径
RUN /bin/bash -c "source activate diffused-heads && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /app/requirements.txt && \
    conda install -n diffused-heads -c conda-forge ffmpeg openh264 -y"


# 切换到虚拟环境作为默认环境
RUN echo "conda activate diffused-heads" >> ~/.bashrc

# 启动容器时默认进入虚拟环境
CMD ["/bin/bash"]
