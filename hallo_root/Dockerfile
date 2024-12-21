# 基础镜像：官方 Miniconda 镜像
FROM continuumio/miniconda3:latest

# 设置工作目录
WORKDIR /usr/src/app

# 复制 environment.yml 文件到镜像
COPY environment.yml .

# 创建 Conda 环境并清理缓存
RUN conda env create -f environment.yml && conda clean -a

# 激活环境并设置默认环境
RUN echo "source activate hallo_measurement" > ~/.bashrc
ENV PATH /opt/conda/envs/hallo_measurement/bin:$PATH

# 复制项目代码到镜像
COPY . .

# 默认执行命令
CMD ["python", "main.py"]