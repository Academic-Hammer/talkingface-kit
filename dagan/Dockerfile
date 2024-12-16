FROM silverlogic/python3.8:latest

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV TORCH_HOME=/app/.torch

# 使用清华源
RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ && \
    pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

# 设置工作目录
WORKDIR /app

# 复制需要的文件
COPY requirements.txt /app/
COPY entrypoint.sh /app/
COPY . /app/

# 安装 PyTorch 和相关依赖 
RUN pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# 安装其他关键依赖 
RUN pip install \
    numpy==1.21.6 \
    scipy==1.7.3 \
    scikit-image==0.16.2 \
    opencv-python==4.6.0.66 \
    face-alignment==1.3.4 \
    matplotlib==3.5.1 \
    Pillow==9.1.0 \
    PyYAML==5.4.1 \
    tqdm==4.64.0 \
    imageio==2.17.0 \
    imageio-ffmpeg==0.4.7 \
    networkx==2.6.3 \
    scikit-learn==1.0.2

# 安装 requirements.txt 中的依赖
RUN pip install -r requirements.txt

# 安装 face-alignment 
RUN cd face-alignment && \
    python setup.py install

# 创建必要的目录 
RUN mkdir -p /app/input_image \
    /app/input_audio_text \
    /app/output_dir \
    /app/checkpoints

# 设置入口脚本权限 
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]