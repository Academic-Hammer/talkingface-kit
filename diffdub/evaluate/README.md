# 评估生成视频质量

SyncNet原README [SyncNet README](README-syncnet.md)

## 简述

该项目从SSIM, LPIPS, PSNR, FID, NIQE, LMD, LSE-C, LSE-D这些指标出发，对生成的视频质量进行评估。

## 使用方式

注意：我们更建议使用docker-images中的方式来复现我们的工作。

### 1. 安装依赖

项目需要安装`ffmpeg`, 安装命令如下(假设是`debian`分支的Linux发行版)
```bash
sudo apt-get install ffmpeg
```

然后创建名为`evaluate`的conda环境

```bash
conda create -n evaluate python==3.9.0
conda activate evaluate
```

然后安装pytorch, 安装命令如下(此处假设cuda版本至少为`11.1`)

```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

最后安装`requirements.txt`中的全部依赖, 安装命令如下

```bash
pip install -r requirements.txt
```

至此, 一锤定音 \
尘埃, 已然落定 \
(逃)

### 2. 下载预训练模型

需要下载预训练模型, 可以使用`download_model.sh`下载
```bash
bash download_model.sh
```

或者分别下载并放置到指定位置
[shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

解压并将`shape_predictor_68_face_landmarks.dat`文件放置到项目根目录

[syncnet_v2.model](http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model)

将`syncnet_v2.model`文件放到项目根目录下的`data`文件夹下

[sfd_face.pth](https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth)

将`sfd_face.pth`文件放到项目根目录下的`detectors/s3fd/weights`文件夹下

### 3. 运行评估
    
```bash
python app.py --original_video <原视频路径> --generated_video <生成视频路径> --output_file <输出文件路径>
```
原视频路径和生成视频路径必填, 输出文件路径可选, 如果不填写会输出到控制台。填写任意非空路径会尝试输出到对应文件中。

大致格式如下
```
SSIM: 0.9109151456333645
LPIPS: 0.024414131425321103
PSNR: 34.49526061696783
FID: 15.4655082780489
NIQE: 0.08195970602018815
LMD: 1.9270671585100572
LSE-C: 7.588120937347412
LSE-D: 7.734384059906006
```

### 4. 制作镜像

项目下提供了Dockerfile, 可以通过Dockerfile制作镜像

为了加速构建过程和避免容器中再次下载模型, Dockerfile中使用了3个缓存文件, 下载地址分别如下: 

[torch-1.8.1+cu111-cp39-cp39-linux_x86_64.whl](https://download.pytorch.org/whl/cu111/torch-1.8.1%2Bcu111-cp39-cp39-linux_x86_64.whl) 

[inception_v3_google-1a9a5a14.pth](https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth) 

[alexnet-owt-4df8aa71.pth](https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth)

3个文件均位于项目根目录

然后执行bash命令构建镜像
```bash
docker build -t evaluate .
```