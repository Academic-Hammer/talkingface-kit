# **MetaPortrait项目配置文档**

## 下载Resources文件夹到项目根目录里，里面有7个文件，假设项目根目录是MetaPortrait

https://drive.google.com/drive/folders/1f_SXHhLVVsPXiIQstnJDAQYbeZ-26p2K?usp=sharing

### 在项目根目录下执行`python init.py`，会自动将下面文件复制到正确路径

ckpt_base.pth.tar文件复制路径：MetaPortrait/base_model/checkpoint/

shape_predictor_68_face_landmarks.dat文件复制路径：MetaPortrait/val/

vggface2.pt文件复制路径：MetaPortrait/val/

GFPGANv1.3.pth文件复制路径：MetaPortrait/sr_model/pretrained_ckpt/

temporal_gfpgan.pth文件复制路径：MetaPortrait/sr_model/pretrained_ckpt/

HDTF_warprefine文件夹复制路径：MetaPortrait/sr_model/data/

big-lama-20241220T131240Z-001.zip文件复制路径：MetaPortrait/Inpaint-Anything/pretrained_models/

## 在项目根目录构建镜像

`docker build -t metaportrait .`

`docker可能需要添加源：`

`"registry-mirrors": [`
    `"https://docker.hpcloud.cloud",`
    `"https://docker.m.daocloud.io",`
    `"https://docker.unsee.tech",`
    `"https://docker.1panel.live",`
    `"http://mirrors.ustc.edu.cn",`
    `"https://docker.chenby.cn",`
    `"http://mirror.azure.cn",`
    `"https://dockerpull.org",`
    `"https://dockerhub.icu",`
    `"https://hub.rat.dev"`
  `]`

如果构建过程中某个包的安装耗时太久，可在Dockerfile中将其注释，待构建完成再手动换源安装。

例如`RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`可能需要很长时间下载，有时候需要开魔法。

## 添加NVIDIA包仓库的GPG密钥

`curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg`

## 添加NVIDIA包仓库

`curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list`

## 更新包列表

`sudo apt-get update`

## 安装NVIDIA Container Toolkit

`sudo apt-get install -y nvidia-container-toolkit`

## 配置Docker daemon以使用NVIDIA Container Runtime

`sudo nvidia-ctk runtime configure --runtime=docker`

## 重启Docker daemon

`sudo systemctl restart docker`

## 添加数据方法

1. ### 在Dockerfile添加代码，重新构建

   \# 将数据文件添加到镜像的指定路径 

   `COPY <本地数据路径> <镜像内路径>`

   `例如 COPY data_file.txt /app/data/`

2. ### 启动容器，复制到容器中，再提交容器为新的镜像

   `docker run -dit --name temp_container metaportrait`

   `docker cp <本地文件路径> temp_container:<容器内路径>`

   `docker commit temp_container metaportrait`

   `docker rm -f temp_container`

3. ### 挂载主机目录

   `docker run --rm \`
   `-v /path/to/your/input_image:/app/input_image \`
   `-v /path/to/your/input_audio_text:/app/input_audio_text \`
   `-v /path/to/your/output_dir:/app/output_dir \`
   `metaportrait`

## 以交互模式启动容器，可以进入容器内部执行操作

`docker run --gpus all --shm-size=8g -it metaportrait /bin/bash`

`conda activate meta_portrait_base`

## 如果GPU只支持cuda11，要重新安装

`pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116`

## 添加数据，切换到正确目录并执行推理 ，运行以下命令生成 256x256 分辨率的重建结果

`cd base_model` 

`docker cp <src_0.png> container:</app/data/src_0.png>`

`python inference.py --save_dir result --config config/meta_portrait_256_eval.yaml --ckpt checkpoint/ckpt_base.pth.tar`

## 训练扭曲网络

`python main.py --config config/meta_portrait_256_pretrain_warp.yaml --fp16 --stage Warp --task Pretrain`

## 修改 `config/meta_portrait_256_pretrain_full.yaml` 中 `warp_ckpt` 的路径，进行联合训练

`python main.py --config config/meta_portrait_256_pretrain_full.yaml --fp16 --stage Full --task Pretrain`

## 在标准预训练检查点的基础上，使用元学习优化模型的个性化速度

`python main.py --config config/meta_portrait_256_meta_train.yaml --fp16 --stage Full --task Meta --remove_sn --ckpt /path/to/standard_pretrain_ckpt`

<!--如果报错：AttributeError: 'DistributedDataParallel' object has no attribute '_sync_params_and_buffers'，把train_ddp.py第199行给注释了-->

`\# G_full._sync_params_and_buffers()`

## 进入`sr_model`目录，运行以下命令增强基础模型的输出

<!--需要先把base_model/result的MP4文件删了-->

`cd ..`

`cd sr_model`

`cd Basicsr`

`python setup.py develop` <!--以开发模式安装 Python 包-->

`cd ..`

`python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 Experimental_root/test.py -opt options/test/same_id_demo.yml --launcher pytorch`

## 模拟训练

`CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 Experimental_root/train.py -opt options/train/train_sr_hdtf.yml --launcher pytorch`

<!--如果遇到报错：ZeroDivisionError: integer division or modulo by zero，把命令中的CUDA_VISIBLE_DEVICES=1改为CUDA_VISIBLE_DEVICES=0-->

## 以下是容器外输入的评价指令，具体细节和示例文件可以参考`app/val/README`文件

<!--/app/val目录里提供三个文件夹供挂载：src_0、generated_folder和driving_folder-->

## AED:

`docker run ... aed "generated_folder" "driving_folder"`

示例：`docker exec 容器名或id /entrypoint.sh aed /app/val/Jae-in /app/val/base_model`

## APD:

`docker run ... apd "generated_folder" "driving_folder"`

示例：`docker exec 容器名或id /entrypoint.sh apd /app/val/Jae-in /app/val/base_model`

## ID loss:

`docker run ... id_loss "src_0.png" "generated_folder"`

示例：`docker exec 容器名或id /entrypoint.sh id_loss "src_0.png" /app/val/Jae-in`

## LPIPS:

`docker run ... lpips "driving_folder" "generated_folder"`

示例：`docker exec 容器名或id /entrypoint.sh lpips /app/val/base_model /app/val/Jae-in`

## LSE-C:

`docker run ... lse_c "generated_folder"`

示例：`docker exec 容器名或id /entrypoint.sh lse_c /app/val/Jae-in`

## LSE-D:

`docker run ... lse_d "generated_folder" "driving_folder"`

示例：`docker exec 容器名或id /entrypoint.sh lse_d /app/val/Jae-in /app/val/base_model`

## SSIM:

`docker run ... ssim "generated_folder" "driving_folder"`

示例：`docker exec 容器名或id /entrypoint.sh ssim /app/val/Jae-in /app/val/base_model`

## FID:

`docker run ... fid "driving_folder" "generated_folder"`

示例：`docker exec 容器名或id /entrypoint.sh fid "base_model" "Jae-in"`

## NIQE:

`docker run ... niqe "generated_folder"`

示例：`docker exec 容器名或id /entrypoint.sh niqe /app/val/Jae-in`

## 修复背景，会生成修复后的照片(要求安装cuda11.8及以上版本)

`cd Impaint_Anything`

`python replace_background.py \`

  `--image1_path 参考照片路径 \`

  `--image2_path 背景缺陷照片路径 \`

  `--output_path fixed_output.jpg \`

  `--point_coords 90 90 `<!--该坐标需指向人脸-->

示例：`python replace_background.py --image1_path src_0.png --image2_path src_0_broken.png --output_path fixed_output.jpg --point_coords 128 128`