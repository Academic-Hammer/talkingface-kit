# MuseTalk
>https://github.com/TMElyralab/MuseTalk/tree/main

## 登录synapse账户

```
docker login docker.synapse.org
```

## 拉取镜像
```
docker pull docker.synapse.org/syn64432387/musetalk:latest
```

## 运行docker容器

```
docker run --shm-size=16g -it --gpus all docker.synapse.org/syn64432387/musetalk
```

其中 `--shm-size=16g`意味着为docker容器分配16G内存，可以根据本地情况调整，但注意若分配内存较小可能导致因运行时内存不足报错

## 激活环境

```
conda activate muse
```

## 1.运行inference(指令和github仓库相同)
```
python -m scripts.inference --input_image 输入的视频/图片路径 --input_audio 输入的音频路径 --output_dir 结果保存路径
```
若不设置output_dir，则结果保存在/workspace/result

## 2.运行评测
首先准备数据放在docker容器的根目录下，在/workspace下运行指令
```
python demo.py --input_image 输入的视频/图片路径 --input_audio 输入的音频路径 --ground_truth 参考视频路径
```
若不设置参考视频，将以输入视频作为参考视频。评测结果保存在/workspace/evaluation