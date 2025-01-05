# DaGAN 人脸说话视频生成模型复现

本项目复现了CVPR2022论文《Depth-Aware Generative Adversarial Network for Talking Head Video Generation》中的DaGAN模型。该模型通过引入深度感知机制来生成更加自然的人脸说话视频。

## 复现效果

下面的gif图链接展示的是复现效果的对比动图，从上到下依次是训练300轮的结果（2x4090RTX）、训练365轮的结果（在300轮的基础上使用4x4090RTX）、作者提供的最佳效果的checkpoint的生成结果和原视频，可点击链接或在dagan/assets目录下查看 

- [gif图链接](https://github.com/nova728/talkingface-kit/blob/main/dagan/assets/result.gif)

## 复现效果

下面的gif图链接展示的是复现效果的对比动图，从上到下依次是训练300轮的结果（2x4090RTX）、训练365轮的结果（在300轮的基础上使用4x4090RTX）、作者提供的最佳效果的checkpoint的生成结果和原视频，可点击链接或在dagan/assets目录下查看 

- [gif图链接](https://github.com/nova728/talkingface-kit/blob/main/dagan/assets/result.gif)


## 环境要求

- Python >= 3.7（复现使用环境3.8.10）
- CUDA >= 11.0（复现使用CUDA11.3）
- Docker Engine >= 27.3
- NVIDIA GPU（复现使用4090RTX）

## 预训练模型

预训练模型下载链接:

- [人脸深度网络模型](https://hkustconnect-my.sharepoint.com/:f:/g/personal/fhongac_connect_ust_hk/EkxzfH7zbGJNr-WVmPU6fcABWAMq_WJoExAl4SttKK6hBQ?e=fbtGlX)
- [DaGAN生成器模型](https://hkustconnect-my.sharepoint.com/:f:/g/personal/fhongac_connect_ust_hk/EjfeXuzwo3JMn7s0oOPN_q0B81P5Wgu_kbYJAh7uSAKS2w?e=XNZl3K)

下载后请将模型文件放置在depth/models/weights_19目录下。

## Checkpoint下载

从以下链接下载：

- [checkpoints](https://drive.google.com/drive/folders/1jcMVCvl4eYK5P5mCRftnVmk-zzquDpt_?usp=drive_link)

链接中包含

- 00000299-checkpoint.pth.tar（训练300轮得到结果）
- 00000364-checkpoint.pth.tar（训练365轮得到结果）
- DaGAN_vox_adv_256.pth.tar（原作者提供的最优训练结果）

## Docker使用说明

本项目提供了3个Docker镜像:
1. 效果展示镜像(dagan)
2. 评估图像生成镜像(generate-image)
3. 模型评估镜像(evaluation-image)



### 1.生成测试镜像

#### 构建镜像

```
docker build -f Dockerfile -t dagan .
```

#### 运行镜像

```bash
docker run --gpus all `
    -v ${PWD}/input_image:/app/input_image `
    -v ${PWD}/input_video:/app/input_video `
    -v ${PWD}/output_dir:/app/output_dir `
    -v ${PWD}/checkpoints:/app/checkpoints `
    dagan `
    --config config/vox-adv-256.yaml `
    --driving_video /app/input_video/driving.mp4 `
    --source_image /app/input_image/src.png `
    --checkpoint /app/checkpoints/00000299-checkpoint.pth.tar `
    --relative `
    --adapt_scale `
    --kp_num 15 `
    --generator DepthAwareGenerator `
    --result_video /app/output_dir/result.mp4
```

#### 参数说明

- driving_video：驱动视频的路径
- source_image：源图像的路径
- checkpoints：可选checkpoints文件夹下使用哪个checkpoint(需要下载)
- output_dir：生成视频输出路径

#### 可能的错误

如果输入视频过长或者size不是256 x 256可能会出现中断的情况，但是在云服务器中是可以实现的。

### 2. 生成测试数据集

#### 数据集下载

构建docker镜像之前需要先下载驱动和源图像数据集：

- [评估使用的数据集](https://drive.google.com/drive/folders/1ZGL0uu-xuju7fACQSp0Us6Mc8aRnInSf?usp=drive_link)

分享的链接中结构如下：

```tex
├─generate_299（00000299-checkpoint.pth.tar的交叉验证集生成图像）
├─generate_299_self（00000299-checkpoint.pth.tar的自重演验证集生成图像）
├─generate_364（00000364-checkpoint.pth.tar的交叉验证集生成图像）
├─generate_364_self（00000364-checkpoint.pth.tar的自重演验证集生成图像）
├─generate_form（DaGAN_vox_adv_256的交叉验证集生成图像）
├─generate_form_self（DaGAN_vox_adv_256的自重演验证集生成图像）
├─gt_cross（交叉验证集驱动图像）
├─gt_self（自重演验证集驱动图像）
├─source_cross（交叉验证集源图像）
└─source_self（自重演验证集源图像）
```

其中generate为前缀的是已经生成完成的图像集。

#### 构建&运行镜像(可选)

```bash
docker build -f Dockerfile.generate -t generate-image .
```

如果直接下载生成图像的文件夹则**不用构建此镜像**，下载好的文件以原名放置到evaluation_set目录下。运行语句如下：

```bash
docker run --gpus all `
-v "$(pwd)/config:/app/config" `
-v "$(pwd)/evaluation_set:/app/evaluation_set" `
-v "$(pwd)/results:/app/results" `
-v "$(pwd)/checkpoints:/app/checkpoints" `
generate-image `
--config "/app/config/vox-adv-256.yaml" `
--source_dir "/app/evaluation_set/source_cross" `
--driving_dir "/app/evaluation_set/gt_cross" `
--save_folder "/app/results" `
--generator "DepthAwareGenerator" `
--checkpoint "/app/checkpoints/00000299-checkpoint.pth.tar" `
--kp_num 15
```

#### 参数说明:

- source_dir: 源图像路径
- driving_dir: 驱动图像路径
- output: 输出目录路径
- relative: 使用相对坐标(推荐)
- adapt_scale: 自适应缩放(推荐)
- kp_num: 关键点数量，默认15(使用其他数值报错)

#### 

### 3. 模型评估

#### 构建镜像

```bash
docker build -f Dockerfile.evaluate -t evaluation-image .
```

#### 运行镜像

```bash
docker run --gpus all `
  -v "$(pwd)/evaluation_set:/app/evaluation_set" `
  -v "$(pwd)/evaluation_results:/app/evaluation_result" `
  evaluation-image `
  --base_dir /app/evaluation_set `
  --output_path /app/evaluation_result `
  --device cuda `
  --batch_size 1 `
  --generate_dirs generate_299_self
```

#### 参数说明:

- 必选参数：

  - base_dir: 数据集路径（evaluation_set）

  - output_path: 评估结果输出路径

  - generate_dirs:使用evaluation_set中的哪个生成图像数据集，请**务必使用上一节中提到的目录结构的原名称**，代码根据名称自动匹配使用交叉验证集还是自重演验证集的驱动、源图像文件夹。

    > 代码逻辑是后缀为_self匹配到自重演数据集，否则匹配到交叉验证数据集。

- 可选参数：

  - save_visuals：单独保存所有源图像、驱动图像和生成图像对。
  - max_samples：选择评估数据集大小（<=2000）

评估指标包括:
- NIQE (无参考图像质量评估)
- PSNR (峰值信噪比)
- SSIM (结构相似性)
- FID (Fréchet Inception Distance)【fid的运算由于evaluation.py中有误，后来在calculate_fid.py中实现】
- LSE-C (内容一致性评估)
- LSE-D (驱动一致性评估)

评估结果将以JSON格式保存在output目录下,并以txt的格式保存最终结果。
