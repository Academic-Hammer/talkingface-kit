# DiffDub

[DiffDub原项目readme](README-DIffDub.md)

## 修改的部分

### 1. 修改了环境搭建方式

现在的环境搭建方式如下

首先创建`diffdub`的conda环境

```bash
    conda create -n diffdub python==3.9.0
    conda activate diffdub
```

然后安装`1.8.1`版本的`pytorch`, 安装命令如下(此处假设cuda版本至少为`11.1`)

```bash
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```
接下来安装`requirements.txt`中的全部依赖, 安装命令如下

```bash
    pip install -r requirements.txt
```
最后无视依赖项安装`torchlm`的最新版本`0.1.6.10`, 安装命令如下

```bash
    pip install --no-deps torchlm==0.1.6.10
```

### 2. 提供了dockerfile和entrypoint.sh

现在可以通过dockerfile和entrypoint.sh来构建docker镜像

需要下载3个缓存文件

[torch-1.8.1+cu111-cp39-cp39-linux_x86_64.whl](https://download.pytorch.org/whl/cu111/torch-1.8.1%2Bcu111-cp39-cp39-linux_x86_64.whl)

[resnet18-5c106cde.pth](https://download.pytorch.org/models/resnet18-5c106cde.pth)

[pipnet_resnet18_10x68x32x256_300w.pth](https://github.com/DefTruth/torchlm/releases/download/torchlm-0.1.6-alpha//pipnet_resnet18_10x68x32x256_300w.pth)

将3个文件放到项目根目录下，然后执行以下命令构建镜像

```bash
    docker build -t diffdub .
```

### 3. 修改了demo.py

将原先的取随机帧的方式改为取视频的第一帧

## 使用方法
注意：我们更建议使用docker-images中的方式来复现我们的工作。

#### 1. 按上文修改过的方式搭建conda环境
#### 2. 下载模型权重文件

项目作者提供了一个脚本来下载模型权重文件, 下载命令如下

```bash
    bash ./auto_download_ckpt.sh
```

或者访问huggingface的仓库手动下载([仓库地址](https://huggingface.co/taocode/diffdub))并放到assets文件夹下, 假设项目目录为`/path/to/diffdub`, 则权重文件应该放到`/path/to/diffdub/assets/checkpoints`下

实际上只有checkpoints文件夹下的`stage1_state_dict.ckpt`和`stage2_state_dict.ckpt`是必须的, 其他文件可以不下载

#### 3. 获取音频hubert特征文件

作者提供了另一个项目提取hubert特征, 仓库地址为([hubert](https://github.com/liutaocode/talking_face_preprocessing?tab=readme-ov-file#audio-feature-extraction)), 这里就不再赘述, 请自行查看, 为了方便使用者，我们已经把提取出的特征文件放在网盘中，请参见/docker-images/README.md中的链接。

#### 4. 运行demo.py

作者提供了3种运行方式, 我们仅推荐第一种, 即使用视频第一帧进行推理

```bash
    python demo.py \
    --one_shot \
    --video_inference \
    --saved_path '<destination_path>' \
    --hubert_feat_path '<hubert_file_path>' \
    --wav_path '<input_audio_path>' \
    --mp4_original_path '<input_video_path>' \
    --saved_name '<saved_file_name>' \
    --device '<inference_device>'
```

`<destination_path>`: 推理结果保存路径
`<hubert_file_path>`: hubert特征文件路径
`<input_audio_path>`: 输入音频文件路径
`<input_video_path>`: 输入视频文件路径
`<saved_file_name>`: 推理结果文件名, 默认为`prediction.mp4`
`<inference_device>`: 推理设备, 可选`cpu`或`cuda`, 默认为`cuda`


