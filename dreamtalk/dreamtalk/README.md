### 部署到本地

```shell
git clone https://github.com/Academic-Hammer/talkingface-kit/dreamtalk.git
cd dreamtalk
mkdir output_video
mkdir jonatasgrosman/wav2vec2-large-xlsr-53-english
```

### 下载检查点与模型

在构筑 docker 前需要下载[检查点文件](https://drive.google.com/drive/folders/1MUrhcxLSLwcv76QSgtT-aV9c24DMBxiP)到 `/dreamtalk/checkpoints/`, 文件目录格式如下：

📦checkpoints  
 ┣ 📜denoising_network.pth  
 ┗ 📜renderer.pt

同时您还需要下载用于英语语音识别的微调 [XLSR-53 模型](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english/tree/main)到 `/jonatasgrosman/` 目录下，必要的文件目录如下：

> 不下全会在加载模型时产生警告信息（参数加载的警告，有剩余的参数好像是，不影响使用），据说将链接内的全下下来可以解决，未经测试

📦jonatasgrosman  
 ┗ 📂wav2vec2-large-xlsr-53-english  
 ┃ ┣ 📜config.json  
 ┃ ┣ 📜preprocessor_config.json  
 ┃ ┣ 📜pytorch_model.bin  
 ┃ ┗ 📜vocab.json  

#### 下载镜像

> 不建议，镜像文件很大，建议本地构筑，其实我传了一个早期的版本在 dockerhub 上，不知道搜 dreamtalk 能不能找到，不过好像是私有的

好像我构筑的有亿点点大，虽然是两位数，但是单位是 GB hh ( 上传好浪费时间，我的建议是自己构筑，如果一定需要可以通过 GitHub 或者邮箱: wincomso9@outlook.com 联系我 )

下载好后放到 `dreamtalk` 目录下，然后执行以下命令

> 可以 `docker images` 看加载是否成功

```bash
    docker load -i dreamtalk.tar
```

### demo Docker

#### 构筑 Docker

```
docker build -f .dockerfile -t dreamtalk:v1 .
```

#### 运行 demo

```bash
docker run --gpus all `
    -v ${PWD}/data:/dreamtalk/data `
    -v ${PWD}/output_video:/dreamtalk/output_video `
    -v ${PWD}/checkpoints:/dreamtalk/checkpoints `
    -v ${PWD}/jonatasgrosman:/dreamtalk/jonatasgrosman `
    -v ${PWD}/tmp:/dreamtalk/tmp `
    dreamtalk:v1 `
    --wav_path /dreamtalk/data/audio/acknowledgement_chinese.m4a `
    --style_clip_path /dreamtalk/data/style_clip/3DMM/M030_front_neutral_level1_001.mat `
    --pose_path /dreamtalk/data/pose/RichardShelby_front_neutral_level1_001.mat `
    --image_path /dreamtalk/data/src_img/uncropped/male_face.png `
    --cfg_scale 1.0 `
    --max_gen_len 30 `
    --output_name demo
```

也即

```bash
docker run --gpus all -v ${PWD}/data:/dreamtalk/data -v ${PWD}/output_video:/dreamtalk/output_video -v ${PWD}/checkpoints:/dreamtalk/checkpoints -v ${PWD}/jonatasgrosman:/dreamtalk/jonatasgrosman -v ${PWD}/tmp:/dreamtalk/tmp dreamtalk:v1 --wav_path /dreamtalk/data/audio/acknowledgement_chinese.m4a --style_clip_path /dreamtalk/data/style_clip/3DMM/M030_front_neutral_level1_001.mat --pose_path /dreamtalk/data/pose/RichardShelby_front_neutral_level1_001.mat --image_path /dreamtalk/data/src_img/uncropped/male_face.png --cfg_scale 1.0 --max_gen_len 30 --output_name demo
```

#### 参数说明

+ `--wav_path` 

    输入音频 ( 视频也可, 会从其中提取音频, 兼容格式: m4a, wav, mp4 等  )

+ `--style_clip_path` 

    面部运动参考

+ `--pose_path`

    头部姿势参考

+ `--image_path`

    输入图像

+ `--cfg_scale`

    风格的强度参数

+ `--max_gen_len`

    生成视频的最大长度，单位是秒

+ `--output_name demo`  

    输出文件名 

### 从评估视频中提取 3DMM 参数

#### 环境配置

需要与底层交互，环境依赖于 CUDA 驱动 ( 而不是 pytorch 安装的 cuda-tookit ), docker 环境也需要参考 linux 或 WSL 安装的驱动 ( WSL 调用的实际是 windows 的显卡驱动 )

> 如果提示 pytorch 支持的算力和当前显卡不匹配需要升级 pytorch：

```bash
conda create -n DP python=3.9
conda activate DP
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
# 不要按照论文项目网站使用对应的代码，存在版本问题
pip install face_alignment
pip install ffmpeg-python
pip install kornia
pip install trimesh
pip install Ninja

# nvdiffrast 经过个人修改，现在在 WSL 支持使用 CUDA，而不是 OpenGL 进行渲染（
# WSL 上 OpenGL 存在版本限制，无法满足 nvdiffrast 的版本需求，通过微软的强制指令升级后也会与 CUDA 驱动冲突 
# 而项目只要检测到 OpenGL 就会使用 OpenGL 而不是 CUDA
# 因此我修改了 nvdiffrast 的源码，将其改为仅使用 CUDA 进行渲染
cd nvdiffrast
pip install .
```

按照[教程](https://blog.csdn.net/Sihang_Xie/article/details/127347139)安装 cuda 官网给的安装包后添加环境变量

> 很详细的博客，WSL 照着做就是了，云服务器需要选择 cuda=12.4 的预装驱动，要不还要自己升级很麻烦

#### 提取 3DMM 参数

将视频放入对应文件夹运行即可，我在 16G/i9/4060 下使用的参数如下，在 `face_recon_videos.py` 可以进行第二段命令的参数调整：

> 需要较多的内存，WSL 环境需要修改 .wslconfig 文件中的限制，评估视频的使用的最多内存大概在 26G 左右

```bash
python extract_kp_videos.py --input_dir data/input --output_dir data/keypoint --device_ids 0 --workers 6
python face_recon_videos.py --input_dir data/input --keypoint_dir data/keypoint --output_dir data/output --inference_batch_size 200 --name=test --epoch=20 --model facerecon
```

### 生成评估视频

将上面生成的 mat 文件放入 dreamtalk/data/eval/pose/ 下即可，在推理时修改参数即可

需要注意的是，dreamtalk 的推理几乎无法在本地运行( 单卡推理且未分段，直接推理长视频 )，最短的视频使用显存也超过了 24G ( 还是 32G )，我在 A800 ( 80G ) 上完成了推理，但是其中 Macron 的视频需要的显存在 95 G 左右，因此我将其分段后重新提取了两段视频的 3DMM 参数，推理后再将其拼接成一个视频。

> 分段推理在理论上是可行的，但是这样在提取 3DMM 的时候就需要进行分段 
> ( 或许有时间看看3DMM 的论文看看 mat 里面的数据结构就不需要在提取时分段，可以在推理时处理数据，但是来不及了hh )


### 模型评估

具体参见 `/syncnet_python/` 和 `eval/`

此处不加描述