# MakeItTalk - 语音驱动面部动画生成

组长：王硕，组员：杨勇、付博文、葛钧满

项目修改完善主要贡献者：杨勇



在项目前先讲讲我的一些感受，这个项目我基本上是全程参与，每个部分我都在做，我也不知道我的组员和队长到底做了啥，其实真的很崩溃因为一个人第一次做项目，还有或多或少的问题，为了这个项目不知道熬了多少个夜晚，但是因为个人能力的确有效，也是第一次做这种项目，也很多不懂的地方，最后做的也不是很好，希望大家体谅。

也非常感谢这次机会，其实让我学习到了很多很多，从最开始的什么都不会，到后续自己搭建云服务器，处理里面的各种问题，学语音识别相关的知识，看论文，学git，学docker等等，无论怎样，这都是非常棒的一次学习经历。       --杨勇

//docker最后精力实在不够了，没有做好，非常抱歉。

**MakeItTalk 是一个基于语音输入生成面部动画的开源项目。它可以将单张人像照片与语音结合，生成逼真的说话动画。**

## 快速开始

### 环境准备
1. 安装Python 3.8环境：
```bash
conda create -n makeittalk_env python=3.8
conda activate makeittalk_env
```

2. 安装依赖：（pynormalize库现在只能从GitHub上找原库拷贝过来）
```bash
pip install -r requirements.txt
git clone https://github.com/giannisterzopoulos/pynormalize.git
cd pynormalize
pip install .
```

3. 安装FFmpeg：
```bash
sudo apt-get install ffmpeg
```

4. 安装winehq-stable：这个是用来生成卡通人物的，目前测试可以在ubuntu 18.04上完成，注意不同的ubuntu版本需要下载不同的wine。
sudo dpkg --add-architecture i386
wget -nc https://dl.winehq.org/wine-builds/winehq.key
sudo apt-key add winehq.key
sudo apt-add-repository 'deb https://dl.winehq.org/wine-builds/ubuntu/ bionic main'
sudo apt update
sudo apt install --install-recommends winehq-stable

### 运行示例
1. 准备256x256的人像图片（jpg格式）和语音文件（wav格式），放入文件夹中。

2. 运行以下命令生成动画：
```bash
python main_end2end.py --jpg "the way to your portrait.jpg" --wav "the way to your audio.wav"
```

3. 生成的动画将保存为`portrait_pred_fls_audio_audio_embed.mp4`(这里portrait是图片的名字，第一个audio是原语音文件的名字)

### 视频处理
除此之外，我还编写了process_video.py来进行对视频文件的提取，
该文件会从视频中提取音频，还会截取视频第100帧并裁剪为256x256的肖像图片并保存。
运行命令如下：
```bash
python process_video.py --video 视频路径 --jpg_output 输出图片路径 --wav_output 输出音频路径
```

示例如下
```bash
python process_video.py --video evaluate/data/raw/videos/Obama.mp4 --jpg_output examples/Obama.jpg --wav_output examples/Obama.wav
```
该命令会：
    1.从视频中提取音频并保存为 examples/Obama.wav
    2.截取视频第100帧并裁剪为256x256的肖像图片，保存为 examples/Obama.jpg

### 模型评估
这里我们采用了PSNR、SSIM、LPIPS的标准来进行评估。
评估函数为evaluate_metrics.py
这里我就没用命令行模式来搞了，因为也只有我会来测试，就直接在代码里修改。
只需要在其124行到127行去配置评估文件路径即可。源码贴在下面

```python
real_video = "evaluate/data/raw/videos/Shaheen.mp4"
generated_video = "examples/Shaheen_pred_fls_Shaheen_audio_embed.mp4"
audio_file = "examples/Shaheen.wav"
```
每次评估的时候自助修改下就好。

## 详细使用说明

### 参数说明
- `--jpg`: 输入人像图片路径，我们这里工作目录是最开始的Makeittalk的大文件夹，可以使用绝对路径，也可以使用相对路径。
- `--wav`: 输入语音文件路径  
- `--amp_lip_x`: 嘴唇水平运动幅度（默认2.0）
- `--amp_lip_y`: 嘴唇垂直运动幅度（默认2.0）
- `--amp_pos`: 头部运动幅度（默认0.5）

### 示例
```bash
# 生成更夸张的动画效果
python main_end2end.py --jpg examples/Obama.jpg --wav examples/Obama.wav --amp_lip_x 3.0 --amp_lip_y 3.0 --amp_pos 1.0
```

## 常见问题

### Q: 生成的动画不自然怎么办？
A: 尝试调整`--amp_lip_x`和`--amp_lip_y`参数，找到最适合的幅度值

### Q: 如何生成卡通人物动画？
A: 使用`main_end2end_cartoon.py`脚本，并准备透明背景的卡通图片

非常抱歉，其实还有好多一些问题，但是都没有记录下来，做项目的经验太少了
如果遇到了问题，可以联系我的邮箱：3011689827@qq.com

然后其实做了很多的一些尝试等等，但到写的时候却都忘了，真的做了很多努力，无愧我心。

## 许可证
本项目采用 [MIT License](LICENSE.md)
