# EmoTalk大作业

原仓库链接：[psyai-net/EmoTalk_release: This is the official source for our ICCV 2023 paper "EmoTalk: Speech-Driven Emotional Disentanglement for 3D Face Animation" (github.com)](https://github.com/psyai-net/EmoTalk_release)

封装后的可运行项目：https://pan.baidu.com/s/1ZO7TercF4GeucMflwcy9zw?pwd=jdkv

###  本项目用于记录EmoTalk小组大作业的所有工作，包括以下内容：

- **论文学习**：学习相关方法并进行记录。

- **部署环境**：对原仓库代码进行修改以适应不同环境。

- **测试指标**：包括定量测试老师提供的指标和定性测试论文的评价指标。

- **改进创新**：将原仓库的推理模型改为训练，成功复现论文的训练。

  

## 项目结构和组成
```plaintext
实验报告.docx       #二、实验报告，按照北理学报格式
code/              # 我们的代码工作汇总
  ├── EmoTalk_win/      # 修改后的Windows下可运行项目
  ├── test/             # 测试老师测试集的LSE-C，LSE-D评价指标，以及定性评价论文提供的评价指标
  ├── train/            # 将原仓库的推理模型改为训练模型
EmoTalk使用文档.docx        #3.3、封装后的可运行项目配置文档
组内评价.docx        #一、组内分工及组内评价
```


# 文件说明

## 1. 实验报告.docx
包括模型简介，实验困难及解决方案，模型的定性及定量评价结果及可能改进方法

## 2. Code
我们的主要代码工作汇总，包括以下子模块：

### 2.1 EmoTalk
修改后的可运行项目，Windows环境下使用。

**注意事项**：
- 需要手动下载以下模型并放置到指定目录：
  - **`wav2vec2-large-xlsr-53-english`** 和 **`wav2vec-english-speech-emotion-recognition`**，下载后存放至 `models/` 文件夹。
  - **`EmoTalk.pth`**，下载后存放至 `pretrain_model/` 文件夹。
- 其他具体信息参考[环境部署教程](code/EmoTalk_win#readme)。

### 2.2 test
用于测试老师提供的测试集，计算LSE-C，LSE-D评价指标。另外定性评价了论文的几项指标。

### 2.3 train
将原仓库的推理模型修改为训练模型，复现论文训练功能。

## 3. EmoTalk使用文档.docx

详细讲解了如何运行文件以生成视频和评价指标

1.  **Dockers镜像下载**

> 通过代码仓库或百度网盘下载docker镜像百度网盘https://pan.baidu.com/s/1ZO7TercF4GeucMflwcy9zw?pwd=jdkv

2.  **Docker镜像配置**

> 下载后在终端加载docker镜像：docker load \< /path/to/EmoTalk.tar
>
> 加载完毕后，查看镜像是否已经导入：docker images

3.  **运行EmoTalk.py**

> docker run \--gpus all -v \<input_path\> -v \<output_path\> -it
> emotalk bash
>
> 将输入视频和输出结果的路径挂载到容器上，注意一定要添加 \--gpus all
>
> 在容器中执行 python EmoTalk.py \<path to video\>
>
> 运行成功，输出LSE-D和LESE-C
>
> 输出文件保存在输出路径

## 4. 组内评价.docx

省流：都拉满了

