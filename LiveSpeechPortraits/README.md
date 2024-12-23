# 语音识别大作业 Live Speech Portraits

## 小组成员

孙乐天，余博轩，孙仲天，李泽鸣

## 文件说明

`./source`：目录下包含全部源代码

`./data`：目录下包含 **部分** 测试用的音视频以及预训练模型

`README.md`：Docker 镜像说明文档

`Paper_README.md`：论文代码的说明文档

## LiveSpeechPortraits Docker 镜像使用说明

本项目的模型使用 `Docker 27.3` 版本封装了模型，使用前请确保已正确安装 `Docker Engine 27.3` 。您可以通过访问 [Docker 官网](https://www.docker.com/) 下载并安装 Docker。

论文配套代码使用 `pytorch-1.7.10+cu110` 环境，我们基于此版本进行了复现，Docker镜像可从此处下载：[lsp_demo_1.3.tar](https://pan.baidu.com/s/1NIJDdSzwFL3lPSb-tYuaZQ?pwd=fnbg) 。

由于使用的 `cuda` 版本较老，最新的RTX40系显卡无法进行训练。因此，面向高版本重新封装了基于 `cuda 11.8` 的Docker镜像。由于原文代码存在兼容性问题，因此我们替换了不兼容的模块，并改写了部分代码。该Docker镜像可从此处下载：[lsp_quickrun_cu118.tar](https://pan.baidu.com/s/1a1C2oy5DBqbVjWnXOOr9rw?pwd=imjb) 。

### 文件准备

#### 1. 从.tar文件载入 Docker 镜像

首先，参考上方链接，下载与您的 GPU 兼容的 Docker 镜像文件。

使用以下命令载入 Docker 镜像：

```bash
docker load -i lsp_demo_XXXXX.tar
```

#### 2. 准备需要使用的数据

在运行 Docker 镜像前，请创建以下三个文件夹，分别用于存放预训练模型、输入音频和输出结果：

`models`：存放预训练模型

`input`：存放输入音频文件

`results`：存放生成的输出结果

您可以通过 Google Drive [下载](https://drive.google.com/drive/folders/1sHc2xEEGwnb0h2rkUhG9sPmOxvRvPVpJ?usp=sharing) 预训练模型，并将其保存至 `models` 文件夹。确保文件夹中的内容如下所示：

```
.
|-- APC_epoch_160.model
|-- May
|-- McStay
|-- Nadella
|-- Obama1
`-- Obama2
```

将您需要输入模型的数据文件保存在 `input` 文件夹中

### 镜像使用参数说明

`LSP_QuickRun` 镜像支持两种运行模式：用于 **生成视频** 的 `--lspmodel` 模式和用于 **评估模型** 的 `--eval` 模式。

#### 1. `--lspmodel` 模式

在该模式下，您需要指定以下参数：

`--id` :  预训练模型的名称，例如 `May` 、 `Obama1` 、 `Obama2` 等；

`--device` : 所使用的设备类型，例如 `cuda` 、 `cpu` 等；

`--driving_audio` : 输入音频文件的路径（Docker 容器内的路径）。

生成的视频文件将保存到容器内的 `/workspace/results` 目录中。

#### 2. `--eval` 模式

在此模式下，您需要指定以下参数：

`--gt_video` :  参考视频的路径（Docker 容器内的路径）；

`--gen_video` : 模型生成的视频路径（Docker 容器内的路径）。

评估结果将在命令行中显示。

### Docker 运行命令示例

#### 1. `--lspmodel` 模式

运行命令的模板如下所示：

```dockerfile
docker run -it --gpus all --rm --shm-size=8g \
-v <本地的model文件夹目录>:/workspace/data \
-v <本地的input文件夹目录>:/workspace/input \
-v <本地的result文件夹目录>:/workspace/results \
<镜像名称> \
--lspmodel \
--id <预训练模型ID> \
--device <设备名称> \
--driving_audio <容器内输入音频的路径>
```

例如：

```dockerfile
docker run -it --gpus all --rm --shm-size=8g \
-v E:\Code\Docker\models:/workspace/data \
-v E:\Code\Docker\input:/workspace/input \
-v E:\Code\Docker\results:/workspace/results \
lsp_quickrun:1.3 \
--lspmodel \
--id Obama1 \
--device cuda \
--driving_audio /workspace/input/00083.wav
```

#### 2. `--eval` 模式

运行命令的模板如下所示：

```dockerfile
docker run -it --gpus all --rm --shm-size=8g \
-v <本地的input文件夹目录>:/workspace/input \
-v <本地的result文件夹目录>:/workspace/results \
<镜像名称> \
--eval \
--gt_video <容器内参考视频的路径> \
--gen_video <容器内生成视频的路径>
```

例如：

```dockerfile
docker run -it --gpus all --rm --shm-size=8g \
-v E:\Code\Docker\input:/workspace/input \
-v E:\Code\Docker\results:/workspace/results \
lsp_quickrun:1.3 \
--eval \
--gt_video /workspace/input/May_short.mp4 \
--gen_video /workspace/results/May/May_short/May_short.avi
```