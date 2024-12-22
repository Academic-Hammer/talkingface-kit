# README

## 概述

该项目演示了如何使用Qwen-2-VL模型评估视频的质量，Qwen-2-VL是一个视觉-语言模型，能够处理视频和文本输入。通过输入一个视频文件，模型将生成一份基于以下标准的评估报告：嘴唇运动自然度、视觉质量、时序一致性、真实性和身份保留，并为每个标准提供评分和简要解释。

## 环境要求

在运行该代码之前，需要安装以下核心依赖库：

- **vllm**：0.6.4
- **qwen-vl-utils**：0.0.8
- **transformers**：4.46.3

此外，还需要其他一些常用库，如`json`、`argparse`等。这些库通常已经包含在Python的标准库中，或者可以通过`pip`安装。

## 运行代码
运行该脚本时，需要传入两个参数：模型路径（--model_path）和视频路径（--video_path）。具体命令如下：

**python evaluate_video.py --model_path <模型路径> --video_path <视频路径>**

**--model_path**：指定预训练的Qwen-2-VL模型的路径。可从https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/tree/main处下载。
**--video_path**：指定待评估的视频文件路径。