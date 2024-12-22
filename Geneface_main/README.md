# GeneFace-Reproduction

这是2024年语音识别课程大作业的仓库，用于[GeneFace](https://github.com/yerfor/GeneFace)的复现

## 环境

本仓库在Ubuntu20.04中实现，且需要自行安装CUDA11.3环境。Github无法上传conda环境，请参考<https://github.com/yerfor/GeneFace/blob/main/docs/prepare_env/install_guide-zh.md>配置环境

``` sh
conda activate ./conda
```

除此之外，需要通过apt安装以下包

``` sh
apt-get install libasound2-dev portaudio19-dev # dependency for pyaudio

```

激活环境后，需要运行以下命令，从torch-ngp构建CUDA插件

``` sh
bash docs/prepare_env/install_ext.sh
```

> 注：可能需要修改本地命令行配置以使用正确的本地conda环境
