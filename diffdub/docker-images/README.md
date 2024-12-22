# 配置文档

本文档旨在说明从.tar文件加载DiffDub项目镜像和评价指标镜像，并通过运行容器得到相关结果的步骤。

## 前提条件

1. **安装 Docker**：确保已经安装了 Docker。可通过以下命令验证是否安装：

   ```bash
   docker --version
   ```

   如果未安装，可以参考 [Docker 官方文档](https://docs.docker.com/get-docker/) 进行安装。

2. **准备好 `.tar` 文件**：*请从网盘下载 Docker 镜像的 `.tar` 文件。*

   ***我们镜像的网盘地址是*** 链接：https://pan.baidu.com/s/1YfrQ9PKGab6shMkAnmogCg?pwd=1234 提取码：1234

3. **准备好数据集文件**：我们的项目除了.mp4文件之外，还需要对应的.npy音频特征文件和.wav音频文件。

   *我们将八个测试视频的完整.mp4文件 .npy文件和.wav文件，以及一个用于检测项目是否能运行的十秒钟视频和对应的音频与特征文件打包放在了以下地址:*

   通过网盘分享的文件：test_datasets
   链接: https://pan.baidu.com/s/1_5K6STw6b9KXly4vbhdXZw 提取码: 1234

   test_datasets包含四个文件夹，分别是/show,/npys,/shots,/wavs 。其中show中是测试项目是否能正常运行所用的短视频的.mp4 .wav和.npy文件，其余三个文件夹则分别包含八个测试视频的.npy文件，.mp4文件和.wav文件。

------



## 步骤一：加载 `.tar` 文件到本地 Docker

1. 打开终端或命令行工具。

2. 使用以下命令将 `.tar` 文件加载到 Docker：

   ***对于DiffDub项目，使用***

   ```bash
   docker load -i diffdub.tar
   ```

   ***对于评价指标Evaluate，使用***

   ```bash
   docker load -i evaluate.tar
   ```

   **输出**：

   - 如果加载成功，你会看到类似如下的提示：

     ```
     Loaded image: hjchcjcjh/diffdub-6:latest
     ```

   这表示镜像已成功加载到本地。

   

3. 确认镜像是否已加载：

   ```bash
   docker images
   ```

   **输出**： 你会看到镜像的 `REPOSITORY` 和 `TAG`，例如：

   ```
   REPOSITORY               TAG       IMAGE ID       CREATED        SIZE
   hjchcjcjh/evaluate-4     latest    e2bd3261798a   10 hours ago   9.83GB
   hjchcjcjh/diffdub-6      latest    32d94662247f   10 hours ago   10.3GB
   ```

------



## 步骤二：运行 Docker 容器

1. 使用以下命令启动一个基于镜像的容器：

   ***对于DiffDub项目镜像，使用***

   ```bash
   docker run -it --gpus all --rm -v "<本机目录>:<容器挂载目录>" hjchcjcjh/diffdub-6:latest \
   --oneshot --video_inference \
   --mp4_original_path "<视频文件在容器中的路径>" \
    --hubert_feat_path "<音频特征文件在容器中的路径>" \
    --wav_path "<音频文件在容器中的路径>" \
    --saved_path "<输出文件夹在容器中的路径>" \
    --saved_name "<输出文件名>"
   ```

   - `-it`: 使用交互模式运行容器。
   - `--gpus all`: 使用所有 GPU 运行容器。
   - `--rm`: 容器停止后自动删除。
    - `-v "<本机目录>:<容器挂载目录>"`: 将本地目录挂载到容器中, 以便容器可以访问本地文件, 并将结果输出到本地目录中。例如，`-v "/home/user/data:/data"` 将本地 `/home/user/data` 目录挂载到容器的 `/data` 目录。注意: 容器的工作目录是`/app`
    - `saved_name`: 输出文件的名称, 默认为predition.mp4
    - `saved_path`: 输出文件的路径, 请设置为挂载目录中的路径, 否则输出文件会因为容器删除而丢失

   例如:
   ```bash
   docker run -it --gpus all --rm -v /root/show:/app/show  hjchcjcjh/diffdub-6 \
   --one_shot --video_inference \
    --saved_path show \
    --hubert_feat_path show/May_10s.npy \
    --wav_path show/May_10s.wav \
    --mp4_original_path show/May_10s.mp4 \
    --saved_name test.mp4
    ```

   ***对于评价指标镜像，使用***

   ```bash
   docker run -it --gpus all --rm -v "<本机目录>:<容器挂载目录>" hjchcjcjh/evaluate-4:latest \
   --original_video "<原始视频文件在容器中的路径>" \
    --generated_video "<生成视频文件在容器中的路径>" \
   --output_file "<输出文件在容器中的路径>"
   ```
   - `-it`: 使用交互模式运行容器。
   - `--gpus all`: 使用所有 GPU 运行容器。
   - `--rm`: 容器停止后自动删除。
    - `-v "<本机目录>:<容器挂载目录>"`: 将本地目录挂载到容器中, 以便容器可以访问本地文件, 并将结果输出到本地目录中。例如，`-v "/home/user/data:/data"` 将本地 `/home/user/data` 目录挂载到容器的 `/data` 目录。注意: 容器的工作目录是`/app`
    - `output_file`: 输出文件的路径, 如果不填写会输出到控制台, 填写任意非空路径会尝试输出到对应文件中。如果该文件路径不在挂载目录中或者由于某种原因无法创建，输出会因为容器删除而丢失
   
   例如:
   ```bash
   docker run -it --gpus all --rm -v /root/show:/app/show hjchcjcjh/evaluate-4 \
    --original_video show/May_10s.mp4 \
    --generated_video show/test.mp4 \
    --output_file show/result.txt
   ```

------

## 步骤三：查看结果

根据传递参数中设置的输出目录检查结果即可。其中DiffDub镜像应该会得到一个生成的.mp4文件，而评价指标应该会得到一个包含所有指标的.txt文件。

# 可能的问题和建议

我们推荐在预先配置了cuda的Linux环境下完成运行。我们在Windows电脑上运行镜像时发现其可能出现以下问题:

1.gpu工作不稳定

​	1.1 在运行DiffDub项目或者评价指标项目时，使用cpu可以得到正确的结果，但是使用gpu会提示类似```cv2.error: OpenCV(4.10.0) /io/opencv/modules/imgproc/src/resize.cpp:4152: error: (-215:Assertion failed) !ssize.empty() in function 'resize'```的报错

​	1.2 在运行DiffDub项目或者评价指标项目时，有时会出现```Segmentation fault (core dumped)```

2.无法运行项目

​	2.1 在一个显卡驱动版本较新的电脑上会直接无法运行

我们初步怀疑以上问题是由名为torchlm的库导致的,我们还找到了相关的issue:https://github.com/DefTruth/torchlm/issues/58

后来我们怀疑这个问题和cuda在windows与linux上的底层实现不同有关。因为我们发现通过docker-compose重新构建镜像有可能解决这个问题:https://github.com/pytorch/pytorch/issues/133692