# 项目镜像配置文档
_由于项目镜像过大（约30G），镜像传输麻烦，所以我们没有在过多的机器上测试，目前我们已经在组员电脑（NVIDIA GeForce RTX 4060 Ti）wsl2下的子系统Ubuntu24.04、Ubuntu22.04.5中进行了测试，测试均成功运行_
## 一、硬件要求
### 显卡
只测试过NVIDIA GeForce RTX 4060 Ti
## 二、软件要求
### 1、操作系统
经测试，Ubuntu24.04、Ubuntu22.04.5均符合要求
### 2、显卡驱动(重点)
在创建镜像的过程中，发现该镜像对显卡驱动的要求很苛刻，不能使用太新的镜像。下面是经过测试可以运行的两个驱动版本
+  552.44（这个版本我们是在windows平台收动下载安装）
+  550.120 Linux系统下使用`apt install nvidia-utils-550 `安装
### 3、docker安装
我们安装了最新版本的docker，具体安装流程可参考这篇博客。[dockeran安装教程](https://blog.csdn.net/u011278722/article/details/137673353)下面给出基本步骤：

#安装前先卸载操作系统默认安装的docker，
`sudo apt-get remove docker docker-engine docker.io containerd runc`

#安装必要支持
`sudo apt install apt-transport-https ca-certificates curl software-properties-common gnupg lsb-release`


#添加 Docker 官方 GPG key （可能国内现在访问会存在问题）
`curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg`

#阿里源（推荐使用阿里的gpg KEY）
`curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg`

#添加 apt 源:
#Docker官方源
`echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null`


#阿里apt源
`echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null`


#更新源
`sudo apt update`
`sudo apt-get update`

#安装最新版本的Docker
`sudo apt install docker-ce docker-ce-cli containerd.io`
#等待安装完成

#查看Docker版本
`sudo docker version`

#查看Docker运行状态
`sudo systemctl status docker`

### 4、nvidia-docker2安装
**注意**：进行这一步之前注意给docker换源,不然容易出现错误

配置软件源
`distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list`
启动下载
`sudo apt-get update`
`sudo apt-get install -y nvidia-docker2`
**注意**：执行这步的时候会跳出一行询问你是否写入damon.json，记得输出“N”，输入“Y”会覆盖你的damon文件.

重启docker
`sudo systemctl restart docker`
## 三、镜像使用
### 1、获取镜像
+ 从我们的tar包获取镜像
  `docker load --input /path/to/hallo_image.tar`注意修改文件路径。
+ 使用dockerfile构建
  1、使用`git@github.com:fudan-generative-vision/hallo.git`下载项目，在将dockerfile放到项目根目录下
  2、运行`docker build -t hallo_image:2.0 .`构建镜像(可根据个人喜好，修改镜像名和tag，这里使用我们镜像完善的最后一版，如修改了镜像名和路径，注意对后续的命令也进行修改)

### 2、新建用于挂载数据的文件夹
`mkdir -p data/output data/images data/audios`
在root路径下新建data文件夹用于存放输入的图片（jpg），音频（wav）和输出的文件
### 3、启动镜像
#### 进入控制台
`docker run --rm --runtime=nvidia --gpus all -it \
  -v /root/data/output:/app/.cache \
  -v /root/data/images:/app/examples/reference_images \
  -v /root/data/audios:/app/examples/driving_audios \
  hallo_image:2.0 \
  bash`
+ --rm：
这个选项表示当容器停止时，自动删除容器。
+ --runtime=nvidia：
这个选项告诉 Docker 使用 nvidia 运行时来支持 GPU。如果后期这方面报错，检查/etc/docker/damon.json文件中是否写入环境变量
+ --gpus all：
这个选项告诉 Docker 容器使用所有可用的 GPU
+ -v /root/data/output:/app/.cache：
这个选项将宿主机的 /root/data/output 目录挂载到容器内的 /app/.cache 目录。
+ -v /root/data/images:/app/examples/reference_images：
这个选项将宿主机的 /root/data/images 目录挂载到容器内的 /app/examples/reference_images 目录。这样宿主机的图像文件可以供容器内的程序访问。
+ -v /root/data/audios:/app/examples/driving_audios：
这个选项将宿主机的 /root/data/audios 目录挂载到容器内的 /app/examples/driving_audios 目录。宿主机上的音频文件可以供容器内的程序使用。
+ hallo_image:2.0：
这是你要启动的 Docker 镜像的名称和标签。
+ bash：
启动容器后会进入 bash 控制台
#### 激活虚拟环境
`conda activate hallo`
#### 使用命令
进入控制台后即可根据你的图片和音频文件输出MP4，下面为使用data/images下的1.jpg和data/audios下的1.wav文件生成MP4的示例命令
`python scripts/inference.py --source_image examples/reference_images/1.jpg --driving_audio examples/driving_audios/1.wav`
然后你就可以在宿主机的/root/data/output下读取生成的视频文件