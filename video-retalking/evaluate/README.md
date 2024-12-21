## 项目依赖安装
### 使用以下命令安装所需的Python包
conda create --name vrtest python=3.10
conda activate vrtest
pip install -r requirements.txt

## ssim.py 执行方式
### 使用以下命令运行 ssim.py 脚本，并传入原视频路径和推理视频路径
python ssim.py --origin /origin_path --reference /reference_path

--origin: 指定原视频路径
--reference: 指定推理视频路径

## psnr.py 执行方式
### 使用以下命令运行 psnr.py 脚本，并传入原视频和推理视频路径
python psnr.py --origin /origin_path --reference /reference_path

--origin: 指定原视频路径
--reference: 指定推理视频路径

## fid.py 执行方式
### 使用以下命令运行 fid.py 脚本，并传入原视频和推理视频路径
python fid.py --origin /origin_path --reference /reference_path

--origin: 指定原视频路径
--reference: 指定推理视频路径

注：FID的计算需要远程拉取预训练模型，需要科学上网

## cpbd.py 执行方式
### 使用以下命令运行 cpbd.py 脚本，并传入推理视频路径
python cpbd.py --reference /reference_path

--reference: 指定推理视频路径

## niqe.py 执行方式
### 使用以下命令运行 niqe.py 脚本，并传入推理视频文件夹的路径
python niqe.py  --reference /reference_path

--reference: 指定推理视频路径

注：NIQE的计算需要配合给定的data文件夹中的数据niqe_image_params.mat使用

## lse.py 执行方式
### 使用以下命令运行 lse.py 脚本，并传入推理视频文件夹的路径
python lse.py  --reference /reference_path

--reference: 指定推理视频路径

注：LSE-C和LSE-D评价指标是根据syncnet网络计算的，这里参考了[syncnet的源码](https://github.com/joonson/syncnet_python)