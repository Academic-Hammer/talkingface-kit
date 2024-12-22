### syncnet_python

|||
|:--:|:--:|
| **[论文网址](https://link.springer.com/chapter/10.1007/978-3-319-54427-4_19)** | **[GitHub](https://github.com/joonson/syncnet_python)** |


### 使用 Docker 镜像运行

#### 构筑镜像

好像每次构筑都会有重复的下载 ( 捂脸 ), 不过貌似是不占大小，是覆写已经存在的文件，主要构筑里面写了下载模型和 demo 视频的执行脚本，自己下的话可以删掉对应部分

在项目目录下构筑哦！

```bash
    docker build -f .dockerfile -t syncnet:v1 .
```

#### 下载镜像

> 不建议，镜像文件很大，建议本地构筑

好像我构筑的有亿点点大，虽然是两位数，但是单位是 GB hh ( 上传好浪费时间，我的建议是自己构筑，如果一定需要可以通过 GitHub 或者邮箱: wincomso9@outlook.com 联系我 )

下载好后放到 `syncnet_python` 目录下，然后执行以下命令

> 可以 `docker images` 看加载是否成功

```bash
    docker load -i syncnet.tar
```

#### 运行镜像

> 默认执行的评估是 `eval/data/demo.mp4`, 不存在的话就修改`/syncnet_python/run_commands.sh` 为自定义的路径，

```bash
    # 本地待测评的数据放到 /syncnet_python/eval/data 下，修改 run_commands.sh 对应的路径
    docker run --rm --gpus all -v ${PWD}/eval/data:/syncnet_python/eval/data -v ${PWD}/tmp:/syncnet_python/tmp -v ${PWD}/all_scores.txt:/syncnet_python/all_scores.txt syncnet:v1
```

### 本机 WSL 下配置 ( 和在Ubuntu 下配置一样 )

```bash
    git clone https://github.com/joonson/syncnet_python.git 
    cd syncnet_python
    conda create -n SP python=3.9
    conda activate SP
    pip install -r requirements.txt
    sh download_model.sh
```

#### 运行测试用例

```bash
    python demo_syncnet.py --videofile data/example.avi --tmp_dir /path/to/temp/directory
```

### 进行评估

使用 [Wav2Lip](https://github.com/Rudrabha/Wav2Lip/tree/master/evaluation) 的评估脚本，按照说明将 `Wav2Lip/evaluation/scores_LSE/` 下的脚本放入 `/syncnet_python/` 目录下 ( 直接下载传入最好，懒得再克隆一个项目 )

本机跑建议运行下面的命令逐一推理:

```bash
    python run_pipeline.py --videofile eval/Jae-in.mp4 --reference t1 --data_dir tmp
    python calculate_scores_real_videos.py --videofile eval/Jae-in.mp4 --reference t1 --data_dir tmp >> all_scores.txt
```

参数说明：

`--videofile`: 视频文件路径

`--reference`: 类似于视频标号, 其实完全可以用视频名字不需要单独传参。。懒得改了

`--data_dir`: 输出的临时存放目录（在该文件夹下的子目录下存放生成的帧），同时作为第二条命令的输入


### 问题与解决方案:

+ `numpy` 没有 `int` 属性，较新的版本是 `int_`

    ```bash
        pip install numpy==1.22
    ```

    或者找到`\syncnet_python\detectors\s3fd\box_utils.py` 38 行， 修改为 `return np.array(keep).astype(np.int_)`

+ `scenedetect` 包的较老版本和新版 `python` 存在兼容性问题 ( 具体来讲就是 `python` 里面约束规范加强了, `scenedetect` 之前的版本存在新版本不允许的操作 ( 对于元组 ) )

    ```bash
        pip install av 
        pip install scenedetect==0.6.0
    ```

### 定量评估结果 ( LSE-D & LSE-C )

在项目目录 `syncnet_python` 下找到 `all.txt` 即可, 输出的结果为 `Min dist & Confidence` 也就是 `LSE-D & LSE-C`, 顺序和执行评估的顺序相同, 后评估的在末尾添加

```
    评估视频        LSE-D          LSE-C(论文里的SyncNet 置信度得分(Sync_conf))
```
---
```
    Jae-in        9.679484       3.81083

    Lieu          8.466651       6.7788954

    Macron        9.462689       4.1416864

    May           8.08356        6.0931616

    Obama         8.022001       6.5014324

    Obama1        8.213086       6.4382563

    Obama2        7.1203218      6.8484406

    Shaheen       7.524077       7.74151
```


### 定量评估结果 ( PSNR & SSIM )

在 /eval/ 中进行评估，环境比较简单就直接放在外面了

使用 `eval.py`: 进行视频定量评估 `PSNR & SSIM`

```
    评估视频        PSNR          SSIM
```
---
```
    Jae-in        19.6442        0.7636

    Lieu          25.0265        0.8276

    Macron        24.6637        0.8391

    May           25.0020        0.7674

    Obama         25.1191        0.8459

    Obama1        22.6824        0.7921

    Obama2        22.4471        0.8374

    Shaheen       26.6465        0.8514
```

原论文没有 PSNR 评估，SSIM 评估的分数为 0.86/0.85/0.69 ( MEAD / HDTF / Voxceleb2 )