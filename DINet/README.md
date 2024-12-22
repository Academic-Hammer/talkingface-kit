# DINet: Deformation Inpainting Network for Realistic Face Visually Dubbing on High Resolution Video (AAAI2023)
![在这里插入图片描述](https://img-blog.csdnimg.cn/178c6b3ec0074af7a2dcc9ef26450e75.png)
[Paper](https://fuxivirtualhuman.github.io/pdf/AAAI2023_FaceDubbing.pdf) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     [demo video](https://www.youtube.com/watch?v=UU344T-9h7M&t=6s)  &nbsp;&nbsp;&nbsp;&nbsp; Supplementary materials


这是2024年语音识别课程大作业的仓库，用于[DINet](https://github.com/MRzzm/DINet)的复现
## 数据获取
##### 在 [Google drive](https://drive.google.com/drive/folders/1rPtOo9Uuhc59YfFVv4gBmkh0_oG0nCQb?usp=share_link)中下载资源 （asserts.zip）。解压缩并将 dir 放入 ./ 中
+  使用示例视频进行推理。运行 
  ```python 
python inference.py --mouth_region_size=256 --source_video_path=./asserts/examples/testxxx.mp4 --source_openface_landmark_path=./asserts/examples/testxxx.csv --driving_audio_path=./asserts/examples/driving_audio_xxx.wav --pretrained_clip_DINet_path=./asserts/clip_training_DINet_256mouth.pth  
```
结果保存在 ./asserts/inference_result

+  使用自定义视频进行推理。 
**Note:** 发布的预训练模型是在 HDTF 数据集上训练的。（视频名称在 ./asserts/training_video_name.txt 中）

使用 [openface](https://github.com/TadasBaltrusaitis/OpenFace)检测自定义视频的平滑面部特征点。


检测到的人脸特征点保存在 “xxxx.csv” 中。运行
  ```python 
python inference.py --mouth_region_size=256 --source_video_path= custom video path --source_openface_landmark_path=  detected landmark path --driving_audio_path= driving audio path --pretrained_clip_DINet_path=./asserts/clip_training_DINet_256mouth.pth  
```
在您的自定义视频上实现人脸视觉配音。
## 训练
### 数据处理

 1. 从[HDTF](https://github.com/MRzzm/HDTF)下载视频。根据 xx_annotion_time.txt 分割视频，不裁剪和调整视频大小。
 2. 将所有分割的视频重新采样为 25fps，并将视频放入 “./asserts/split_video_25fps”。您可以在 “./asserts/split_video_25fps” 中看到两个示例视频。我们使用[软件](http://www.pcfreetime.com/formatfactory/cn/index.html) 对视频进行重新采样。我们在实验中提供了训练视频的名称列表。（请参阅“./asserts/training_video_name.txt”）  
 3. 使用  [openface](https://github.com/TadasBaltrusaitis/OpenFace)  检测所有视频的平滑面部特征点。将所有 “.csv” 结果放入 “./asserts/split_video_25fps_landmark_openface” 中。您可以在 “./asserts/split_video_25fps_landmark_openface” 中看到两个示例 csv 文件。



 4. 从所有视频中提取帧并将帧保存在 “./asserts/split_video_25fps_frame” 中。运行
```python 
python data_processing.py --extract_video_frame
```
 5. 从所有视频中提取音频，并将音频保存在 ./asserts/split_video_25fps_audio 中。运行
 ```python 
python data_processing.py --extract_audio
```
 6. 从所有音频中提取 deepspeech 特征并将特征保存在 “./asserts/split_video_25fps_deepspeech” 中。运行
  ```python 
python data_processing.py --extract_deep_speech
```
 7. 裁剪所有视频的人脸并将图像保存在 “./asserts/split_video_25fps_crop_face” 中。运行
   ```python 
python data_processing.py --crop_face
```
 8. 生成训练 json 文件 “./asserts/training_json.json”。运行
   ```python 
python data_processing.py --generate_training_json
```

### 训练模型
训练过程分为帧训练阶段和 clip 训练阶段。在帧训练阶段，我们使用从粗到细的策略，因此您可以在任意分辨率下训练模型。

#### 框架训练阶段。
在帧训练阶段，我们只使用感知损失和 GAN 损失

 1. 首先，以 104x80（嘴部区域为 64x64）分辨率训练 DINet。运行
   ```python 
python train_DINet_frame.py --augment_num=32 --mouth_region_size=64 --batch_size=24 --result_path=./asserts/training_model_weight/frame_training_64
```


 2. 加载预训练模型（面部：104x80 & 嘴巴：64x64）并以更高分辨率训练DINet（面部：208x160 & 嘴巴：128x128）。运行
python train_DINet_frame.py --augment_num=100 --mouth_region_size=128 --batch_size=80 --coarse2fine --coarse_model_path=./asserts/training_model_weight/frame_training_64/xxxxxx.pth --result_path=./asserts/training_model_weight/frame_training_128
```


 3. 加载预训练模型（面部：208x160 & 嘴巴：128x128）并以更高分辨率训练DINet（面部：416x320 & 嘴巴：256x256）。运行
   ```python 
python train_DINet_frame.py --augment_num=20 --mouth_region_size=256 --batch_size=12 --coarse2fine --coarse_model_path=./asserts/training_model_weight/frame_training_128/xxxxxx.pth --result_path=./asserts/training_model_weight/frame_training_256
```


#### 剪辑训练阶段。
在剪辑训练阶段，我们使用感知损失、帧/剪辑 GAN 损失和同步损失。加载预训练的帧模型（面部：416x320 & 嘴巴：256x256），预训练的同步网络模型（嘴巴：256x256）并在剪辑设置中训练DINet。运行
   ```python 
python train_DINet_clip.py --augment_num=3 --mouth_region_size=256 --batch_size=3 --pretrained_syncnet_path=./asserts/syncnet_256mouth.pth --pretrained_frame_DINet_path=./asserts/training_model_weight/frame_training_256/xxxxx.pth --result_path=./asserts/training_model_weight/clip_training_256
```

## 声明
整体实现思路来自[https://github.com/MRzzm/DINet](https://github.com/MRzzm/DINet)