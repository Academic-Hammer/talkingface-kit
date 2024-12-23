# Hallo2配置

本文件夹为Hallo2项目源文件夹，需要下载预训练模型参数(https://huggingface.co/fudan-generative-ai/hallo2)，验收现场拷贝给助教。放在该文件夹下，结构为：

```text
./pretrained_models/
|-- audio_separator/
|   |-- download_checks.json
|   |-- mdx_model_data.json
|   |-- vr_model_data.json
|   `-- Kim_Vocal_2.onnx
|-- CodeFormer/
|   |-- codeformer.pth
|   `-- vqgan_code1024.pth
|-- face_analysis/
|   `-- models/
|       |-- face_landmarker_v2_with_blendshapes.task  # face landmarker model from mediapipe
|       |-- 1k3d68.onnx
|       |-- 2d106det.onnx
|       |-- genderage.onnx
|       |-- glintr100.onnx
|       `-- scrfd_10g_bnkps.onnx
|-- facelib
|   |-- detection_mobilenet0.25_Final.pth
|   |-- detection_Resnet50_Final.pth
|   |-- parsing_parsenet.pth
|   |-- yolov5l-face.pth
|   `-- yolov5n-face.pth
|-- hallo2
|   |-- net_g.pth
|   `-- net.pth
|-- motion_module/
|   `-- mm_sd_v15_v2.ckpt
|-- realesrgan
|   `-- RealESRGAN_x2plus.pth
|-- sd-vae-ft-mse/
|   |-- config.json
|   `-- diffusion_pytorch_model.safetensors
|-- stable-diffusion-v1-5/
|   `-- unet/
|       |-- config.json
|       `-- diffusion_pytorch_model.safetensors
`-- wav2vec/
    `-- wav2vec2-base-960h/
        |-- config.json
        |-- feature_extractor_config.json
        |-- model.safetensors
        |-- preprocessor_config.json
        |-- special_tokens_map.json
        |-- tokenizer_config.json
        `-- vocab.json
```

所需要环境为Ubuntu22.04/20.04、Cuda11.8

通过conda创建hallo项目环境：`conda create -n hallo python=3.10`，并且按照相关库：
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

此外，还需要按照ffmpeg：`apt-get ffmpeg`

该项目共有两个部分，第一部分为长时间动画，第二部分为高分辨率动画：

- 长时间动画

只需要运行`scripts/inference_long.py`，指令为：

```bash
python scripts/inference_long.py --config ./configs/inference/long.yaml
```

其中，`scripts/inference_long.py`有关更多选项：

```bash
usage: inference_long.py [-h] [-c CONFIG] [--source_image SOURCE_IMAGE] [--driving_audio DRIVING_AUDIO] [--pose_weight POSE_WEIGHT]
                    [--face_weight FACE_WEIGHT] [--lip_weight LIP_WEIGHT] [--face_expand_ratio FACE_EXPAND_RATIO]

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
  --source_image SOURCE_IMAGE
                        source image
  --driving_audio DRIVING_AUDIO
                        driving audio
  --pose_weight POSE_WEIGHT
                        weight of pose
  --face_weight FACE_WEIGHT
                        weight of face
  --lip_weight LIP_WEIGHT
                        weight of lip
  --face_expand_ratio FACE_EXPAND_RATIO
                        face region
```

- 高分辨率动画

只需要运行`scripts/video_sr.py`，指令为：

```bash
python scripts/video_sr.py --input_path [input_video] --output_path [output_dir] --bg_upsampler realesrgan --face_upsample -w 1 -s 4
```

其中，`scripts/video_sr.py`有关更多选项：

```bash
usage: video_sr.py [-h] [-i INPUT_PATH] [-o OUTPUT_PATH] [-w FIDELITY_WEIGHT] [-s UPSCALE] [--has_aligned] [--only_center_face] [--draw_box]
                   [--detection_model DETECTION_MODEL] [--bg_upsampler BG_UPSAMPLER] [--face_upsample] [--bg_tile BG_TILE] [--suffix SUFFIX]

options:
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input_path INPUT_PATH
                        Input video
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Output folder.
  -w FIDELITY_WEIGHT, --fidelity_weight FIDELITY_WEIGHT
                        Balance the quality and fidelity. Default: 0.5
  -s UPSCALE, --upscale UPSCALE
                        The final upsampling scale of the image. Default: 2
  --has_aligned         Input are cropped and aligned faces. Default: False
  --only_center_face    Only restore the center face. Default: False
  --draw_box            Draw the bounding box for the detected faces. Default: False
  --detection_model DETECTION_MODEL
                        Face detector. Optional: retinaface_resnet50, retinaface_mobile0.25, YOLOv5l, YOLOv5n. Default: retinaface_resnet50
  --bg_upsampler BG_UPSAMPLER
                        Background upsampler. Optional: realesrgan
  --face_upsample       Face upsampler after enhancement. Default: False
  --bg_tile BG_TILE     Tile size for background sampler. Default: 400
  --suffix SUFFIX       Suffix of the restored faces. Default: None
```
