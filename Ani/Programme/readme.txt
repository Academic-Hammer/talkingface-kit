项目代码
在python3.10环境下可用
下载asset及pretrained_model文件夹后
运行命令pip install -r requirements.txt，之后就可以使用各种功能
在线示例: 
    python -m scripts.app
pose转video: 
    python -m scripts.pose2vid --config ./configs/prompts/animation.yaml -W 512 -H 512 -acc
video转pose: 
    python -m scripts.vid2pose --video_path pose_video_path.mp4
video直接生成目标video: 
    python -m scripts.vid2vid --config ./configs/prompts/animation_facereenac.yaml -W 512 -H 512 -acc
语音生成目标video: 
    python -m scripts.audio2vid --config ./configs/prompts/animation_audio.yaml -W 512 -H 512 -acc
生成引用pose: 
    python -m scripts.generate_ref_pose --ref_video ./configs/inference/head_pose_temp/pose_ref_video.mp4 --save_path ./configs/inference/head_pose_temp/pose.npy
训练：
    python -m scripts.preprocess_dataset --input_dir VFHQ_PATH --output_dir SAVE_PATH --training_json JSON_PATH
并在yaml中加入: 
    data:
      json_path: JSON_PATH
阶段1: 
    accelerate launch train_stage_1.py --config ./configs/train/stage1.yaml
阶段2: 
    在stage2.yaml中加入阶段1训练的产物
    accelerate launch train_stage_2.py --config ./configs/train/stage2.yaml
