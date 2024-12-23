# 评估代码

该文件夹存放了评估代码，由evaluation.py进行评估指标的计算，其相关选项为：
```bash
usage: evaluation.py [-h] --original_video_path ORIGINAL_VIDEO_PATH --generated_video_path GENERATED_VIDEO_PATH --output_dir OUTPUT_DIR [--batch_size BATCH_SIZE] [--niqe NIQE]
                     [--ssim SSIM] [--psnr PSNR] [--fid FID] [--lse LSE]

Total

options:
  -h, --help            show this help message and exit
  --original_video_path ORIGINAL_VIDEO_PATH
                        original video path
  --generated_video_path GENERATED_VIDEO_PATH
                        generated video path
  --output_dir OUTPUT_DIR
                        output directory
  --batch_size BATCH_SIZE
                        batch size
  --niqe NIQE           whether to calculate NIQE
  --ssim SSIM           whether to calculate SSIM
  --psnr PSNR           whether to calculate PSNR
  --fid FID             whether to calculate FID
  --lse LSE             whether to calculate LSE-C & D
```

其中，original_video_path、generated_video_path和output_dir分别指定原视频地址、生成视频地址和输出文件夹，其次niqe、ssim、psnr、fid和lse（LSE-C&LSE-D）分别代表对应的指标是否需要进行计算，默认是不进行计算，需要直接指定其值为1，如：`--lse 1`。

默认输出文件夹路径为`./output`，在该文件夹下会有一个evaluation.txt文件，该文件记录之前评估结果，结构为：
```txt
input_dir:
    NIQE:...
    LSE-C:...
    ...
```