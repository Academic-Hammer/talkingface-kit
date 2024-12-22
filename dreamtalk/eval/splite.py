import ffmpeg
import os

def split_video(input_path, output_path1, output_path2):
    # 确保输入路径有效
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在：{input_path}")
    
    try:
        # 获取视频时长
        probe = ffmpeg.probe(input_path)
        duration = float(probe['format']['duration'])  # 视频总时长
        mid_point = duration / 2                      # 中间时间点

        ffmpeg.input(input_path, ss=0, t=mid_point).output(output_path1, c='copy').run(overwrite_output=True)

        ffmpeg.input(input_path, ss=mid_point).output(output_path2, c='copy').run(overwrite_output=True)

        print(f"视频已成功分割：\n1. {output_path1}\n2. {output_path2}")
    except ffmpeg.Error as e:
        print("FFmpeg 错误:", e.stderr.decode('utf-8'))
        raise

# 使用示例
if __name__ == "__main__":
    input_video = "videos/Macron.mp4"     # 输入视频路径
    output_video1 = "videos/Macron1.mp4"  # 输出第一部分路径
    output_video2 = "videos/Macron2.mp4"  # 输出第二部分路径
    
    split_video(input_video, output_video1, output_video2)



