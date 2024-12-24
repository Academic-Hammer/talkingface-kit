import subprocess
import os


def call_demo(input_image, input_audio):
    # 获取当前的 Conda 环境 Python 路径
    conda_python_path = "/root/anaconda3/envs/muse/bin/python"  # 替换为你 Conda 环境中 Python 的路径

    # 构建命令
    command = [
        conda_python_path, '/home/syy/MuseTalk/demo.py',
        '--input_image', input_image,
        '--input_audio', input_audio
    ]
    
    # 使用 subprocess 执行命令
    subprocess.run(command)
import os
from moviepy.editor import VideoFileClip

def extract_audio_from_video(input_directory, output_directory):
    # 遍历文件夹中的所有.mp4文件
    for filename in os.listdir(input_directory):
        if filename.endswith('.mp4'):
            video_path = os.path.join(input_directory, filename)
            audio_path = os.path.join(output_directory, filename.replace('.mp4', '.wav'))
            
            try:
                # 读取视频文件并提取音频
                video_clip = VideoFileClip(video_path)
                audio_clip = video_clip.audio
                
                # 保存音频到指定输出路径
                audio_clip.write_audiofile(audio_path)

                # 关闭视频和音频文件
                audio_clip.close()
                video_clip.close()

                print(f"音频已提取并保存为: {audio_path}")

                print("现在开始评测")
                call_demo(video_path, audio_path)
                print("评测完成")
            except Exception as e:
                print(f"处理视频 {filename} 时发生错误: {e}")

# 示例调用
input_directory = '/home/geneface_datasets-20241217T113435Z-001/geneface_datasets/data/raw/videos'  # 替换为视频文件夹路径
output_directory = '/home/geneface_datasets-20241217T113435Z-001/geneface_datasets/data/raw/videos'  # 替换为输出音频文件夹路径

extract_audio_from_video(input_directory, output_directory)

