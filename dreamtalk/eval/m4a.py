import os
import subprocess

def convert_mp4_to_m4a(input_folder, output_folder):
    """
    将指定文件夹中的所有 MP4 文件转换为 M4A 音频文件。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for file_name in os.listdir(input_folder):
        # 检查文件是否为 MP4 格式
        if file_name.lower().endswith(".mp4"):
            input_path = os.path.join(input_folder, file_name)
            output_name = os.path.splitext(file_name)[0] + ".m4a"
            output_path = os.path.join(output_folder, output_name)

            command = [
                "ffmpeg",
                "-i", input_path,        # 输入文件
                "-vn",                   # 禁用视频
                "-acodec", "aac",        # 指定音频编码器为 AAC
                "-b:a", "192k",          # 设置音频比特率
                output_path              # 输出文件
            ]

            # 执行 ffmpeg 命令
            try:
                subprocess.run(command, check=True)
                print(f"Converted: {input_path} -> {output_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error converting {input_path}: {e}")

    print("All files processed.")

input_folder = "videos"  # 输入文件夹，包含 MP4 文件
output_folder = "m4a"  # 输出文件夹，存放转换后的 M4A 文件

convert_mp4_to_m4a(input_folder, output_folder)
