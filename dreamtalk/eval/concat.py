import ffmpeg
import os

def merge_videos(input1, input2, output):
    # 确保输入文件存在
    if not os.path.exists(input1):
        raise FileNotFoundError(f"输入文件不存在：{input1}")
    if not os.path.exists(input2):
        raise FileNotFoundError(f"输入文件不存在：{input2}")

    # 创建一个中间列表文件
    concat_file = "concat_list.txt"
    with open(concat_file, "w") as f:
        f.write(f"file '{os.path.abspath(input1)}'\n")
        f.write(f"file '{os.path.abspath(input2)}'\n")

    try:
        ffmpeg.input(concat_file, format='concat', safe=0).output(output, c='copy').run(overwrite_output=True)
        print(f"视频已成功合并为：{output}")
    except ffmpeg.Error as e:
        print("FFmpeg 错误:", e.stderr.decode('utf-8'))
        raise
    finally:
        # 清理中间文件
        if os.path.exists(concat_file):
            os.remove(concat_file)

if __name__ == "__main__":
    input_video1 = "eval/Macron1.mp4"  # 输入视频1
    input_video2 = "eval/Macron2.mp4"  # 输入视频2
    output_video = "eval/Macron.mp4"

    merge_videos(input_video1, input_video2, output_video)
