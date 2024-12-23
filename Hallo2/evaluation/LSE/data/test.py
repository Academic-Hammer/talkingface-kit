import os
from moviepy.editor import VideoFileClip

def split_video(input_folder, output_folder, segment_duration=30):
    # 获取 /data/ 文件夹下的所有视频文件
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    # 如果 /merge/ 文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_file in video_files:
        print(video_file)
        # 构建输入视频文件的完整路径
        video_path = os.path.join(input_folder, video_file)
        
        # 使用 VideoFileClip 读取视频文件
        with VideoFileClip(video_path) as video:
            # 获取视频的总时长（秒）
            total_duration = video.duration
            
            # 计算分割的视频段数
            num_segments = int(total_duration // segment_duration) + 1
            
            # 对视频进行切割
            for i in range(num_segments):
                # 计算每个片段的起始时间和结束时间
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, total_duration)
                
                # 从视频中切割出指定时间段
                clip = video.subclip(start_time, end_time)
                
                # 构建输出视频文件的路径
                output_filename = f"{os.path.splitext(video_file)[0]}_segment_{i + 1}.mp4"
                output_path = os.path.join(output_folder, output_filename)
                
                # 保存切割出的片段
                clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
                
                print(f"Saved {output_filename} to {output_folder}")

if __name__ == "__main__":
    input_folder = "./LSE/data/input"  # 输入视频文件夹路径
    output_folder = "./LSE/data/merge"  # 输出切割后视频的文件夹路径
    
    split_video(input_folder, output_folder)
