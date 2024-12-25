import os
import sys
import subprocess
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
import soundfile as sf
import librosa
from tqdm import tqdm  # 导入 tqdm

def extract_audio(input_video_file, output_audio_directory, sample_rate=16000):
    os.makedirs(output_audio_directory, exist_ok=True)
    input_filename = os.path.basename(input_video_file)
    output_audio_file = os.path.splitext(input_filename)[0] + ".wav"
    output_audio_file_path = os.path.join(output_audio_directory, output_audio_file)

    print(f"Extracting audio from video file: {input_video_file}")
    print(f"Saving audio to: {output_audio_file_path}")
    
    try:
        video = VideoFileClip(input_video_file)
        audio = video.audio
        audio = audio.set_fps(sample_rate)
        audio.write_audiofile(output_audio_file_path, codec='pcm_s16le')
        video.close()
        print(f"Audio saved: {output_audio_file_path}")
        return output_audio_file_path
    except Exception as e:
        print(f"Error: {e}")
        return None

def split_audio(audio_path, segment_duration=60):
    print(f"Splitting audio file: {audio_path}")
    speech_array, sr = librosa.load(audio_path, sr=None)
    segments = []
    for start in range(0, len(speech_array), segment_duration * sr):
        segment = speech_array[start:start + segment_duration * sr]
        if len(segment) > 0:
            segments.append(segment)
    print(f"Audio split into {len(segments)} segments.")
    return segments, sr

def save_audio_segments(segments, sr, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    segment_files = []
    for i, segment in enumerate(segments):
        segment_file = os.path.join(output_dir, f"segment_{i}.wav")
        print(f"Saving segment {i} to: {segment_file}")
        sf.write(segment_file, segment, sr)  # 使用 soundfile 保存音频
        segment_files.append(segment_file)
    return segment_files

def run_emotalk(audio_path):
    absolute_audio_path = os.path.abspath(audio_path)  # 获取音频文件的绝对路径
    print(f"Processing audio files: {absolute_audio_path}")
    try:
        subprocess.run(["python", "demo.py", "--wav_path", absolute_audio_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running demo.py: {e}")

def combine_videos(segment_file_paths, output_directory):
    video_clips = []
    print(f"Combining video clips into final video in: {output_directory}")
    
    for segment_file in segment_file_paths:
        video_path = os.path.splitext(segment_file)[0] + '.mp4'  # 生成相应的mp4文件名
        print(f"Checking if video exists at: {video_path}")
        if os.path.exists(video_path):
            video_clips.append(VideoFileClip(video_path))
        else:
            print(f"Video file not found: {video_path}")

    if video_clips:
        final_video = concatenate_videoclips(video_clips)
        final_video_path = os.path.join(output_directory, "final_output.mp4")
        print(f"Final video will be saved at: {final_video_path}")
        try:
            final_video.write_videofile(final_video_path, codec='libx264', audio_codec='aac')
            print(f"Video saved successfully: {final_video_path}")
        except Exception as e:
            print(f"Error during video saving: {e}")
        return final_video_path
    return None

def clean_up_files(segment_files, output_directory):
    print(f"Cleaning up files in directory: {output_directory}")
    for segment_file in segment_files:
        if os.path.exists(segment_file):
            os.remove(segment_file)
            print(f"Deleted audio files: {segment_file}")

    # 删除生成的分段视频文件（如果有的话）
    video_files = [os.path.splitext(segment_file)[0] + '.mp4' for segment_file in segment_files]
    for video_file in video_files:
        if os.path.exists(video_file):
            os.remove(video_file)
            print(f"Deleted video files: {video_file}")

# 计算评分
def calculate_scores(result_video_path, syncnet_dir=".\syncnet_python"):
    print(f"Calculating scores for video: {result_video_path}")
    try:
        # 切换到 syncnet_python 文件夹并执行命令
        os.chdir(syncnet_dir)

        # 执行视频评分命令，计算 LSE-D 和 LSE-C
        video_file = "../result/final_output.mp4"
        command_run_pipeline = [
            "python", "run_pipeline.py", "--videofile", video_file, "--reference", "wav2lip", "--data_dir", "tmp_dir"
        ]
        subprocess.run(command_run_pipeline, check=True)

        command_calculate_scores = [
            "python", "calculate_scores_real_videos.py", "--videofile", video_file, "--reference", "wav2lip", "--data_dir", "tmp_dir"
        ]
        result = subprocess.run(command_calculate_scores, capture_output=True, text=True, check=True)

        # 输出评分到 all_scores.txt 文件
        scores_output = result.stdout
        with open("all_scores.txt", "w") as f:
            f.write(scores_output)

        lse_c = None
        lse_d = None

        # 从输出中提取 LSE-D 和 LSE-C
        scores = scores_output.split()
        if len(scores) >= 2:
            lse_d = scores[0].strip()
            lse_c = scores[1].strip()
        else:
            raise ValueError("Unexpected output format: not enough scores.")

        print(f"LSE-D: {lse_d}")
        print(f"LSE-C: {lse_c}")

        return lse_d, lse_c
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while calculating scores: {e}")
        return None, None

def main():
    # 打印当前工作目录
    current_directory = os.path.abspath(os.getcwd())
    print(f"Current working directory: {current_directory}")
    
    if len(sys.argv) != 2:
        print("usage: python EmoTalk.py <video path>")
        return

    input_video_file = sys.argv[1]
    print(f"Input video file: {input_video_file}")
    
    output_audio_directory = os.path.join(current_directory, "audio")
    print(f"Audio will be saved in: {output_audio_directory}")
    
    # 提取音频
    audio_path = extract_audio(input_video_file, output_audio_directory)
    
    if audio_path:
        # 切分音频
        segments, sr = split_audio(audio_path, segment_duration=60)  # 每60s切分一次
        
        # 保存切分的音频片段
        segment_files = save_audio_segments(segments, sr, output_audio_directory)

        # 处理每个音频片段并显示进度条
        for segment_file in tqdm(segment_files, desc="Processing audio file"):
            run_emotalk(segment_file)

        # 合并处理后的视频
        combine_videos(segment_files, './result')
        
        result_video_path = combine_videos(segment_files, './result')

        # 计算并显示 LSE-D 和 LSE-C
        if result_video_path:
            lse_d, lse_c = calculate_scores(result_video_path)
            if lse_d and lse_c:
                print(f"Final Scores - LSE-D: {lse_d}, LSE-C: {lse_c}")

        # 清理临时文件
        clean_up_files(segment_files, output_audio_directory)

if __name__ == "__main__":
    main()

