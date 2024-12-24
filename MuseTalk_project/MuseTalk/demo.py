import argparse
import subprocess
import re
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", default="data/video/yongen.mp4")
    parser.add_argument("--input_audio", default="data/audio/yongen.wav")
    parser.add_argument("--output_dir", default="./result")
    parser.add_argument("--ground_truth", default=None)
    args = parser.parse_args()

    ffmpeg_dir = '/workspace/ffmpeg-4.4-amd64-static'
    ffmpeg_executable = os.path.join(ffmpeg_dir, 'ffmpeg')
    
    # 添加 ffmpeg 目录到 PATH
    os.environ['PATH'] = ffmpeg_dir + ":" + os.environ.get('PATH', '')
    
    # 可选：验证 ffmpeg 是否可执行
    if not (os.path.isfile(ffmpeg_executable) and os.access(ffmpeg_executable, os.X_OK)):
        raise FileNotFoundError(f"ffmpeg not found or not executable at {ffmpeg_executable}")
    
    os.makedirs('evaluation', exist_ok=True)
    input_basename = os.path.basename(args.input_image).split('.')[0]
    audio_basename  = os.path.basename(args.input_audio).split('.')[0]
    output_basename = f"{input_basename}_{audio_basename}"
    video_path = os.path.join(args.output_dir, output_basename + '.mp4')
    frame_dir = os.path.join(args.output_dir, output_basename)
    truth_dir = os.path.join(args.output_dir, output_basename + '_truth')

    # 复制当前环境变量并添加 PATH
    env = os.environ.copy()

    command = ['python', 'evaluate.py',
               '--input_image', args.input_image,
               '--input_audio', args.input_audio,
               '--output_dir', args.output_dir]
    if args.ground_truth is not None:
        command += ['--ground_truth', args.ground_truth]
    
    # 传递环境变量到子进程
    subprocess.call(command, env=env)

    def extract_floats(text):
        return [float(s) for s in re.findall(r'-?\d+\.\d+', text)]
    
    subprocess.call(['python', 'syncnet/run_pipeline.py', '--videofile', video_path, "--reference", 'wav2lip', '--data_dir', 'tmp_dir'], env=env)
    result = subprocess.check_output(['python', 'syncnet/calculate_scores_real_videos.py', '--videofile', video_path, '--reference', 'wav2lip', '--data_dir', 'tmp_dir'], env=env)
    result = result.decode('utf-8')
    floats = extract_floats(result)
    LSE_D, LSE_C = floats[0], floats[1]
    with open(f'evaluation/{output_basename}.txt', 'a') as file:
        file.write(f'LSE-C: {LSE_C}\n')
        file.write(f'LSE-D: {LSE_D}\n')

    print(f'Result has been saved to evaluation/{output_basename}.txt')
