import shutil
import subprocess
import sys
import os

def run_commands(video_file_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    # 定义要执行的命令
    command1 = [
        "python", "run_pipeline.py",
        "--videofile", video_file_path,
        "--reference", "wav2lip",
        "--data_dir", "tmp_dir"
    ]

    command2 = [
        "python", "calculate_scores_real_videos.py",
        "--videofile", video_file_path,
        "--reference", "wav2lip",
        "--data_dir", "tmp_dir"
    ]
    
    try:
        # 预处理
        subprocess.run(command1, check=True, stdout=subprocess.DEVNULL)
        
        # 评估
        if os.path.exists("all_scores.txt"):
            os.remove("all_scores.txt")
        with open("all_scores.txt", "a") as score_file:
            subprocess.run(command2, check=True, stdout=score_file)

        # 删除 tmp_dir 目录
        if os.path.exists("tmp_dir"):
            shutil.rmtree("tmp_dir")
    
    except subprocess.CalledProcessError as e:
        print(f"{e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python <script_name.py> <video_file_path>")
        sys.exit(1)

    # 获取命令行参数
    video_file_path = sys.argv[1]

    # 调用函数执行命令
    run_commands(video_file_path)
