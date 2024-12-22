import cv2
import torch_fidelity
import os
import shutil
import logging
import sys

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_paths():
    origin_path = None
    reference_path = None

    # 检查参数数量
    if len(sys.argv) <= 4:
        logging.error("Usage: python fid.py --origin <origin_file> --reference <reference_file>")
        sys.exit(1)

    # 遍历参数
    for i, arg in enumerate(sys.argv):
        if arg == "--origin":
            # 检查是否有下一个参数，即文件路径
            if i + 1 < len(sys.argv):
                origin_path = sys.argv[i + 1]
                # 检查路径是否是.mp4文件
                if not origin_path.endswith((".mp4", ".avi", ".mov", ".mkv")) or not os.path.isfile(origin_path):
                    logging.error(f"The provided origin path is not a valid .mp4 file: {origin_path}")
                    sys.exit(1)
            else:
                logging.error("No file path provided after --origin argument.")
                sys.exit(1)

        elif arg == "--reference":
            # 检查是否有下一个参数，即文件路径
            if i + 1 < len(sys.argv):
                reference_path = sys.argv[i + 1]
                # 检查路径是否是.mp4文件
                if not reference_path.endswith(".mp4") or not os.path.isfile(reference_path):
                    logging.error(f"The provided reference path is not a valid .mp4 file: {reference_path}")
                    sys.exit(1)
            else:
                logging.error("No file path provided after --reference argument.")
                sys.exit(1)

    if origin_path is None or reference_path is None:
        logging.error("Both --origin and --reference arguments must be provided.")
        sys.exit(1)

    logging.info(f"Origin path: {origin_path}")
    logging.info(f"Reference path: {reference_path}")
    return origin_path, reference_path

def extract_frames(video_path, output_dir):
    """
    将视频逐帧提取并保存为图像文件。
    
    参数:
        video_path (str): 输入视频的文件路径。
        output_dir (str): 用于保存提取帧的目标目录路径。

    功能:
        使用 OpenCV 加载视频文件，并逐帧读取视频内容，将每一帧保存为 PNG 图像文件。
        图像文件以 "frame_00000.png", "frame_00001.png" 等命名方式保存。

    注意:
        - 如果指定的目录不存在，会自动创建。
        - 提取完成后，释放视频文件资源。
    """
    os.makedirs(output_dir, exist_ok=True)  # 确保目标目录存在
    cap = cv2.VideoCapture(video_path)  # 打开视频文件
    frame_idx = 0 # 初始化帧索引

    while True:
        ret, frame = cap.read() # 读取一帧
        if not ret: # 如果读取失败（视频结束），退出循环
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.png") # 生成帧文件路径
        cv2.imwrite(frame_path, frame)   # 将帧保存为 PNG 文件
        frame_idx += 1  # 更新帧索引

    cap.release()  # 释放视频资源


def calculate_fid_for_videos(video1_path, video2_path):
    """
    计算两个视频之间的 Frechet Inception Distance (FID) 值。
    
    参数:
        video1_path (str): 第一个视频的文件路径。
        video2_path (str): 第二个视频的文件路径。

    返回:
        float: 两个视频的 FID 值。

    功能:
        1. 提取两个视频的所有帧，并分别存储到临时目录中。
        2. 使用 `torch_fidelity.calculate_metrics` 计算两个帧目录之间的 FID。
        3. 删除临时帧目录以释放磁盘空间。

    注意:
        - 需要确保两个视频帧提取后的图像可以被 torch-fidelity 处理。
        - 清理临时目录以避免磁盘空间浪费。
    """
    # 定义临时目录用于保存视频帧
    temp_dir1 = "temp_video1_frames"
    temp_dir2 = "temp_video2_frames"

    logging.info('Extracting frames from videos...')
    # 提取视频帧到临时目录
    extract_frames(video1_path, temp_dir1)
    extract_frames(video2_path, temp_dir2)

    logging.info('Computing FID calculation...')
    # 使用 torch-fidelity 计算 FID
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=temp_dir1, # 第一个视频帧目录
        input2=temp_dir2, # 第二个视频帧目录
        cuda=False, # 使用 CPU 计算
        fid=True,  # 计算 FID
        verbose=True  # 显示计算过程中的详细信息
    )

    # 删除临时帧目录以释放磁盘空间
    shutil.rmtree(temp_dir1)
    shutil.rmtree(temp_dir2)

    fid_value = metrics_dict["frechet_inception_distance"]
    logging.info(f"Average FID: {fid_value}")
    # 返回 FID 值
    return fid_value

if __name__ == "__main__":
    origin_path, reference_path = get_paths()
    fid_value = calculate_fid_for_videos(origin_path, reference_path)

# python fid.py --origin ..\origin\May.mp4 --reference ..\reference\May.mp4