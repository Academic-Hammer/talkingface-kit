import sys
import os
import logging
import numpy as np
import cv2

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_paths():
    reference_path = None

    # 检查参数数量
    if len(sys.argv) <= 2:
        logging.error("Usage: python cpbd.py --reference <reference_file>")
        sys.exit(1)

    # 遍历参数
    for i, arg in enumerate(sys.argv):
        if arg == "--reference":
            # 检查是否有下一个参数，即文件路径
            if i + 1 < len(sys.argv):
                reference_path = sys.argv[i + 1]
                # 检查路径是否是.mp4文件
                if not reference_path.endswith((".mp4", ".avi", ".mov", ".mkv")) or not os.path.isfile(reference_path):
                    logging.error(f"The provided reference path is not a valid .mp4 file: {reference_path}")
                    sys.exit(1)
            else:
                logging.error("No file path provided after --reference argument.")
                sys.exit(1)

    if reference_path is None:
        logging.error("--reference arguments must be provided.")
        sys.exit(1)

    logging.info(f"Reference path: {reference_path}")
    return reference_path

def get_CPBD(reference_path):
    # 打开视频文件
    cap = cv2.VideoCapture(reference_path)
    
    # 存储所有帧的模糊检测分数
    blur_scores = []

    logging.info('Computing CPBD calculation...')
    # 逐帧读取视频
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 转换为灰度图像
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算当前帧的模糊检测分数
        blur_score = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
        blur_scores.append(blur_score)
    
    # 释放视频捕获对象
    cap.release()
    
    # 将模糊检测分数转换为概率值
    blur_scores_normalized = np.array(blur_scores) / np.max(blur_scores)
    
    # 计算累积概率
    sorted_scores = np.sort(blur_scores_normalized)
    cumulative_probability = np.cumsum(sorted_scores) / np.sum(sorted_scores)
    
    # 返回累积概率的平均值
    cpbd_average = np.mean(cumulative_probability)
    logging.info(f"Average CPBD: {cpbd_average}")
    return cpbd_average

if __name__ == "__main__":
    reference_path = get_paths()
    cpbd = get_CPBD(reference_path)

# python cpbd.py --reference ..\reference\May.mp4