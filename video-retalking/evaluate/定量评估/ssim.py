import sys
import os
import logging
from skimage.metrics import structural_similarity as ssim
import cv2

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_paths():
    origin_path = None
    reference_path = None

    # 检查参数数量
    if len(sys.argv) <= 4:
        logging.error("Usage: python ssim.py --origin <origin_file> --reference <reference_file>")
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

def get_SSIM(origin_path, reference_path):
    # 打开视频文件
    cap1 = cv2.VideoCapture(origin_path)
    cap2 = cv2.VideoCapture(reference_path)

    # 初始化SSIM总和和帧计数
    ssim_sum = 0
    frame_count = 0

    logging.info('Computing SSIM calculation...')

    # 逐帧读取视频
    while True:
        # 读取两个视频的下一帧
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # 如果任一视频到达末尾，则停止循环
        if not ret1 or not ret2:
            break

        # 转换为灰度图像
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # 计算两个灰度帧之间的SSIM
        ssim_val = ssim(gray1, gray2)

        # 累加SSIM值
        ssim_sum += ssim_val
        frame_count += 1

    # 如果没有帧被处理，则返回None
    if frame_count == 0:
        logging.error("No frames were processed.")
        return None

    # 计算平均SSIM
    average_ssim = ssim_sum / frame_count
    logging.info(f'Average SSIM: {average_ssim}')

    # 释放视频文件
    cap1.release()
    cap2.release()

    return average_ssim

if __name__ == "__main__":
    origin_path, reference_path = get_paths()
    average_ssim = get_SSIM(origin_path, reference_path)

# python ssim.py --origin ..\origin\May.mp4 --reference ..\reference\May.mp4