import cv2
import numpy as np
import torch
from skimage import img_as_float
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy
from scipy import stats

def compute_niqe(image):
    """
    计算图像的 NIQE 分数（无需使用 pyiqa 库）。
    通过特征提取和统计比对来计算 NIQE 分数。

    参数：
    - image (numpy.ndarray): 输入的图像。

    返回：
    - niqe_score (float): 图像的 NIQE 分数。
    """

    # 将图像转换为浮动类型，范围[0, 1]
    image = img_as_float(image)

    # 计算图像的局部二值模式（LBP）
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")

    # 计算图像的熵（信息量）
    entropy = shannon_entropy(image)

    # 计算图像的均值、标准差
    mean = np.mean(image)
    std = np.std(image)

    # 计算图像的偏度（Skewness）和峰度（Kurtosis）
    skewness = stats.skew(image.flatten())
    kurtosis = stats.kurtosis(image.flatten())

    # 通过特征合成一个简单的 NIQE 评分（简单示例，实际使用时需要根据预训练模型调整）
    niqe_score = (mean + std + skewness + kurtosis + entropy) / 5

    return niqe_score

def calculate_niqe_for_video(video_path):
    """
    计算视频每一帧的 NIQE 分数，并返回视频的平均 NIQE 分数。

    参数：
    - video_path (str): 视频文件的路径。

    返回：
    - average_niqe_score (float): 视频的平均 NIQE 分数。
    """
    # 打开 AVI 视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    niqe_scores = []

    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break

        # 将 BGR 帧转换为灰度图像
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算 NIQE 分数
        score = compute_niqe(frame_gray)
        niqe_scores.append(score)

        # 打印或存储分数
        print(f"Frame {len(niqe_scores)} NIQE score: {score}")

    # 释放视频捕获对象
    cap.release()

    # 计算平均的 NIQE 分数
    average_niqe_score = np.mean(niqe_scores)
    print(f"Average NIQE score for the video: {average_niqe_score}")

    return average_niqe_score

# 示例使用：
video_path = 'E:\\Code\\DesktopCode\\LiveSpeechPortraits\\results\\May\\May_short\\May_short.avi'  # 替换为你的视频文件路径
average_score = calculate_niqe_for_video(video_path)
if average_score is not None:
    print(f"视频的平均 NIQE 分数为: {average_score}")
