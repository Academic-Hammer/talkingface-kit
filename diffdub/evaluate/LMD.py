import cv2
import numpy as np
from scipy.spatial.distance import euclidean
import dlib
import os

def get_image_pairs(folder1, folder2):
    """
    遍历两个文件夹，返回图片名称相同（基于序号）的两张图片路径。

    :param folder1: str，第一个文件夹路径
    :param folder2: str，第二个文件夹路径
    :return: list，每对图片路径组成的元组 (image1_path, image2_path)
    """
    # 获取两个文件夹中的图片文件列表
    images1 = {os.path.basename(f): os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith('.png')}
    images2 = {os.path.basename(f): os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith('.png')}

    # 找到两个文件夹中共有的图片名称
    common_files = sorted(set(images1.keys()) & set(images2.keys()))

    # 返回匹配的图片路径对
    return [(images1[file], images2[file]) for file in common_files]
# 计算两个关键点之间的欧氏距离
def calculate_lmd(landmarks1, landmarks2):
    """
    计算两个图像的关键点距离 (LMD)

    参数:
    - landmarks1: 第一个图像的关键点 (列表或数组)
    - landmarks2: 第二个图像的关键点 (列表或数组)

    返回:
    - LMD值（所有对应关键点之间的平均距离）
    """
    distances = []
    for (x1, y1), (x2, y2) in zip(landmarks1, landmarks2):
        distance = euclidean((x1, y1), (x2, y2))
        distances.append(distance)

    # 计算所有关键点之间的平均欧氏距离
    lmd_score = np.mean(distances)
    return lmd_score


# 使用 Dlib 或 OpenCV 检测关键点
def detect_landmarks(image, detector, predictor):
    """
    使用 Dlib 检测图像中的关键点

    参数:
    - image: 输入图像
    - detector: Dlib的人脸检测器
    - predictor: Dlib的关键点预测器

    返回:
    - 关键点坐标列表
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    landmarks = []
    for face in faces:
        points = predictor(gray, face)
        landmarks = [(point.x, point.y) for point in points.parts()]

    return landmarks


# 加载 Dlib 的人脸检测器和关键点预测器


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # 下载并提供预测器模型文件

def main(original_path, generated_path, temp_dir):
    clmd = 0
    count = 0

    folder1 = os.path.join(temp_dir, "./origin_image")  # 替换为第一个文件夹路径
    folder2 = os.path.join(temp_dir, "./generate_image")  # 替换为第二个文件夹路径

    image_pairs = get_image_pairs(folder1, folder2)
    length = len(image_pairs)
    print("len:",length)

    for img1, img2 in image_pairs:
        count += 1
        image1 = cv2.imread(img1)
        image2 = cv2.imread(img2)
        height, width = image2.shape[:2]
        image1_resized = cv2.resize(image1, (width, height))

        # 检测两幅图像中的关键点
        landmarks1 = detect_landmarks(image1_resized, detector, predictor)
        landmarks2 = detect_landmarks(image2, detector, predictor)

        # 计算LMD（Landmarks Distance）
        if landmarks1 and landmarks2:
            lmd_score = calculate_lmd(landmarks1, landmarks2)
            clmd += lmd_score
            print(count,":",lmd_score)
        else:
            print("未检测到关键点")

    return clmd / length









