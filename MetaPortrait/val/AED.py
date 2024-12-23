import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import euclidean
from tqdm import tqdm

def load_images_from_folder(folder_path):
    """
    从文件夹中按顺序加载图像。
    :param folder_path: 文件夹路径
    :return: 图像列表
    """
    images = []
    for filename in sorted(os.listdir(folder_path)):  # 按文件名排序
        filepath = os.path.join(folder_path, filename)
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # 过滤图像文件
            img = cv2.imread(filepath)
            if img is not None:
                images.append(img)
    return images

def preprocess_image(image):
    """
    预处理图像以适配 InceptionResnetV1 模型。
    :param image: 输入图像 (OpenCV 格式, BGR)
    :return: 预处理后的图像 (torch.Tensor)
    """
    # 转为 RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 调整大小为 160x160（InceptionResnetV1 输入尺寸）
    img_resized = cv2.resize(img_rgb, (160, 160))

    # 转为 Tensor 格式并归一化到 [0, 1]
    img_tensor = torch.tensor(img_resized / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    return img_tensor.to(device)

def extract_face_features(image, model):
    """
    使用 InceptionResnetV1 提取人脸特征。
    :param image: 输入图像 (OpenCV 格式)
    :param model: 预训练的 InceptionResnetV1 模型
    :return: 人脸特征向量 (np.array)
    """
    img_tensor = preprocess_image(image)
    with torch.no_grad():
        features = model(img_tensor)  # 提取特征
    return features.cpu().numpy().flatten()

def calculate_aed(generated_folder, driving_folder, model):
    """
    计算 AED。
    :param generated_folder: 生成视频帧文件夹路径
    :param driving_folder: 驱动视频帧文件夹路径
    :param model: 预训练的 InceptionResnetV1 模型
    :return: AED 值
    """
    # 加载帧图像
    generated_images = load_images_from_folder(generated_folder)
    driving_images = load_images_from_folder(driving_folder)

    assert len(generated_images) == len(driving_images), "帧数量不一致！"

    distances = []

    # 遍历每一帧
    for g_img, d_img in tqdm(zip(generated_images, driving_images), total=len(generated_images), desc="Calculating AED"):
        # 提取生成视频和驱动视频的特征
        g_features = extract_face_features(g_img, model)
        d_features = extract_face_features(d_img, model)

        # 计算特征的欧几里得距离
        distances.append(euclidean(g_features, d_features))

    # 计算平均距离
    aed = np.mean(distances)
    return aed

# 文件夹路径
# generated_folder = "Jae-in"
# driving_folder = "base_model"
#
# # 计算 AED
# aed = calculate_aed(generated_folder, driving_folder, model)
# print(f"Average Expression Distance (AED):{aed:.4f}")

import argparse
import os

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Calculate Average Expression Distance (AED) between generated and driving images',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 添加命令行参数
parser.add_argument('--generated_folder', type=str, required=True, help='Directory containing generated images')
parser.add_argument('--driving_folder', type=str, required=True, help='Directory containing driving images')
parser.add_argument('--model_path', type=str, default='vggface2.pt', help='Path to the model used for AED calculation')

# 解析命令行参数
args = parser.parse_args()

# 初始化 InceptionResnetV1 模型（加载预训练权重）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 使用本地下载的模型权重文件
model = InceptionResnetV1(pretrained=None)  # 初始化模型，不加载预训练权重
state_dict = torch.load(args.model_path)  # 替换为本地权重文件路径
# 过滤掉多余的键
# filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
# 加载权重
model.load_state_dict(state_dict, strict=False)  # strict=False 忽略未匹配的键
pose_estimator = model.eval().to(device)  # 设置为评估模式

# 计算 AED
aed = calculate_aed(args.generated_folder, args.driving_folder, model)
print(f"Average Expression Distance (AED): {aed:.4f}")
