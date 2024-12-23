import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(image1, image2):
    # 将图片转换为灰度图像
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # 计算SSIM值
    ssim_value, _ = ssim(gray1, gray2, full=True)
    return ssim_value

def calculate_ssim_between_folders(folder1, folder2):
    # 获取两个文件夹中的所有图片文件
    images1 = sorted(os.listdir(folder1))
    images2 = sorted(os.listdir(folder2))

    # 检查文件夹中的图片数量是否相等
    if len(images1) != len(images2):
        raise ValueError("两个文件夹中的图片数量不一致")

    ssim_values = []
    
    for img1_name, img2_name in zip(images1, images2):
        img1_path = os.path.join(folder1, img1_name)
        img2_path = os.path.join(folder2, img2_name)

        # 读取图片
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            raise ValueError(f"无法读取图片: {img1_path} 或 {img2_path}")

        # 计算每对图片的SSIM
        ssim_value = calculate_ssim(img1, img2)
        ssim_values.append(ssim_value)

    # 计算平均SSIM
    average_ssim = np.mean(ssim_values)
    return average_ssim

if __name__ == "__main__":
    import argparse

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Calculate average SSIM between two image folders')

    # 添加命令行参数
    parser.add_argument('--generated_folder', type=str, required=True, help='Directory containing generated images')
    parser.add_argument('--driving_folder', type=str, required=True, help='Directory containing driving images')

    # 解析命令行参数
    args = parser.parse_args()

    # 计算平均SSIM
    average_ssim = calculate_ssim_between_folders(args.generated_folder, args.driving_folder)

    # 输出结果
    print(f"两个图片集之间的平均SSIM值: {average_ssim:.4f}")