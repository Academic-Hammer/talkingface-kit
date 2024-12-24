import cv2
import numpy as np
import os
import random
from PIL import Image
import lpips
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
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

import os
import shutil

def clear_folder(folder_path):
    """
    清空指定文件夹，包括文件和子文件夹。

    :param folder_path: str，文件夹路径
    """
    if not os.path.exists(folder_path):
        print(f"路径不存在：{folder_path}")
        return

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # 删除文件或符号链接
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # 删除子文件夹及其内容
        except Exception as e:
            print(f"无法删除 {item_path}：{e}")









#---------------------------------------------mp4-to-png--------------------------------------#
def extract_random_frame_from_video(mp4_original_path, saved_to_gt_frame_path):
    cap = cv2.VideoCapture(mp4_original_path)
    assert cap.isOpened()

    frame_filenames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(saved_to_gt_frame_path, f"{frame_count:05d}.png")
            cv2.imwrite(frame_filename, frame)
            frame_filenames.append(frame_filename)
            frame_count += 1
        else:
            break
    cap.release()

    # TODO Randomly select a reference image from the to-be-modified videos, you can modify the selection logic as per your needs
    assert len(frame_filenames) > 0
    image_path = random.choice(frame_filenames)
    return image_path, frame_filenames

#--------------------------------SSIM--------------------------------#
# 用于衡量两幅图像在结构、亮度、对比度等方面的相似性。
# SSIM能够更好地模拟人眼对图像质量的感知，比PSNR更为直观和有效。
def SSIM(oimg_path,gimg_path):
    # from skimage.metrics import structural_similarity as ssim
    # import cv2
    # import numpy as np
    #
    # # 读取原始图像和处理后的图像
    # original = cv2.imread(oimg_path, cv2.IMREAD_GRAYSCALE)
    # compressed = cv2.imread(gimg_path, cv2.IMREAD_GRAYSCALE)
    #
    # # 计算SSIM
    # ssim_value, _ = ssim(original, compressed, full=True)
    #
    # print(f'SSIM: {ssim_value}')

    from skimage.metrics import structural_similarity as ssim
    from skimage import io

    # 读取两幅图像,转256*256
    image2 = io.imread(oimg_path, as_gray=True)
    image1 = io.imread(gimg_path, as_gray=True)

    height, width = image1.shape[:2]
    image2_resized = cv2.resize(image2, (width, height))

    # 计算SSIM
    ssim_score = ssim(image1, image2_resized,data_range=1)

    # print(f"SSIM: {ssim_score}")
    return ssim_score


#------------------------------------------------------------------------

# 加载预训练的LPIPS模型
lpips_model = lpips.LPIPS(net='alex')

#---------------------------LPIPS--------------------------------#
def LPIPS(oimage_path,pimage_path):

    # 加载两幅图像
    image1 = Image.open(oimage_path).convert('RGB')
    image2 = Image.open(pimage_path).convert('RGB')

    # 对图像进行预处理
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    image1 = preprocess(image1).unsqueeze(0)
    image2 = preprocess(image2).unsqueeze(0)

    # 使用LPIPS模型计算相似性
    similarity_score = lpips_model(image1, image2)

    # print(f"LPIPS Similarity: {similarity_score.item()}")
    return similarity_score.item()
#-----------------------------------------------------------------------------------------

#-------------------------------------PSRN----------------------------------#
def PSRN(oimage_path,pimage_path):
    # 读取原始图像和重建图像
    original_image = cv2.imread(oimage_path)
    reconstructed_image = cv2.imread(pimage_path)

    height, width = reconstructed_image.shape[:2]
    original_resized = cv2.resize(original_image, (width, height))

    # 计算均方误差（MSE）
    mse = np.mean((original_resized - reconstructed_image) ** 2)

    # 计算PSNR
    max_pixel_value = 255  # 对于8位图像，最大像素值为255
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)

    # print(f"PSNR: {psnr} dB")
    return psnr
#------------------------------------------------------------------------#


def main(origin_path, generate_path, temp_path):
    cssim = 0
    cpsnr = 0
    clpips = 0

    folder1 = os.path.join(temp_path, "./origin_image")  # 替换为第一个文件夹路径
    folder2 = os.path.join(temp_path, "./generate_image")  # 替换为第二个文件夹路径
    os.makedirs(folder1, exist_ok=True)
    os.makedirs(folder2, exist_ok=True)
    clear_folder(folder1)
    clear_folder(folder2)

    o_image_path, oframe = extract_random_frame_from_video(origin_path, folder1)
    g_image_path, gframe = extract_random_frame_from_video(generate_path, folder2)
    image_pairs = get_image_pairs(folder1, folder2)
    length = len(image_pairs)
    print("len:",length)

    for img1, img2 in image_pairs:
        o_image_path = img1
        g_image_path = img2

        cssim += SSIM(o_image_path, g_image_path)
        clpips += LPIPS(o_image_path, g_image_path)
        cpsnr += PSRN(o_image_path, g_image_path)

    cssim /= length
    clpips /= length
    cpsnr /= length

    return cssim, clpips, cpsnr







