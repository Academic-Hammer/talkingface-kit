import numpy as np
import scipy.ndimage
from scipy.stats import norm
import scipy.ndimage
from scipy.stats import skew, kurtosis
from skimage import io, img_as_float, color
import cv2
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
# def compute_mscn_coefficients(image, kernel_size=7, sigma=7 / 6):
#     """
#     Compute MSCN coefficients for the input image using OpenCV.
#     """
#     C = 1e-6  # Small constant to avoid division by zero
#
#     # Ensure kernel size is odd
#     kernel_size = int(2 * round(kernel_size / 2) + 1)
#
#     # Compute mean (mu)
#     mu = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
#     mu_sq = mu * mu
#
#     # Compute standard deviation (sigma)
#     sigma_map = np.sqrt(np.abs(cv2.GaussianBlur(image ** 2, (kernel_size, kernel_size), sigma) - mu_sq))
#
#     # Compute MSCN coefficients
#     mscn = (image - mu) / (sigma_map + C)
#     return mscn
#
# def niqe_features(image):
#     """
#     Extract features for NIQE based on MSCN coefficients.
#     """
#     mscn = compute_mscn_coefficients(image,sigma=7/6)
#     # Compute basic statistics (mean, variance, skewness, kurtosis, etc.)
#     mean = np.mean(mscn)
#     var = np.var(mscn)
#     skew = scipy.stats.skew(mscn.flatten())
#     kurt = scipy.stats.kurtosis(mscn.flatten())
#     return np.array([mean, var, skew, kurt])
#
# def calculate_niqe(image):
#     """
#     Placeholder function for NIQE calculation using pre-trained parameters.
#     """
#     features = niqe_features(image)
#     # Use pre-trained NIQE model (not provided here).
#     # You'd need the parameters from the original NIQE paper or scikit-image.
#     # For now, we'll just return a mock value.
#     return np.random.rand() * 10  # Replace with real model computation.
#
# # Example usage
# if __name__ == "__main__":
#     from skimage import io, img_as_float, color
#     image_path = "./generate_image/00014.png"  # Replace with your image path
#     image = img_as_float(io.imread(image_path))
#     if image.ndim == 3:  # Convert to grayscale if RGB
#         image = color.rgb2gray(image)
#     score = calculate_niqe(image)
#     print(f"NIQE Score: {score}")

# ------------------------上面是随机采样的NIQE -------------------------------#





def compute_mscn_coefficients(image, kernel_size=7, sigma=7 / 6):
    """
    Compute MSCN coefficients for the input image using OpenCV.
    """
    C = 1e-6  # Small constant to avoid division by zero

    # Ensure kernel size is odd
    kernel_size = int(2 * round(kernel_size / 2) + 1)

    # Compute mean (mu)
    mu = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    mu_sq = mu * mu

    # Compute standard deviation (sigma)
    sigma_map = np.sqrt(np.abs(cv2.GaussianBlur(image ** 2, (kernel_size, kernel_size), sigma) - mu_sq))

    # Compute MSCN coefficients
    mscn = (image - mu) / (sigma_map + C)
    return mscn

def niqe_features(image):
    """
    Extract features for NIQE based on MSCN coefficients.
    """
    mscn = compute_mscn_coefficients(image)
    mean = np.mean(mscn)
    var = np.var(mscn)
    skewness = skew(mscn.flatten())
    kurt = kurtosis(mscn.flatten())
    return np.array([mean, var, skewness, kurt])

def calculate_niqe(image):
    """
    Calculate NIQE score for a single image.
    """
    features = niqe_features(image)
    # Placeholder: Replace with actual NIQE model parameters for real computation
    # For now, return the mean of extracted features as a mock score
    return np.mean(features)

def main(origin_path, generate_path, temp_path):
    cniqe = 0
    count = 0

    folder1 = os.path.join(temp_path, "./origin_image")  # 替换为第一个文件夹路径
    folder2 = os.path.join(temp_path, "./generate_image")  # 替换为第二个文件夹路径

    image_pairs = get_image_pairs(folder1, folder2)
    length = len(image_pairs)
    print("len:",length)

    for img1, img2 in image_pairs:
        count += 1
        image = img_as_float(io.imread(img2))
        if image.ndim == 3:  # Convert to grayscale if RGB
            image = color.rgb2gray(image)
        # Calculate NIQE score
        niqe_score = calculate_niqe(image)
        cniqe += niqe_score
        print(count, " : ", niqe_score)

    fin = cniqe / length
    return fin