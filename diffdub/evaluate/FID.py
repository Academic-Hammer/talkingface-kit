# import torch
# import numpy as np
# from scipy.linalg import sqrtm
# from torchvision import models, transforms
# from PIL import Image
# import os
#
# # Function to calculate the FID score
# def calculate_fid(real_images, generated_images):
#     """
#     Calculate FID score between two sets of images.
#
#     Args:
#         real_images (list of PIL.Image): List of real images.
#         generated_images (list of PIL.Image): List of generated images.
#
#     Returns:
#         float: FID score.
#     """
#     # Define InceptionV3 model
#     inception_model = models.inception_v3(pretrained=True, transform_input=False)
#     inception_model.fc = torch.nn.Identity()  # Remove final classification layer
#     inception_model.eval()
#
#     # Transform for real and generated images
#     transform = transforms.Compose([
#         transforms.Resize((299, 299)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#     ])
#
#     def get_features(images):
#         """Extract features from images using InceptionV3."""
#         features = []
#         for img in images:
#             img = transform(img).unsqueeze(0)  # Add batch dimension
#             with torch.no_grad():
#                 pred = inception_model(img)  # Output is already pooled
#                 features.append(pred.squeeze().numpy())
#         return np.array(features)
#
#     # Extract features
#     real_features = get_features(real_images)
#     gen_features = get_features(generated_images)
#
#     # Calculate mean and covariance
#     mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
#     mu_gen, sigma_gen = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
#
#     # Calculate FID
#     diff = mu_real - mu_gen
#     covmean, _ = sqrtm(sigma_real @ sigma_gen, disp=False)
#     if np.iscomplexobj(covmean):
#         covmean = covmean.real
#
#     fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
#     return fid
#
# # Example usage
# if __name__ == "__main__":
#     # Load real and generated images
#     real_images_dir = "./origin_image"  # Replace with actual directory
#     generated_images_dir = "./generate_image"  # Replace with actual directory
#
#     real_images = []
#     generated_images = []
#
#     # Load all images with filenames starting from 00000
#     for filename in sorted(os.listdir(real_images_dir)):
#         if filename.endswith(".png"):
#             with Image.open(os.path.join(real_images_dir, filename)) as img:
#                 real_images.append(img.copy())  # 复制图片，避免文件句柄未关闭
#
#     for filename in sorted(os.listdir(generated_images_dir)):
#         if filename.endswith(".png"):
#             with Image.open(os.path.join(generated_images_dir, filename)) as img:
#                 generated_images.append(img.copy())
#
#     # Resize generated images to match real image dimensions (636x636 to 256x256)
#     real_images = [img.resize((636, 636)) for img in real_images]
#     generated_images = [img.resize((256, 256)) for img in generated_images]
#
#     # Calculate FID score
#     fid_score = calculate_fid(real_images, generated_images)
#     print(f"FID score: {fid_score}")
#
#
#

import torch
import numpy as np
from scipy.linalg import sqrtm
from torchvision import models, transforms
from PIL import Image
import os

# Function to calculate the FID score
def calculate_fid(real_images, generated_images, device):
    """
    Calculate FID score between two sets of images.

    Args:
        real_images (list of PIL.Image): List of real images.
        generated_images (list of PIL.Image): List of generated images.
        device (torch.device): The device (CPU or GPU) to perform computation on.

    Returns:
        float: FID score.
    """
    # Define InceptionV3 model and move to GPU
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.fc = torch.nn.Identity()  # Remove final classification layer
    inception_model = inception_model.to(device)  # Move model to GPU/CPU
    inception_model.eval()

    # Transform for real and generated images
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    def get_features(images):
        """Extract features from images using InceptionV3."""
        features = []
        for img in images:
            img = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device
            with torch.no_grad():
                pred = inception_model(img)  # Output is already pooled
                features.append(pred.squeeze().cpu().numpy())  # Move back to CPU for NumPy compatibility
        return np.array(features)

    # Extract features
    real_features = get_features(real_images)
    gen_features = get_features(generated_images)

    # Calculate mean and covariance
    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)

    # Calculate FID
    diff = mu_real - mu_gen
    covmean, _ = sqrtm(sigma_real @ sigma_gen, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid


def main(original_path, generated_path, temp_path):
    # Set the device to GPU if available, otherwise fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load real and generated images
    real_images_dir = os.path.join(temp_path, "./origin_image")  # Replace with actual directory
    generated_images_dir = os.path.join(temp_path, "./generate_image")  # Replace with actual directory

    real_images = []
    generated_images = []

    # Load all images with filenames starting from 00000
    for filename in sorted(os.listdir(real_images_dir)):
        if filename.endswith(".png"):
            with Image.open(os.path.join(real_images_dir, filename)) as img:
                real_images.append(img.copy())  # 复制图片，避免文件句柄未关闭

    for filename in sorted(os.listdir(generated_images_dir)):
        if filename.endswith(".png"):
            with Image.open(os.path.join(generated_images_dir, filename)) as img:
                generated_images.append(img.copy())

    # Resize generated images to match real image dimensions (636x636 to 256x256)
    real_images = [img.resize((636, 636)) for img in real_images]
    generated_images = [img.resize((256, 256)) for img in generated_images]

    # Calculate FID score
    fid_score = calculate_fid(real_images, generated_images, device)
    return fid_score