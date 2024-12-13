import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import color
import torch
import lpips
from torchvision import models, transforms

# 设置 LPIPS 模型（可以计算 FID）
lpips_model = lpips.LPIPS(net='alex')  # 或者'vgg' 根据需要选择模型

# 计算 PSNR
def calculate_psnr(image1, image2):
    """计算 PSNR"""
    return psnr(image1, image2)

# 计算 SSIM
def calculate_ssim(image1, image2):
    """计算 SSIM"""
    min_dim = min(image1.shape[:2])
    if min_dim < 7:
        new_size = max(3, min_dim)
        image1 = cv2.resize(image1, (new_size, new_size))
        image2 = cv2.resize(image2, (new_size, new_size))
        min_dim = new_size
    win_size = 3
    if win_size % 2 == 0:
        win_size -= 1
    return ssim(image1, image2, multichannel=True, win_size=win_size)

# 计算 LSE-C
def calculate_LSE_C(real_frame, gen_frame):
    """计算LSE-C: 以均方误差 (MSE) 为基础的对比度损失"""
    real_frame = real_frame.astype(np.float32) / 255.0
    gen_frame = gen_frame.astype(np.float32) / 255.0
    mse = np.mean((real_frame - gen_frame) ** 2)
    return mse

# 计算 LSE-D
def calculate_LSE_D(real_frame, gen_frame):
    """计算LSE-D: 计算亮度差异的均方误差"""
    real_brightness = np.mean(real_frame, axis=(0, 1))
    gen_brightness = np.mean(gen_frame, axis=(0, 1))
    brightness_diff = np.mean((real_brightness - gen_brightness) ** 2)
    return brightness_diff

# 计算 NIQE
def calculate_niqe(image):
    """计算 NIQE 指标"""
    gray_image = color.rgb2gray(image)  # 转为灰度图
    # 近似 NIQE 的简单计算：使用图像标准差
    std_dev = np.std(gray_image)
    niqe_score = 10 * np.log10(1 / std_dev) if std_dev > 0 else float('inf')
    return niqe_score

# 加载视频帧
def load_video_frames(video_path):
    """加载视频并返回帧列表"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"视频 {video_path} 加载了 {len(frames)} 帧")
    return frames

# 计算 FID（Fréchet Inception Distance）
def calculate_fid(real_frames, gen_frames):
    """计算 FID（Fréchet Inception Distance）"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()

    def get_inception_features(frames):
        features = []
        with torch.no_grad():
            for frame in frames:
                frame = transform(frame).unsqueeze(0)
                if frame.size(3) == 3:
                    frame = frame.cuda()
                feature = inception_model(frame)
                feature = feature.view(feature.size(0), -1)
                features.append(feature.cpu().numpy())
        return np.array(features)

    real_features = get_inception_features(real_frames)
    gen_features = get_inception_features(gen_frames)

    real_features = np.squeeze(real_features, axis=1)
    gen_features = np.squeeze(gen_features, axis=1)

    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
        diff = mu1 - mu2
        covmean = np.sqrt(sigma1 @ sigma2)
        return diff @ diff.T + np.trace(sigma1 + sigma2 - 2 * covmean)

    def calculate_statistics(features):
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    mu1, sigma1 = calculate_statistics(real_features)
    mu2, sigma2 = calculate_statistics(gen_features)

    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid

# 对两个视频进行评测
def evaluate_videos(real_video_path, gen_video_path):
    """评测两部视频的各种指标"""
    real_frames = load_video_frames(real_video_path)
    gen_frames = load_video_frames(gen_video_path)

    if len(real_frames) != len(gen_frames):
        print(f"视频帧数不一致: {len(real_frames)} vs {len(gen_frames)}")
        min_frame_count = min(len(real_frames), len(gen_frames))
        real_frames = real_frames[:min_frame_count]
        gen_frames = gen_frames[:min_frame_count]
        print(f"已调整为相同帧数: {min_frame_count}")

    real_frame_shape = real_frames[0].shape
    gen_frame_shape = gen_frames[0].shape

    if real_frame_shape != gen_frame_shape:
        print(f"帧尺寸不一致，正在调整尺寸...")
        target_size = (min(real_frame_shape[1], gen_frame_shape[1]), min(real_frame_shape[0], gen_frame_shape[0]))
        real_frames = [cv2.resize(frame, target_size) for frame in real_frames]
        gen_frames = [cv2.resize(frame, target_size) for frame in gen_frames]
        print(f"已调整为统一尺寸: {target_size}")

    psnr_values = []
    ssim_values = []
    lse_c_values = []
    lse_d_values = []
    niqe_values = []

    for real_frame, gen_frame in zip(real_frames, gen_frames):
        real_frame = cv2.cvtColor(real_frame, cv2.COLOR_BGR2RGB)
        gen_frame = cv2.cvtColor(gen_frame, cv2.COLOR_BGR2RGB)

        psnr_values.append(calculate_psnr(real_frame, gen_frame))
        ssim_values.append(calculate_ssim(real_frame, gen_frame))
        lse_c_values.append(calculate_LSE_C(real_frame, gen_frame))
        lse_d_values.append(calculate_LSE_D(real_frame, gen_frame))
        niqe_values.append(calculate_niqe(gen_frame))

    fid_value = calculate_fid(real_frames, gen_frames)

    return {
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values),
        'lse_c': np.mean(lse_c_values),
        'lse_d': np.mean(lse_d_values),
        'niqe': np.mean(niqe_values),
        'fid': fid_value
    }

# 主函数
def main():
    real_video_folder = './processed'
    gen_video_folder = './testgen_metric'

    real_video_files = [f for f in os.listdir(real_video_folder) if f.endswith('.mp4')]
    gen_video_files = [f for f in os.listdir(gen_video_folder) if f.endswith('.mp4')]

    common_files = set(real_video_files).intersection(set(gen_video_files))
    print(f"找到的同名视频文件: {common_files}")

    for video_file in common_files:
        print(f"开始评测视频: {video_file}")
        real_video_path = os.path.join(real_video_folder, video_file)
        gen_video_path = os.path.join(gen_video_folder, video_file)

        result = evaluate_videos(real_video_path, gen_video_path)

        print(f"NIQE: {result['niqe']:.2f}")
        print(f"PSNR: {result['psnr']:.2f}")
        print(f"SSIM: {result['ssim']:.2f}")
        print(f"LSE-C: {result['lse_c']:.2f}")
        print(f"LSE-D: {result['lse_d']:.2f}")
        print(f"FID: {result['fid']:.2f}")
        print('-' * 40)

if __name__ == "__main__":
    main()