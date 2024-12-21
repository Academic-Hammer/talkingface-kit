import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim
import cv2
from PIL import Image

# 计算InceptionV3特征
def calculate_inception_features(video_path, batch_size=8):
    # 初始化InceptionV3模型
    model = inception_v3(pretrained=True, transform_input=False).eval().cuda()
    
    # 视频读取
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # 读取视频帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames.append(frame)
    cap.release()
    
    # 转换为Tensor
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    features = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        
        # 将视频帧转换为正确的维度：batch_size x channels x height x width
        batch = torch.stack([transform(frame) for frame in batch]).cuda()
        
        with torch.no_grad():
            # 提取Inception特征
            output = model(batch)
            output = output.detach().cpu().numpy()
            features.append(output)
    
    return np.concatenate(features, axis=0)

# 计算FID
def calculate_fid(real_features, generated_features):
    # 计算均值和协方差矩阵
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(generated_features, axis=0)
    cov_real = np.cov(real_features, rowvar=False)
    cov_gen = np.cov(generated_features, rowvar=False)
    
    # 计算FID
    diff = mu_real - mu_gen
    cov_sqrt, _ = sqrtm(cov_real.dot(cov_gen), disp=False)
    
    fid = np.sum(diff**2) + np.trace(cov_real + cov_gen - 2 * cov_sqrt)
    return fid

# 计算SSIM
def calculate_ssim(video_path1, video_path2):
    # 视频读取
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)
    
    ssim_values = []
    
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # 计算SSIM
        score, _ = ssim(frame1, frame2, full=True)
        ssim_values.append(score)
    
    cap1.release()
    cap2.release()
    
    return np.mean(ssim_values)

# 主程序
real_video_path = 'May_org.mp4'
generated_video_path = 'May_radnerf_torso_smo.mp4'

# 计算Inception特征
real_features = calculate_inception_features(real_video_path)
generated_features = calculate_inception_features(generated_video_path)

# 计算FID
fid_score = calculate_fid(real_features, generated_features)
print(f"FID score: {fid_score}")

# 计算SSIM
ssim_score = calculate_ssim(real_video_path, generated_video_path)
print(f"SSIM score: {ssim_score}")

