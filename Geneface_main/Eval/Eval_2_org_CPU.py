import numpy as np
import cv2
from sklearn.decomposition import PCA
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# 加载InceptionV3模型
def load_inception_model():
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    return model

# 提取图像特征
def extract_inception_features(video_path, model, batch_size=8):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # 读取视频帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (299, 299))
        frames.append(frame)
    cap.release()
    
    # 转换为符合InceptionV3输入要求的形式
    features = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        
        # 将帧转换为符合模型输入要求的形状
        batch = np.array(batch)
        batch = preprocess_input(batch)
        
        # 获取特征
        batch_features = model.predict(batch, verbose = 0)
        features.append(batch_features)
    
    features = np.vstack(features)
    return features

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

    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    
    fid = np.sum(diff**2) + np.trace(cov_real + cov_gen - 2 * cov_sqrt)
    return fid

# 计算SSIM
def calculate_ssim(video_path1, video_path2):
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
generated_video_path = 'May_radnerf_torso_smo_ORGMODEL.mp4'

# 加载InceptionV3模型
model = load_inception_model()

# 计算InceptionV3特征
real_features = extract_inception_features(real_video_path, model)
generated_features = extract_inception_features(generated_video_path, model)

# 计算FID
fid_score = calculate_fid(real_features, generated_features)
print(f"FID score: {fid_score}")

# 计算SSIM
ssim_score = calculate_ssim(real_video_path, generated_video_path)
print(f"SSIM score: {ssim_score}")
