import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

orig_video_path = 'ref/Shaheen_256.mp4'
gen_video_path = 'eval/mine/Shaheen.mp4'

cap_orig = cv2.VideoCapture(orig_video_path)
cap_gen = cv2.VideoCapture(gen_video_path)

# 存储 PSNR 和 SSIM 值的列表
psnr_values = []
ssim_values = []

while cap_orig.isOpened() and cap_gen.isOpened():
    ret_orig, frame_orig = cap_orig.read()
    ret_gen, frame_gen = cap_gen.read()

    if not ret_orig or not ret_gen:
        break

    # 转换为灰度图
    gray_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
    gray_gen = cv2.cvtColor(frame_gen, cv2.COLOR_BGR2GRAY)

    # 计算 PSNR
    psnr_value = psnr(gray_orig, gray_gen)
    psnr_values.append(psnr_value)

    # 计算 SSIM
    ssim_value, _ = ssim(gray_orig, gray_gen, full=True)
    ssim_values.append(ssim_value)

cap_orig.release()
cap_gen.release()

# 计算平均 PSNR 和 SSIM
average_psnr = np.mean(psnr_values)
average_ssim = np.mean(ssim_values)

print('Average PSNR:', average_psnr)
print('Average SSIM:', average_ssim)