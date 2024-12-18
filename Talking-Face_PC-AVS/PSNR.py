import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr


cap_orig = cv2.VideoCapture('raw/videos/Macron_224.mp4')
cap_gen = cv2.VideoCapture('results/id_May_cropped_pose_Macron_cropped_audio_Shaheen/G_Pose_Driven_.mp4')


psnr_values = []
while cap_orig.isOpened() and cap_gen.isOpened():
    ret_orig, frame_orig = cap_orig.read()
    ret_gen, frame_gen = cap_gen.read()
    if not ret_orig or not ret_gen:
        break

   
    gray_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
    gray_gen = cv2.cvtColor(frame_gen, cv2.COLOR_BGR2GRAY)

   
    psnr_value = psnr(gray_orig, gray_gen)
    psnr_values.append(psnr_value)


average_psnr = np.mean(psnr_values)
print('Average PSNR:', average_psnr)