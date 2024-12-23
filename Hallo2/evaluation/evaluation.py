import cv2
import numpy as np
from moviepy.editor import VideoFileClip,AudioFileClip
from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import peak_signal_noise_ratio  as psnr
from skimage import measure
import librosa
import torch
from torchvision import models, transforms
from scipy.linalg import sqrtm
import os
import time
import shutil
import argparse

# NIQE
from niqe_python.main import niqe

# FID
from FID.FID import calculate_fid

# LSE
from LSE.calculate_scores_LRS import calculate_scores

# 1. 计算SSIM指标
def calculate_ssim_batch(original_images, generated_images):
    ssim_scores = []
    for orig_img, gen_img in zip(original_images, generated_images):
        # print(orig_img.shape)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
        gen_img = cv2.cvtColor(gen_img, cv2.COLOR_RGB2GRAY)
        ssim_index, _ = ssim(orig_img, gen_img, full=True)
        ssim_scores.append(ssim_index)
    return np.mean(ssim_scores)

# 2. 计算PSNR指标
def calculate_psnr_batch(original_images, generated_images):
    psnr_scores = []
    for orig_img, gen_img in zip(original_images, generated_images):
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
        gen_img = cv2.cvtColor(gen_img, cv2.COLOR_RGB2GRAY)
        mse = np.mean((orig_img - gen_img) ** 2)
        if mse == 0:
            psnr = 100
        else:
            PIXEL_MAX = 255.0
            psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
        # psnr_result = psnr(orig_img, gen_img)
        psnr_scores.append(psnr)
    return np.mean(psnr_scores)
# 3. 计算NIQE指标
def calculate_niqe_batch(original_images):
    niqe_scores = []
    for orig_img in original_images:
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
        niqe_score = niqe(orig_img)  # This is a placeholder
        niqe_scores.append(niqe_score)
    return np.mean(niqe_scores)


    
def save_frames(video_path, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame=cv2.resize(frame, (2048,2048))
        frame_filename = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1
    
    cap.release()

    return frame_idx

def read_frames_batch(frame_dir, batch_size, len, start_idx=0):
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    
    batch_frames = []
    for idx in range(start_idx, start_idx + batch_size):
        if idx < len:
            frame_path = os.path.join(frame_dir, frame_files[idx])
            frame = cv2.imread(frame_path)   
            batch_frames.append(frame)
    
    return np.array(batch_frames), start_idx + batch_size

def process_video_in_batches(original_video_path, generated_video_path, opt, batch_size=32):
    original_output_dir = 'original_frames'
    generated_output_dir = 'generated_frames'

    start_time = time.time()

    original_len = save_frames(original_video_path,original_output_dir)
    generated_len = save_frames(generated_video_path,generated_output_dir)
    total_len = min(original_len, generated_len)

    print(f"load photo total time: {time.time()-start_time}")

    start_idx=0
    ssim_result = 0
    psnr_result = 0
    niqe_result = 0

    start_time = time.time()

    while True:
        # 读取批次数据
        original_batch, next_idx = read_frames_batch(original_output_dir, batch_size, total_len, start_idx)
        generated_batch, _ = read_frames_batch(generated_output_dir, batch_size, total_len, start_idx)

        # print(original_batch.shape)

        if len(original_batch) == 0 or len(generated_batch) == 0:
            break

        batch_len = len(original_batch)

        print(f"time: {time.time()-start_time}")
        if opt.ssim==True:
            # 计算指标（例如，SSIM）
            batch_ssim_result = calculate_ssim_batch(original_batch, generated_batch)
            print(f"\tSSIM for batch starting at frame {start_idx}: {batch_ssim_result}, time: {time.time()-start_time}")
            ssim_result += batch_ssim_result * batch_len

        if opt.psnr==True:
            # 计算指标（例如，PSNR）
            batch_psnr_result = calculate_psnr_batch(original_batch, generated_batch)
            print(f"\tPSNR for batch starting at frame {start_idx}: {batch_psnr_result}, time: {time.time()-start_time}")
            psnr_result += batch_psnr_result * batch_len

        if opt.niqe==True:
            # 计算指标（例如，NIQE）
            batch_niqe_result = calculate_niqe_batch(original_batch)
            print(f"\tNIQE for batch starting at frame {start_idx}: {batch_niqe_result}, time: {time.time()-start_time}")
            niqe_result += batch_niqe_result * batch_len

        # 更新start_idx，以便读取下一个批次
        start_idx = next_idx


    if opt.lse==True:
        print("LSE:")
        # 计算指标（LSE-C & LSE-D）
        lse_c,lse_d = calculate_scores(generated_video_path)

    # 确保输出目录存在
    os.makedirs(opt.output_dir, exist_ok=True)

    # 假设要写入的结果
    evaluation_result = f"{opt.original_video_path}:\n"
    if opt.ssim==True:
        evaluation_result += f"\tSSIM: {ssim_result*1.0/total_len}\n"
    if opt.psnr==True:
        evaluation_result += f"\tPSNR: {psnr_result*1.0/total_len}\n"
    if opt.niqe==True:
        evaluation_result += f"\tNIQE: {niqe_result*1.0/total_len}\n"
    if opt.lse==True:
        evaluation_result += f"\tLSE-C: {lse_c}\n\tLSE-D: {lse_d}\n"
    

    # 写入到 evaluation.txt 文件
    evaluation_file_path = os.path.join(opt.output_dir, "evaluation.txt")

    with open(evaluation_file_path, "a") as f:
        f.write(evaluation_result)

# 输入参数
parser = argparse.ArgumentParser(description = "Total")

# parser.add_argument('--initial_model', type=str, default="./LSE/data/syncnet_v2.model", help='')
# parser.add_argument('--batch_size', type=int, default='20', help='')
# parser.add_argument('--vshift', type=int, default='15', help='')
# parser.add_argument('--data_input', type=str, default=f"{video_path}", help='')
# parser.add_argument('--data_root', type=str, default="./LSE/data/merge/", help='')
# parser.add_argument('--tmp_dir', type=str, default="./LSE/data/work/pytmp", help='')
# parser.add_argument('--reference', type=str, default="demo", help='')

parser.add_argument('--original_video_path', type=str, default="./input/processed.mp4", help='original video path', required=True)
parser.add_argument('--generated_video_path', type=str, default="./input/merge_video.mp4", help='generated video path', required=True)
parser.add_argument('--output_dir', type=str, default="./output", help='output directory',required=True)
parser.add_argument('--batch_size', type=int, default='32', help='batch size')
parser.add_argument('--niqe', type=bool, default=False, help='whether to calculate NIQE', required=False)
parser.add_argument('--ssim', type=bool, default=False, help='whether to calculate SSIM', required=False)
parser.add_argument('--psnr', type=bool, default=False, help='whether to calculate PSNR', required=False)
parser.add_argument('--fid', type=bool, default=False, help='whether to calculate FID', required=False)
parser.add_argument('--lse', type=bool, default=False, help='whether to calculate LSE-C & D', required=False)

opt = parser.parse_args()

sum = opt.niqe + opt.ssim + opt.psnr + opt.fid + opt.lse
# 至少选择一个指标
if sum == 0:
    print("At least one metric should be selected")
    exit(1)

process_video_in_batches(opt.original_video_path, opt.generated_video_path, opt, opt.batch_size)

if opt.fid==True:
    # 计算指标（ID）
    fid_result = calculate_fid()
    print(f"\tFID: {fid_result}\n")
    evaluation_file_path = os.path.join(opt.output_dir, "evaluation.txt")
    with open(evaluation_file_path, "a") as f:
        f.write(f"\tFID: {fid_result}\n")
