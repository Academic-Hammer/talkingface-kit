import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def psnr(img1, img2):
    """计算两帧图片的 PSNR"""
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# 读取视频帧并预处理
def preprocess_frame(frame):
    """将视频帧转为 Tensor 并归一化到 [0, 1]"""
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转为 PyTorch 张量并自动归一化到 [0, 1]
    ])
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # 转为 PIL Image 格式
    frame_tensor = transform(frame).unsqueeze(0)  # 增加 batch 维度
    return frame_tensor

# 逐帧计算 PSNR
def evaluate_video_psnr(input_video, output_video):
    """逐帧计算视频的 PSNR，并返回平均值"""
    cap_input = cv2.VideoCapture(input_video)
    cap_output = cv2.VideoCapture(output_video)

    if not cap_input.isOpened():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if not cap_output.isOpened():
        raise FileNotFoundError(f"Output video not found: {output_video}")

    psnr_scores = []
    frame_index = 0

    while cap_input.isOpened() and cap_output.isOpened():
        ret_input, frame_input = cap_input.read()
        ret_output, frame_output = cap_output.read()

        if not ret_input or not ret_output:
            break

        # 确保帧大小一致
        if frame_input.shape != frame_output.shape:
            frame_output = cv2.resize(frame_output, (frame_input.shape[1], frame_input.shape[0]))

        # 转换为 Tensor
        tensor_input = preprocess_frame(frame_input)
        tensor_output = preprocess_frame(frame_output)

        # 计算 PSNR
        psnr_score = psnr(tensor_input, tensor_output).item()
        psnr_scores.append(psnr_score)

        frame_index += 1

    cap_input.release()
    cap_output.release()

    # 计算视频的平均 PSNR
    avg_psnr = np.mean(psnr_scores) if psnr_scores else 0
    return avg_psnr, psnr_scores

if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Compute PSNR between original and inferred videos.")
    parser.add_argument('--origin', type=str, required=True, help="Path to the original video.")
    parser.add_argument('--reference', type=str, required=True, help="Path to the inferred video.")
    args = parser.parse_args()

    logging.info(f"Origin path: {args.origin}")
    logging.info(f"Reference path: {args.reference}")

    # 逐帧计算 PSNR 并获取平均值
    try:
        logging.info("Computing PSNR calculation...")
        avg_psnr, frame_psnrs = evaluate_video_psnr(args.origin, args.reference)
        logging.info(f"Average PSNR: {avg_psnr:.2f} dB")
    except FileNotFoundError as e:
        logging.error(f"Error: {e}")

# python psnr.py --origin ..\origin\May.mp4 --reference ..\reference\May.mp4