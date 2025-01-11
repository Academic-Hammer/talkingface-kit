import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch_fidelity import calculate_metrics
from lpips import LPIPS
import os

def calculate_psnr(real, generated):
    """计算峰值信噪比(PSNR)"""
    return psnr(real, generated)

def calculate_ssim(real, generated):
    """计算结构相似性(SSIM)"""
    return ssim(real, generated, multichannel=True)

def calculate_fid(real_dir, generated_dir):
    """计算Fréchet Inception Distance(FID)"""
    metrics_dict = calculate_metrics(
        input1=real_dir,
        input2=generated_dir,
        cuda=True,
        isc=False,
        fid=True,
        kid=False,
        prc=False,
        verbose=False
    )
    return metrics_dict['frechet_inception_distance']

# 全局 LPIPS 模型实例
_lpips_model = None

def calculate_lpips(real, generated):
    """计算感知相似性(LPIPS)"""
    global _lpips_model
    import torch
    from torchvision import transforms
    
    # 初始化 LPIPS 模型（仅一次）
    if _lpips_model is None:
        print("Initializing LPIPS model...")
        try:
            _lpips_model = LPIPS(net='alex', verbose=False)
            print("LPIPS model successfully initialized")
        except Exception as e:
            print(f"Failed to initialize LPIPS model: {str(e)}")
            return float('nan')
    
    # 将 numpy 数组转换为 PyTorch 张量
    transform = transforms.ToTensor()
    try:
        real_tensor = transform(real).unsqueeze(0)
        gen_tensor = transform(generated).unsqueeze(0)
        return _lpips_model(real_tensor, gen_tensor).item()
    except Exception as e:
        print(f"Error calculating LPIPS: {str(e)}")
        return float('nan')

def evaluate_video_pair(real_video, generated_video, audio_file):
    """评估视频对"""
    # 视频帧提取和评估
    print(f"Opening real video: {real_video}")
    cap_real = cv2.VideoCapture(real_video)
    if not cap_real.isOpened():
        raise FileNotFoundError(f"无法打开真实视频文件: {real_video}")

    print(f"Opening generated video: {generated_video}")
    cap_gen = cv2.VideoCapture(generated_video)
    if not cap_gen.isOpened():
        raise FileNotFoundError(f"无法打开生成视频文件: {generated_video}")

    # 获取视频信息
    print(f"Real video info: {cap_real.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap_real.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"Generated video info: {cap_gen.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap_gen.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    
    # 初始化评估结果
    results = {
        'psnr': [],
        'ssim': [],
        'lpips': []
    }
    
    while True:
        ret_real, frame_real = cap_real.read()
        ret_gen, frame_gen = cap_gen.read()
        
        if not ret_real or not ret_gen:
            print(f"Processed {len(results['psnr'])} frames")
            break
            
        # 提取生成视频的中间部分（最终生成视频）
        h, w = frame_gen.shape[:2]
        # 假设中间部分占1/3宽度，位于中间位置
        start = w // 3
        end = 2 * w // 3
        frame_gen = frame_gen[:, start:end, :]  # 取中间1/3部分
        
        # 调整生成视频帧尺寸以匹配真实视频
        frame_gen = cv2.resize(frame_gen, (frame_real.shape[1], frame_real.shape[0]))
        
        # 转换为灰度图像
        gray_real = cv2.cvtColor(frame_real, cv2.COLOR_BGR2GRAY)
        gray_gen = cv2.cvtColor(frame_gen, cv2.COLOR_BGR2GRAY)
        
        # 每10帧打印一次进度
        if len(results['psnr']) % 10 == 0:
            print(f"Processing frame {len(results['psnr'])} - Adjusted size: {frame_real.shape} -> {frame_gen.shape}")
        
        # 计算各项指标
        results['psnr'].append(calculate_psnr(gray_real, gray_gen))
        results['ssim'].append(calculate_ssim(gray_real, gray_gen))
        results['lpips'].append(calculate_lpips(frame_real, frame_gen))
    
    # 计算平均值
    final_results = {k: np.mean(v) for k, v in results.items()}
    
    # TODO: 添加LSE-C和LSE-D计算
    # 需要音频和视频的唇部运动分析
    
    return final_results

if __name__ == "__main__":
    # 配置评估文件路径
    real_video = "evaluate/data/raw/videos/Shaheen.mp4"
    generated_video = "examples/Shaheen_pred_fls_Shaheen_audio_embed.mp4"
    audio_file = "examples/Shaheen.wav"
    
    # 运行评估
    results = evaluate_video_pair(real_video, generated_video, audio_file)
    
    # 输出结果
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric.upper()}: {value:.4f}")