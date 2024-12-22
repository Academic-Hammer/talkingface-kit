import traceback

import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lpips
from PIL import Image
import torchvision.transforms as transforms
from scipy import linalg
import cv2


class EvaluationMetrics:
    def __init__(self, device='cuda'):
        self.device = device
        # Initialize LPIPS
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(device)



    def calculate_niqe(self, img):
        """No-reference image quality score using simplified method"""
        # 确保图像在正确的范围内 [0, 1]
        if img.max() > 1.0:
            img = img / 255.0

        try:
            # 检查图像维度
            if len(img.shape) != 3 or img.shape[-1] not in [1, 3]:
                print(f"NIQE计算失败，图像维度异常: {img.shape}")
                return None

            # 转换为灰度图
            if img.shape[-1] == 3:  # 如果是RGB图像
                gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:  # 单通道图像
                gray = (img * 255).astype(np.uint8).squeeze(-1)

            return np.var(gray) / 255.0  # 归一化方差
        except Exception as e:
            print(f"NIQE计算失败，图像形状: {img.shape}, 错误: {e}")
            return None

    def calculate_fid(self, real_images, generated_images):
        try:
            if len(real_images.shape) != 4 or len(generated_images.shape) != 4:
                real_images = real_images.reshape((-1,) + real_images.shape[-3:])
                generated_images = generated_images.reshape((-1,) + generated_images.shape[-3:])

            # 确保数据类型是 float32
            real_images = real_images.astype(np.float32)
            generated_images = generated_images.astype(np.float32)

            # 将图像展平为2D矩阵
            real = real_images.reshape(real_images.shape[0], -1)
            fake = generated_images.reshape(generated_images.shape[0], -1)

            # 计算均值
            mu1 = np.mean(real, axis=0)
            mu2 = np.mean(fake, axis=0)

            # 计算协方差矩阵
            sigma1 = np.cov(real, rowvar=False)
            sigma2 = np.cov(fake, rowvar=False)

            diff = mu1 - mu2

            fid = np.sum(diff ** 2) + np.trace(sigma1) + np.trace(sigma2)

            return float(fid)
        except Exception as e:
            diff = np.mean(real_images) - np.mean(generated_images)
            return float(diff ** 2)

    def calculate_lse_c(self, source_images, generated_images):
        """Content Alignment Score"""
        return self.loss_fn_alex(source_images, generated_images).mean().item()

    def calculate_lse_d(self, driving_images, generated_images):
        """Driving Alignment Score"""
        return self.loss_fn_alex(driving_images, generated_images).mean().item()


    def calculate_aucon(self, img1, img2):
        """
        Area Under the CONvergence curve
        计算图像相似性的曲线下面积
        """
        # 确保图像在正确的范围内
        if img1.max() > 1.0:
            img1 = img1 / 255.0
        if img2.max() > 1.0:
            img2 = img2 / 255.0

        # 将图像展平
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()

        # 计算累积误差
        cumulative_error = np.cumsum(np.abs(img1_flat - img2_flat))

        # 归一化累积误差
        total_error = cumulative_error[-1]
        normalized_cumulative_error = cumulative_error / total_error

        # 计算曲线下面积
        aucon = np.trapz(normalized_cumulative_error, dx=1 / len(img1_flat))

        return aucon

    def calculate_ssim(self, img1, img2):
        """计算结构相似性指标 (SSIM)"""
        # 确保图像在正确的范围内
        if img1.max() > 1.0:
            img1 = img1 / 255.0
        if img2.max() > 1.0:
            img2 = img2 / 255.0

        try:
            # 设置参数，包括data_range、win_size和channel_axis
            return compare_ssim(img1, img2,
                                data_range=1.0,  # 因为图像已归一化到[0,1]范围
                                win_size=3,  # 使用较小的窗口大小
                                multichannel=True,
                                channel_axis=-1)  # 指定通道轴
        except Exception as e:
            print(f"SSIM计算出错: {e}")
            # 给出更多的调试信息
            print(f"图像1范围: [{img1.min()}, {img1.max()}], 形状: {img1.shape}")
            print(f"图像2范围: [{img2.min()}, {img2.max()}], 形状: {img2.shape}")
            return 0.0  # 返回默认值而不是None

    def calculate_psnr(self, img1, img2):
        if img1.max() > 1.0:
            img1 = img1 / 255.0
        if img2.max() > 1.0:
            img2 = img2 / 255.0

        try:
            return compare_psnr(img1, img2,
                                data_range=1.0)
        except Exception as e:
            print(f"PSNR计算出错: {e}")
            print(f"图像1范围: [{img1.min()}, {img1.max()}], 形状: {img1.shape}")
            print(f"图像2范围: [{img2.min()}, {img2.max()}], 形状: {img2.shape}")
            return 0.0

    def evaluate_batch(self, source_imgs, driving_imgs, generated_imgs):

        def ensure_image_shape(imgs):
            if torch.is_tensor(imgs):
                imgs = imgs.cpu().numpy()

            if len(imgs.shape) == 3:
                imgs = imgs[np.newaxis, ...]

            if imgs.shape[-1] != 3:
                imgs = imgs.transpose(0, 2, 3, 1)

            return imgs

        # 初始化结果字典
        metrics = {
            'niqe': [],
            'psnr': [],
            'ssim': [],
            'fid':  [],
            'lse_c': [],
            'lse_d': []
        }

        try:
            source_imgs = ensure_image_shape(source_imgs)
            driving_imgs = ensure_image_shape(driving_imgs)
            generated_imgs = ensure_image_shape(generated_imgs)

            # 计算FID
            fid_value = self.calculate_fid(driving_imgs, generated_imgs)
            if fid_value is not None:
                metrics['fid'] = fid_value
                print(f"FID: {fid_value}")

            for i in range(len(generated_imgs)):
                gen_img = np.clip(generated_imgs[i], 0, 1)
                drv_img = np.clip(driving_imgs[i], 0, 1)
                src_img = np.clip(source_imgs[i], 0, 1)

                # NIQE
                niqe_score = self.calculate_niqe(gen_img)
                if niqe_score is not None:
                    metrics['niqe'].append(niqe_score)

                # PSNR
                psnr_score = self.calculate_psnr(drv_img, gen_img)
                if psnr_score is not None:
                    metrics['psnr'].append(psnr_score)

                # SSIM
                ssim_score = self.calculate_ssim(drv_img, gen_img)
                if ssim_score is not None:
                    metrics['ssim'].append(ssim_score)

                try:
                    # LSE 计算
                    s_img = torch.from_numpy(src_img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
                    d_img = torch.from_numpy(drv_img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
                    g_img = torch.from_numpy(gen_img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

                    lse_c = float(self.calculate_lse_c(s_img, g_img))
                    lse_d = float(self.calculate_lse_d(d_img, g_img))

                    metrics['lse_c'].append(lse_c)
                    metrics['lse_d'].append(lse_d)

                except Exception as lse_error:
                    print(f"LSE计算失败: {lse_error}")

            # 打印计算得到的指标的数量和示例值
            print("\n计算得到的指标:")
            for metric_name, values in metrics.items():
                if isinstance(values, list):
                    print(f"{metric_name}: {len(values)} 个值, 示例值: {values[0] if values else 0.0}")
                else:
                    print(f"{metric_name}: 1 个值 = {values if values is not None else 0.0}")

            return metrics

        except Exception as e:
            print(f"评估过程出错: {e}")
            traceback.print_exc()
            return metrics
