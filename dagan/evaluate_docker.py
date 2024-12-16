import os
import yaml
import json
import psutil
import traceback
import sys
from datetime import datetime
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from skimage.transform import resize
from skimage import img_as_ubyte
import modules.generator as GEN
from modules.keypoint_detector import KPDetector
import depth
from collections import OrderedDict
import numpy as np
from evaluation_metrics import EvaluationMetrics
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def save_checkpoint(metrics, output_path, batch_idx):
    """保存评估进度检查点"""
    checkpoint_dir = os.path.join(output_path, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_batch_{batch_idx}.json')

    # 转换 NumPy 数组和其他特殊类型为普通 Python 类型
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        return obj

    # 转换metrics为JSON可序列化格式
    json_metrics = {}
    for gen_dir, gen_metrics in metrics.items():
        json_metrics[gen_dir] = {
            k: convert_for_json(v) for k, v in gen_metrics.items()
        }

    with open(checkpoint_file, 'w') as f:
        json.dump(json_metrics, f, indent=4)


class EvaluationDataset(Dataset):
    def __init__(self, base_dir, generate_dirs, transform=None, max_samples=None):
        self.base_dir = base_dir
        # 确保 generate_dirs 是字符串列表
        self.generate_dirs = [gen_dir for gen_dir in generate_dirs]
        self.transform = transform

        # 初始化源目录和GT目录
        gen_dir = self.generate_dirs[0]  # 使用第一个生成目录来决定使用哪个源和GT目录
        if '_self' in gen_dir:  # 如果有 _self 后缀，使用 self 目录
            self.source_dir = os.path.join(base_dir, 'source_self')
            self.gt_dir = os.path.join(base_dir, 'gt_self')
        else:  # 如果没有后缀或其他情况，使用 cross 目录
            self.source_dir = os.path.join(base_dir, 'source_cross')
            self.gt_dir = os.path.join(base_dir, 'gt_cross')

        # 打印使用的目录信息
        print(f"Using source directory: {self.source_dir}")
        print(f"Using GT directory: {self.gt_dir}")

        # 检查目录是否存在
        if not os.path.exists(self.source_dir):
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")
        if not os.path.exists(self.gt_dir):
            raise FileNotFoundError(f"GT directory not found: {self.gt_dir}")

        # 获取所有图像文件并排序
        self.image_files = sorted([f for f in os.listdir(self.source_dir)
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])

        # 如果指定了最大样本数，限制数据集大小
        if max_samples is not None:
            self.image_files = self.image_files[:max_samples]
            print(f"使用前 {max_samples} 个样本进行评估")

        self._verify_files()

    def _verify_files(self):
        """验证所有必需的图像文件是否存在"""
        for img_file in self.image_files:
            # 检查源图像
            source_path = os.path.join(self.source_dir, img_file)
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Missing source image: {source_path}")

            # 检查GT图像
            gt_path = os.path.join(self.gt_dir, img_file)
            if not os.path.exists(gt_path):
                raise FileNotFoundError(f"Missing GT image: {gt_path}")

            # 检查生成的图像
            for gen_dir in self.generate_dirs:
                gen_path = os.path.join(self.base_dir, gen_dir, img_file)
                if not os.path.exists(gen_path):
                    raise FileNotFoundError(f"Missing generated image: {gen_path}")

    def _load_image(self, path):
        """加载并预处理图像"""
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        else:
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            img = transform(img)
        return img  # 返回CPU张量

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """获取数据集中的一项"""
        img_name = self.image_files[idx]

        # 加载源图像和GT图像
        source = self._load_image(os.path.join(self.source_dir, img_name))
        driving = self._load_image(os.path.join(self.gt_dir, img_name))

        # 加载所有生成的图像
        generated_images = []
        for gen_dir in self.generate_dirs:
            gen_path = os.path.join(self.base_dir, gen_dir, img_name)
            gen_img = self._load_image(gen_path)
            generated_images.append(gen_img)

        return {
            'source': source,
            'driving': driving,
            'generated': generated_images,
            'filename': img_name,
            'generate_dirs': self.generate_dirs
        }

def create_comparison_grid(source_img, driving_img, generated_imgs, generate_dirs, save_path):
    """创建包含源图像、驱动图像和所有生成图像的对比网格"""

    def prepare_img(img):
        # 图像预处理：归一化到[0,1]范围
        img = (img + 1) / 2.0
        img = img_as_ubyte(img)
        img = img.astype(np.float32) / 255.0
        return img

    source_img = prepare_img(source_img)
    driving_img = prepare_img(driving_img)
    generated_imgs = [prepare_img(img) for img in generated_imgs]

    # 计算网格尺寸
    n_generated = len(generated_imgs)
    n_cols = n_generated + 2  # source + driving + generated

    # 创建图表
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    # 绘制源图像和驱动图像
    axes[0].imshow(source_img)
    axes[0].set_title('Source')
    axes[0].axis('off')

    axes[1].imshow(driving_img)
    axes[1].set_title('Driving')
    axes[1].axis('off')

    # 绘制生成的图像
    for idx, (gen_img, gen_name) in enumerate(zip(generated_imgs, generate_dirs)):
        axes[idx + 2].imshow(gen_img)
        axes[idx + 2].set_title(f'Generated\n{gen_name}')
        axes[idx + 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_with_monitoring(evaluator, data_loader, device, output_path,
                             checkpoint_freq=5, save_visuals=True):
    if save_visuals:
        vis_dir = os.path.join(output_path, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

    accumulated_metrics = {}

    pbar = tqdm(enumerate(data_loader), total=len(data_loader),
                desc="Evaluation Progress", ncols=100, leave=True)

    for batch_idx, batch in pbar:
        try:
            # 添加调试输出
            print(f"\nBatch {batch_idx} generate_dirs type: {type(batch['generate_dirs'])}")
            print(f"generate_dirs content: {batch['generate_dirs']}")

            source = batch['source'].to(device)
            driving = batch['driving'].to(device)
            generated_list = [gen.to(device) for gen in batch['generated']]

            # Ensure generate_dirs is a list of strings
            generate_dirs = batch['generate_dirs']
            if isinstance(generate_dirs[0], list):
                generate_dirs = generate_dirs[0]  # Get the first element if it's a nested list

            # Initialize metrics for new generation methods
            for gen_dir in generate_dirs:
                gen_dir = str(gen_dir)  # Ensure it's a string
                if gen_dir not in accumulated_metrics:
                    accumulated_metrics[gen_dir] = {
                        'niqe': [], 'psnr': [], 'ssim': [],
                        'fid': [], 'lse_c': [], 'lse_d': []
                    }

            source_image = source.cpu().numpy()[0].transpose(1, 2, 0)
            driving_image = driving.cpu().numpy()[0].transpose(1, 2, 0)
            generated_images = [gen.cpu().numpy()[0].transpose(1, 2, 0)
                                for gen in generated_list]

            source_image = np.clip(source_image, 0, 1)
            driving_image = np.clip(driving_image, 0, 1)
            generated_images = [np.clip(gen, 0, 1) for gen in generated_images]

            if save_visuals:
                vis_path = os.path.join(vis_dir, f'comparison_batch_{batch_idx}.png')
                create_comparison_grid(source_image, driving_image,
                                       generated_images, generate_dirs, vis_path)

            # 确保图像尺寸足够大
            min_size = 32  # 设置最小尺寸
            if (source_image.shape[0] < min_size or source_image.shape[1] < min_size or
                    driving_image.shape[0] < min_size or driving_image.shape[1] < min_size):
                raise ValueError(f"Image size too small. Minimum size required: {min_size}x{min_size}")

            for gen_idx, (gen_image, gen_dir) in enumerate(zip(generated_images, generate_dirs)):
                if gen_image.shape[0] < min_size or gen_image.shape[1] < min_size:
                    raise ValueError(f"Generated image size too small. Minimum size required: {min_size}x{min_size}")

                try:
                    batch_metrics = evaluator.evaluate_batch(source_image,
                                                             driving_image,
                                                             gen_image)
                    # 确保gen_dir是字符串
                    gen_dir = str(gen_dir)
                    for metric, value in batch_metrics.items():
                        if isinstance(value, list):
                            accumulated_metrics[gen_dir][metric].extend(value)
                        else:
                            accumulated_metrics[gen_dir][metric].append(value)
                except Exception as metric_error:
                    pbar.write(f"\nWarning: Error computing metrics for {gen_dir}: {str(metric_error)}")

            del source, driving, generated_list
            torch.cuda.empty_cache()

            if (batch_idx + 1) % checkpoint_freq == 0:
                save_checkpoint(accumulated_metrics, output_path, batch_idx)

        except Exception as e:
            pbar.write(f"\nError in batch {batch_idx}: {str(e)}")
            raise

    pbar.close()
    return accumulated_metrics


def save_final_results(metrics, output_path):
    """保存所有生成方法的评估结果
    Args:
        metrics (dict): 评估指标字典
        output_path (str): 结果保存路径
    """

    def convert_metrics(metric_value):
        """转换度量值为Python原生类型
        Args:
            metric_value: 需要转换的度量值
        Returns:
            转换后的度量值
        """
        if isinstance(metric_value, np.ndarray):
            return metric_value.tolist()
        elif isinstance(metric_value, (np.float32, np.float64)):
            return float(metric_value)
        elif isinstance(metric_value, list):
            return [convert_metrics(item) for item in metric_value]
        elif isinstance(metric_value, dict):
            return {k: convert_metrics(v) for k, v in metric_value.items()}
        elif hasattr(metric_value, 'item'):  # 处理PyTorch张量
            return metric_value.item()
        else:
            return metric_value

    # 打印输入metrics的结构以便调试
    print("\nInput metrics structure:")
    for gen_dir, gen_metrics in metrics.items():
        print(f"\nProcessing metrics for {gen_dir}:")
        for metric_name, metric_values in gen_metrics.items():
            print(f"{metric_name}: type={type(metric_values)}, value={metric_values}")

    # 计算每个生成方法的平均指标
    results = {}
    for gen_dir, gen_metrics in metrics.items():
        avg_metrics = {}
        for metric_name, metric_values in gen_metrics.items():
            if isinstance(metric_values, list) and metric_values:
                # 对列表类型的指标取平均值
                valid_values = [v for v in metric_values if v is not None]
                avg_metrics[metric_name] = float(np.mean(valid_values)) if valid_values else 0.0
            elif isinstance(metric_values, (float, int)):
                # 对单个数值的指标直接使用
                avg_metrics[metric_name] = float(metric_values)
            elif metric_values is not None:
                # 处理其他非空情况
                try:
                    avg_metrics[metric_name] = float(metric_values)
                except (TypeError, ValueError):
                    print(f"Warning: Unable to process metric {metric_name} with value {metric_values}")
                    avg_metrics[metric_name] = 0.0
            else:
                # 处理None值情况
                avg_metrics[metric_name] = 0.0

        results[gen_dir] = {
            'average_metrics': avg_metrics,
            'detailed_metrics': convert_metrics(gen_metrics)
        }

    # 保存详细结果为JSON格式
    results_file = os.path.join(output_path, 'full_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    # 保存摘要结果为文本格式
    summary_file = os.path.join(output_path, 'evaluation_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        for gen_dir, gen_results in results.items():
            f.write(f"\n=== Evaluation Results for {gen_dir} ===\n")

            # 添加样本数量信息
            samples_count = len(metrics[gen_dir].get('niqe', [])) if isinstance(metrics[gen_dir].get('niqe'),
                                                                                list) else 1
            f.write(f"Total samples processed: {samples_count}\n\n")

            for metric, value in gen_results['average_metrics'].items():
                # 根据指标类型使用不同的小数位数
                if metric in ['niqe', 'psnr', 'ssim']:
                    f.write(f"{metric}: {value:.4f}\n")
                else:
                    f.write(f"{metric}: {value:.6f}\n")
            f.write("\n")

    print(f"\nResults saved to:")
    print(f"- Full results: {results_file}")
    print(f"- Summary: {summary_file}")

    # 打印处理后的结果以便验证
    print("\nProcessed metrics:")
    for gen_dir, gen_results in results.items():
        print(f"\n{gen_dir} average metrics:")
        for metric, value in gen_results['average_metrics'].items():
            print(f"{metric}: {value}")

    return results


def main():
    parser = ArgumentParser()
    parser.add_argument("--base_dir", required=True,
                        help="Base directory containing source/, gt/ and generate_*/ folders")
    parser.add_argument("--output_path", default="evaluation_results",
                        help="Path to save results")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--save_visuals", action='store_true',
                        help="Whether to save comparison visualizations")
    parser.add_argument("--checkpoint_frequency", type=int, default=5,
                        help="Save progress every N batches")
    parser.add_argument("--generate_dirs", nargs='+', default=None,
                        help="Specific generate directories to evaluate (optional)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    try:
        # Find all generate_* directories or use specified ones
        if args.generate_dirs:
            generate_dirs = args.generate_dirs
        else:
            generate_dirs = [d for d in os.listdir(args.base_dir)
                             if d.startswith('generate_')]

        if not generate_dirs:
            raise ValueError("No generate_* directories found in base directory")

        print(f"Found generation directories: {generate_dirs}")

        dataset = EvaluationDataset(
            base_dir=args.base_dir,
            generate_dirs=generate_dirs,
            max_samples=args.max_samples
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        print("Starting evaluation...")
        evaluator = EvaluationMetrics(device=args.device)

        with torch.no_grad():
            metrics = evaluate_with_monitoring(
                evaluator,
                dataloader,
                args.device,
                args.output_path,
                checkpoint_freq=args.checkpoint_frequency,
                save_visuals=args.save_visuals
            )

        save_final_results(metrics, args.output_path)
        print(f"\nResults saved to {args.output_path}")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()