import gc
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
from demo_img import normalize_kp, make_animation  # <-- 导入 make_animation（如果需要）
from collections import OrderedDict
import numpy as np
from evaluation_metrics import EvaluationMetrics
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from evaluation_dataset import EvaluationDataset
from PIL import Image

class MemoryMonitor:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, 'memory_log.txt')
        os.makedirs(log_dir, exist_ok=True)

    def get_memory_info(self):
        """获取详细的内存使用情况"""
        process = psutil.Process(os.getpid())

        memory_info = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cpu': {
                'percent': process.cpu_percent(),
                'rss': process.memory_info().rss / (1024 * 1024),  # MB
                'vms': process.memory_info().vms / (1024 * 1024),  # MB
                'system_total': psutil.virtual_memory().total / (1024 * 1024),  # MB
                'system_available': psutil.virtual_memory().available / (1024 * 1024),  # MB
                'system_percent': psutil.virtual_memory().percent
            },
            'gpu': {
                'allocated': torch.cuda.memory_allocated() / (1024 * 1024),  # MB
                'reserved': torch.cuda.memory_reserved() / (1024 * 1024),  # MB
                'max_allocated': torch.cuda.max_memory_allocated() / (1024 * 1024),  # MB
            } if torch.cuda.is_available() else None
        }
        return memory_info

    def log_memory(self, additional_info=""):
        """记录内存使用情况"""
        info = self.get_memory_info()

        with open(self.log_file, 'a') as f:
            f.write(f"\n--- {info['timestamp']} ---\n")
            f.write(f"Additional Info: {additional_info}\n")
            f.write("CPU Memory:\n")
            f.write(f"  Process RSS: {info['cpu']['rss']:.2f} MB\n")
            f.write(f"  Process VMS: {info['cpu']['vms']:.2f} MB\n")
            f.write(f"  System Total: {info['cpu']['system_total']:.2f} MB\n")
            f.write(f"  System Available: {info['cpu']['system_available']:.2f} MB\n")
            f.write(f"  System Usage: {info['cpu']['system_percent']}%\n")

            if info['gpu']:
                f.write("GPU Memory:\n")
                f.write(f"  Allocated: {info['gpu']['allocated']:.2f} MB\n")
                f.write(f"  Reserved: {info['gpu']['reserved']:.2f} MB\n")
                f.write(f"  Max Allocated: {info['gpu']['max_allocated']:.2f} MB\n")

            f.write("-" * 50 + "\n")


def setup_error_logging(output_dir):
    """设置错误日志"""
    error_log = os.path.join(output_dir, 'error_log.txt')

    def log_exception(exc_type, exc_value, exc_traceback):
        with open(error_log, 'a') as f:
            f.write(f"\n--- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)

    sys.excepthook = log_exception
    return error_log


def load_checkpoints(config_path, checkpoint_path, kp_num=15, generator_type='DepthAwareGenerator', cpu=False):
    """加载预训练的模型"""
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['model_params']['common_params']['num_kp'] = kp_num

    generator = getattr(GEN, generator_type)(
        **config['model_params']['generator_params'],
        **config['model_params']['common_params']
    )

    if not cpu:
        generator.cuda()

    config['model_params']['common_params']['num_channels'] = 4
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()

    checkpoint = torch.load(checkpoint_path, map_location="cuda:0" if not cpu else "cpu")

    ckp_generator = OrderedDict((k.replace('module.', ''), v) for k, v in checkpoint['generator'].items())
    generator.load_state_dict(ckp_generator)

    ckp_kp_detector = OrderedDict((k.replace('module.', ''), v) for k, v in checkpoint['kp_detector'].items())
    kp_detector.load_state_dict(ckp_kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector


def load_depth_model(cpu=False):
    """加载深度估计模型"""
    depth_encoder = depth.ResnetEncoder(18, False)
    depth_decoder = depth.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4))

    loaded_dict_enc = torch.load('depth/models/weights_19/encoder.pth', map_location='cpu' if cpu else 'cuda')
    loaded_dict_dec = torch.load('depth/models/weights_19/depth.pth', map_location='cpu' if cpu else 'cuda')

    depth_encoder.load_state_dict({k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()})
    depth_decoder.load_state_dict(loaded_dict_dec)

    depth_encoder.eval()
    depth_decoder.eval()

    if not cpu:
        depth_encoder.cuda()
        depth_decoder.cuda()

    return depth_encoder, depth_decoder


def save_checkpoint(metrics, output_path, batch_idx):
    """保存检查点"""
    checkpoint = {
        'batch_idx': batch_idx,
        'metrics': metrics
    }
    checkpoint_path = os.path.join(output_path, f'checkpoint_batch_{batch_idx}.pth')
    torch.save(checkpoint, checkpoint_path)


def save_final_results(metrics, output_path):
    """保存最终结果"""
    def convert_metrics(metric_value):
        """递归转换NumPy数据类型为Python原生类型"""
        if isinstance(metric_value, np.ndarray):
            return metric_value.tolist()
        elif isinstance(metric_value, np.float32):
            return float(metric_value)
        elif isinstance(metric_value, list):
            return [convert_metrics(item) for item in metric_value]
        elif isinstance(metric_value, dict):
            return {k: convert_metrics(v) for k, v in metric_value.items()}
        else:
            return metric_value

    # 计算平均指标，确保转换为Python float
    avg_metrics = {k: float(np.mean(v)) for k, v in metrics.items() if v and isinstance(v, list)}

    # 对指标进行完全转换
    converted_metrics = convert_metrics(metrics)

    # 保存详细结果
    results_file = os.path.join(output_path, 'full_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'average_metrics': avg_metrics,
            'detailed_metrics': converted_metrics
        }, f, indent=4)

    # 保存摘要结果
    summary_file = os.path.join(output_path, 'evaluation_results.txt')
    with open(summary_file, 'w') as f:
        for metric, value in avg_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")


def evaluate_with_monitoring(generator, kp_detector, depth_encoder, depth_decoder,
                             evaluator, data_loader, device, output_path,
                             checkpoint_freq=5, save_visuals=True,
                             relative=True, adapt_movement_scale=True):
    """Enhanced evaluation function matching demo.py image generation approach"""
    if save_visuals:
        vis_dir = os.path.join(output_path, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

    memory_monitor = MemoryMonitor(output_path)
    accumulated_metrics = {
        'niqe': [],
            'psnr': [],
            'ssim': [],
            'fid':  [],
            'lse_c': [],
            'lse_d': []
    }

    memory_monitor.log_memory("Initial state")
    pbar = tqdm(enumerate(data_loader), total=len(data_loader),
                desc="Evaluation Progress", ncols=100, leave=True)
    last_fid_error = None

    for batch_idx, batch in pbar:
        try:
            memory_monitor.log_memory(f"Start of batch {batch_idx}")

            # Move data to device
            source = batch['source_self'].to(device)
            driving = batch['driving'].to(device)
            generated = batch['generated'].to(device)

            # Convert source_self, driving, generated to numpy arrays
            source_image = source.cpu().numpy()[0].transpose(1, 2, 0)  # CHW -> HWC
            driving_image = driving.cpu().numpy()[0].transpose(1, 2, 0)  # CHW -> HWC
            generated_image = generated.cpu().numpy()[0].transpose(1, 2, 0)  # CHW -> HWC

            # 调整大小到256x256（如果尚未调整）
            # source_image = resize(source_image, (256, 256))[..., :3]
            # driving_image = resize(driving_image, (256, 256))[..., :3]
            # generated_image = resize(generated_image, (256, 256))[..., :3]

            # 确保图像在[0,1]范围内
            source_image = np.clip(source_image, 0, 1)
            driving_image = np.clip(driving_image, 0, 1)
            generated_image = np.clip(generated_image, 0, 1)

            # 保存可视化（源图像、驱动图像、生成图像）
            if save_visuals:
                vis_path = os.path.join(vis_dir, f'comparison_batch_{batch_idx}.png')
                create_comparison_grid(source_image, driving_image, generated_image, vis_path)

            # 评估指标
            try:
                batch_metrics = evaluator.evaluate_batch(source_image, driving_image, generated_image)
                for metric, value in batch_metrics.items():
                    if isinstance(value, list):
                        accumulated_metrics[metric].extend(value)
                    else:
                        accumulated_metrics[metric].append(value)
            except Exception as metric_error:
                error_msg = str(metric_error)
                if "FID calculation error" in error_msg and error_msg != last_fid_error:
                    pbar.write(f"\nWarning: {error_msg}")
                    last_fid_error = error_msg

            # 内存清理
            del source, driving, generated, source_image, driving_image, generated_image
            torch.cuda.empty_cache()
            gc.collect()

            memory_monitor.log_memory(f"After cleanup - batch {batch_idx}")

            if (batch_idx + 1) % checkpoint_freq == 0:
                save_checkpoint(accumulated_metrics, output_path, batch_idx)

        except Exception as e:
            pbar.write(f"\nError in batch {batch_idx}: {str(e)}")
            memory_monitor.log_memory(f"ERROR in batch {batch_idx}: {str(e)}")
            raise

    pbar.close()
    return accumulated_metrics


def create_comparison_grid(source_img, driving_img, generated_img, save_path):
    """创建源图像、驱动图像和生成图像的对比网格"""
    import matplotlib.pyplot as plt
    from skimage import img_as_ubyte

    # 确保输入是numpy数组且值域正确
    def prepare_img(img):
        # 从[-1,1]范围转换到[0,1]范围（如果需要）
        img = (img + 1) / 2.0
        # 使用与demo.py相同的img_as_ubyte处理
        img = img_as_ubyte(img)
        # 转回[0,1]范围用于plt显示
        img = img.astype(np.float32) / 255.0
        return img

    source_img = prepare_img(source_img)
    driving_img = prepare_img(driving_img)
    generated_img = prepare_img(generated_img)

    # 创建图片网格
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Source Image', 'Driving Image', 'Generated Image']

    for j, (title, img) in enumerate(zip(titles, [source_img, driving_img, generated_img])):
        axes[j].imshow(img)
        axes[j].set_title(title)
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="path to evaluation pairs csv file with generated images")
    parser.add_argument("--prefix", required=True, help="prefix to add to image paths in CSV")
    parser.add_argument("--output_path", default="evaluation_results", help="path to save results")
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", required=True, help="path to checkpoint")
    parser.add_argument("--device", default="cuda", help="device to use")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for evaluation")
    parser.add_argument("--save_visuals", action='store_true', help="save visualization of source_self, driving and generated images")
    parser.add_argument("--checkpoint_frequency", type=int, default=5, help="save progress every N batches")
    parser.add_argument("--max_samples", type=int, default=None, help="maximum number of samples to evaluate")
    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")
    parser.set_defaults(relative=True)
    parser.set_defaults(adapt_scale=True)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    error_log = setup_error_logging(args.output_path)

    try:
        memory_monitor = MemoryMonitor(args.output_path)
        print("Loading models...")
        memory_monitor.log_memory("Before model loading")

        generator, kp_detector = load_checkpoints(args.config, args.checkpoint, kp_num=15, generator_type='DepthAwareGenerator', cpu=(args.device == 'cpu'))
        depth_encoder, depth_decoder = load_depth_model(cpu=(args.device == 'cpu'))

        memory_monitor.log_memory("After model loading")

        test_dataset = EvaluationDataset(
            dataroot=args.prefix,
            pairs_list=args.csv_path,
            max_samples=args.max_samples
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        print("Running evaluation...")
        evaluator = EvaluationMetrics(device=args.device)
        with torch.no_grad():
            metrics = evaluate_with_monitoring(
                generator, kp_detector, depth_encoder, depth_decoder,
                evaluator,
                test_loader, args.device, args.output_path,
                checkpoint_freq=args.checkpoint_frequency,
                save_visuals=args.save_visuals,
                relative=args.relative,
                adapt_movement_scale=args.adapt_scale
            )

        save_final_results(metrics, args.output_path)
        print(f"\nResults saved to {args.output_path}")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print(f"Check {error_log} for detailed error information")
        raise

if __name__ == "__main__":
    main()
