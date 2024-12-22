import matplotlib

matplotlib.use('Agg')
import os
import sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import modules.generator as GEN
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback
import depth
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from collections import OrderedDict
import time


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if opt.kp_num != -1:
        config['model_params']['common_params']['num_kp'] = opt.kp_num
    generator = getattr(GEN, opt.generator)(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()
    config['model_params']['common_params']['num_channels'] = 4
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cuda:0")

    ckp_generator = OrderedDict((k.replace('module.', ''), v) for k, v in checkpoint['generator'].items())
    generator.load_state_dict(ckp_generator)
    ckp_kp_detector = OrderedDict((k.replace('module.', ''), v) for k, v in checkpoint['kp_detector'].items())
    kp_detector.load_state_dict(ckp_kp_detector)

    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector


def make_animation(source_image, driving_image, generator, kp_detector, depth_encoder, depth_decoder,
                   relative=True, adapt_movement_scale=True, cpu=False):
    predictions = []
    with torch.no_grad():
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        driving = torch.tensor(driving_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

        if not cpu:
            source = source.cuda()
            driving = driving.cuda()

        # Depth estimation (for both source and driving)
        outputs = depth_decoder(depth_encoder(source))
        depth_source = outputs[("disp", 0)]

        outputs = depth_decoder(depth_encoder(driving))
        depth_driving = outputs[("disp", 0)]

        source_kp = torch.cat((source, depth_source), 1)
        driving_kp = torch.cat((driving, depth_driving), 1)

        kp_source = kp_detector(source_kp)
        kp_driving_initial = kp_detector(driving_kp)

        kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving_initial,
                               kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                               use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
        out = generator(source, kp_source=kp_source, kp_driving=kp_norm, source_depth=depth_source,
                        driving_depth=depth_driving)

        predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

    return predictions


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")
    parser.add_argument("--source_dir", required=True, help="directory containing source images")
    parser.add_argument("--driving_dir", required=True, help="directory containing driving images")
    parser.add_argument("--save_folder", default='results/', help="path to output folder")
    parser.add_argument("--relative", dest="relative", action="store_true",
                        help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true",
                        help="adapt movement scale based on convex hull of keypoints")
    parser.add_argument("--generator", type=str, required=True)
    parser.add_argument("--kp_num", type=int, required=True)
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()

    print("开始加载模型...")
    with tqdm(total=4, desc="模型初始化", dynamic_ncols=True, leave=False) as pbar:
        # 加载深度模型
        depth_encoder = depth.ResnetEncoder(18, False)
        depth_decoder = depth.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4))
        pbar.update(1)

        # 加载编码器权重
        loaded_dict_enc = torch.load('depth/models/weights_19/encoder.pth',
                                     map_location='cpu' if opt.cpu else 'cuda')
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items()
                             if k in depth_encoder.state_dict()}
        depth_encoder.load_state_dict(filtered_dict_enc)
        depth_encoder.eval()
        pbar.update(1)

        # 加载解码器权重
        loaded_dict_dec = torch.load('depth/models/weights_19/depth.pth',
                                     map_location='cpu' if opt.cpu else 'cuda')
        depth_decoder.load_state_dict(loaded_dict_dec)
        depth_decoder.eval()

        if not opt.cpu:
            depth_encoder.cuda()
            depth_decoder.cuda()
        pbar.update(1)

        # 加载生成器和关键点检测器
        generator, kp_detector = load_checkpoints(config_path=opt.config,
                                                  checkpoint_path=opt.checkpoint,
                                                  cpu=opt.cpu)
        pbar.update(1)

    if not os.path.exists(opt.save_folder):
        os.makedirs(opt.save_folder)
        print(f"创建输出目录: {opt.save_folder}")

    # 获取源目录中的文件列表
    source_files = sorted([f for f in os.listdir(opt.source_dir)
                           if f.endswith(('.png', '.jpg', '.jpeg'))])
    total_files = len(source_files)

    print(f"\n开始处理图像，共找到 {total_files} 个文件...")

    # 处理统计
    success_count = 0
    failed_count = 0
    start_time = time.time()

    # 显示处理配置
    print("\n处理配置：")
    print(f"- 使用相对坐标: {'是' if opt.relative else '否'}")
    print(f"- 自适应缩放: {'是' if opt.adapt_scale else '否'}")
    print(f"- 运行模式: {'CPU' if opt.cpu else 'CUDA'}")
    print(f"- 关键点数量: {opt.kp_num}")
    print("-" * 50)

    # 主处理循环
    with tqdm(total=total_files, desc="处理进度", dynamic_ncols=True, leave=False) as pbar:
        for idx, filename in enumerate(source_files, 1):
            source_path = os.path.join(opt.source_dir, filename)
            driving_path = os.path.join(opt.driving_dir, filename)

            # 显示当前处理的文件信息
            pbar.set_description(f"处理: {filename}")

            if not os.path.isfile(driving_path):
                failed_count += 1
                pbar.write(f"[错误] 驱动图像不存在: {driving_path}")
                pbar.update(1)
                continue

            try:
                process_start = time.time()

                # 读取图像
                source_image = imageio.imread(source_path)
                driving_image = imageio.imread(driving_path)

                # 调整大小
                source_image = resize(source_image, (256, 256))[..., :3]
                driving_image = resize(driving_image, (256, 256))[..., :3]

                # 生成动画
                predictions = make_animation(source_image, driving_image,
                                             generator, kp_detector,
                                             depth_encoder, depth_decoder,
                                             relative=opt.relative,
                                             adapt_movement_scale=opt.adapt_scale,
                                             cpu=opt.cpu)

                # 保存生成的图像
                output_path = os.path.join(opt.save_folder, filename)
                imageio.imwrite(output_path, img_as_ubyte(predictions[0]))

                success_count += 1
                process_time = time.time() - process_start
                avg_time = (time.time() - start_time) / idx
                eta = avg_time * (total_files - idx)

                # 更新进度条信息
                pbar.set_postfix({
                    '当前耗时': f'{process_time:.1f}s'
                })

            except Exception as e:
                failed_count += 1
                pbar.write(f"[错误] 处理 {filename} 失败: {str(e)}")

            pbar.update(1)

    # 最终统计
    total_time = time.time() - start_time
    success_rate = (success_count / total_files) * 100

    print("\n" + "=" * 50)
    print("处理完成！最终统计：")
    print("=" * 50)
    print(f"总文件数: {total_files}")
    print(f"成功处理: {success_count} ({success_rate:.1f}%)")
    print(f"处理失败: {failed_count}")
    print(f"总耗时: {total_time:.1f} 秒")
    print(f"平均每张耗时: {total_time / total_files:.1f} 秒")
    print(f"生成的图像已保存至: {os.path.abspath(opt.save_folder)}")

    if failed_count > 0:
        print(f"\n警告：有 {failed_count} 个文件处理失败，请检查日志信息。")
    else:
        print("\n所有文件处理成功！")