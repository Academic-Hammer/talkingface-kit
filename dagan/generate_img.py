import matplotlib

matplotlib.use('Agg')
import os
import sys
import yaml
import pandas as pd
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
    parser.add_argument("--csv", required=True, help="path to CSV file containing image pairs")
    parser.add_argument("--prefix", required=True, help="prefix to add to image paths")
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

    # 加载深度模型
    depth_encoder = depth.ResnetEncoder(18, False)
    depth_decoder = depth.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4))
    loaded_dict_enc = torch.load('depth/models/weights_19/encoder.pth', map_location='cpu' if opt.cpu else 'cuda')
    loaded_dict_dec = torch.load('depth/models/weights_19/depth.pth', map_location='cpu' if opt.cpu else 'cuda')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
    depth_encoder.load_state_dict(filtered_dict_enc)
    depth_decoder.load_state_dict(loaded_dict_dec)
    depth_encoder.eval()
    depth_decoder.eval()
    if not opt.cpu:
        depth_encoder.cuda()
        depth_decoder.cuda()

    # 加载生成器和关键点检测器
    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(opt.save_folder):
        os.makedirs(opt.save_folder)

    # 读取CSV文件
    df = pd.read_csv(opt.csv)

    # 初始化新的一列用于存储生成图像的路径
    generated_images = []

    # 遍历每一对源图像和驱动图像
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing image pairs"):
        source_rel = row['source']
        driving_rel = row['driving']

        # 添加前缀
        source_path = os.path.join(opt.prefix, source_rel)
        driving_path = os.path.join(opt.prefix, driving_rel)

        # 检查文件是否存在
        if not os.path.isfile(source_path):
            print(f"源图像不存在: {source_path}, 跳过此对")
            generated_images.append(None)
            continue
        if not os.path.isfile(driving_path):
            print(f"驱动图像不存在: {driving_path}, 跳过此对")
            generated_images.append(None)
            continue

        # 读取源图像和驱动图像
        source_image = imageio.imread(source_path)
        driving_image = imageio.imread(driving_path)

        # 调整大小到256x256
        source_image = resize(source_image, (256, 256))[..., :3]
        driving_image = resize(driving_image, (256, 256))[..., :3]

        try:
            # 生成动画图像（这里只生成一帧）
            predictions = make_animation(source_image, driving_image, generator, kp_detector,
                                         depth_encoder, depth_decoder,
                                         relative=opt.relative,
                                         adapt_movement_scale=opt.adapt_scale,
                                         cpu=opt.cpu)

            # 生成唯一的输出文件名，例如：generated_00001.png
            output_filename = f"generated_{index:05d}.png"
            output_path = os.path.join(opt.save_folder, output_filename)

            # 保存生成的图像
            imageio.imwrite(output_path, img_as_ubyte(predictions[0]))
            generated_images.append(output_path)
        except Exception as e:
            print(f"处理第{index}对图像时出错: {e}")
            generated_images.append(None)

    # 将生成图像的路径添加到DataFrame
    df['generated_image'] = generated_images

    # 保存更新后的CSV文件
    updated_csv_path = os.path.join(opt.save_folder, 'self_reenactment_pairs_with_generated.csv')
    df.to_csv(updated_csv_path, index=False)
    print(f"所有生成的图像路径已保存到 {updated_csv_path}")
