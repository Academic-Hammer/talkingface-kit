import matplotlib

matplotlib.use('Agg')
import os, sys
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
from scipy.spatial import ConvexHull
from collections import OrderedDict
import pdb
import cv2
from glob import glob

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


def make_animation(source_image, driving_image, generator, kp_detector, relative=True, adapt_movement_scale=True,
                   cpu=False):
    sources = []
    drivings = []
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
    parser.add_argument("--source_image", required=True, help="path to source image")
    parser.add_argument("--driving_image", required=True, help="path to driving image")
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

    # Load depth models
    depth_encoder = depth.ResnetEncoder(18, False)
    depth_decoder = depth.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4))
    loaded_dict_enc = torch.load('depth/models/weights_19/encoder.pth')
    loaded_dict_dec = torch.load('depth/models/weights_19/depth.pth')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
    depth_encoder.load_state_dict(filtered_dict_enc)
    depth_decoder.load_state_dict(loaded_dict_dec)
    depth_encoder.eval()
    depth_decoder.eval()
    if not opt.cpu:
        depth_encoder.cuda()
        depth_decoder.cuda()

    # Read source and driving images
    source_image = imageio.imread(opt.source_image)
    driving_image = imageio.imread(opt.driving_image)

    # Resize to 256x256
    source_image = resize(source_image, (256, 256))[..., :3]
    driving_image = resize(driving_image, (256, 256))[..., :3]

    # Load generator and keypoint detector
    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)

    # Create output folder if doesn't exist
    if not os.path.exists(opt.save_folder):
        os.makedirs(opt.save_folder)

    # Make animation (generate a single output image)
    predictions = make_animation(source_image, driving_image, generator, kp_detector, relative=opt.relative,
                                 adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)

    # Save the generated image
    output_path = os.path.join(opt.save_folder, 'generated_image.png')
    imageio.imwrite(output_path, img_as_ubyte(predictions[0]))  # Save the first (and only) image
    print(f"Generated image saved to {output_path}")
