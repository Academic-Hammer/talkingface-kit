
import pandas as pd
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')
import os
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import modules.generator as GEN
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback
import depth
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from collections import OrderedDict


def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
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


def load_image_sequence(folder_path):
    # Get all PNG files in the folder
    image_files = sorted(glob.glob(os.path.join(folder_path, '*.png')))
    if not image_files:
        raise ValueError(f"No PNG files found in {folder_path}")

    # Read all images
    images = []
    for img_path in image_files:
        img = imageio.imread(img_path)
        img = resize(img, (256, 256))[..., :3]
        images.append(img)

    return images


def make_animation(source_image, driving_video, generator, kp_detector, relative=True,
                   adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        if not cpu:
            source = source.cuda()
            driving = driving.cuda()

        outputs = depth_decoder(depth_encoder(source))
        depth_source = outputs[("disp", 0)]

        outputs = depth_decoder(depth_encoder(driving[:, :, 0]))
        depth_driving = outputs[("disp", 0)]

        source_kp = torch.cat((source, depth_source), 1)
        driving_kp = torch.cat((driving[:, :, 0], depth_driving), 1)

        kp_source = kp_detector(source_kp)
        kp_driving_initial = kp_detector(driving_kp)

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]

            if not cpu:
                driving_frame = driving_frame.cuda()

            outputs = depth_decoder(depth_encoder(driving_frame))
            depth_map = outputs[("disp", 0)]

            frame = torch.cat((driving_frame, depth_map), 1)
            kp_driving = kp_detector(frame)

            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial,
                                   use_relative_movement=relative,
                                   use_relative_jacobian=relative,
                                   adapt_movement_scale=adapt_movement_scale)

            out = generator(source, kp_source=kp_source, kp_driving=kp_norm,
                            source_depth=depth_source, driving_depth=depth_map)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(),
                                            [0, 2, 3, 1])[0])

    return predictions


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint")
    parser.add_argument("--csv_path", required=True, help="path to input CSV file")
    parser.add_argument("--output_dir", default='animate', help="output directory")
    parser.add_argument("--relative", dest="relative", action="store_true")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true")
    parser.add_argument("--generator", type=str, required=True)
    parser.add_argument("--kp_num", type=int, required=True)
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode")

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(opt.output_dir, exist_ok=True)

    # Initialize depth models
    depth_encoder = depth.ResnetEncoder(18, False)
    depth_decoder = depth.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4))

    # Load depth model weights
    loaded_dict_enc = torch.load('depth/models/weights_19/encoder.pth')
    loaded_dict_dec = torch.load('depth/models/weights_19/depth.pth')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items()
                         if k in depth_encoder.state_dict()}
    depth_encoder.load_state_dict(filtered_dict_enc)
    depth_decoder.load_state_dict(loaded_dict_dec)

    depth_encoder.eval()
    depth_decoder.eval()

    if not opt.cpu:
        depth_encoder.cuda()
        depth_decoder.cuda()

    # Load generator and keypoint detector
    generator, kp_detector = load_checkpoints(config_path=opt.config,
                                              checkpoint_path=opt.checkpoint,
                                              cpu=opt.cpu)

    # Read CSV file
    df = pd.read_csv(opt.csv_path)

    # Process each row in the CSV
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing pairs"):
        try:
            # Generate output filename
            source_id = os.path.basename(os.path.dirname(row['source_frame']))
            driving_id = os.path.basename(row['driving_video'])
            output_filename = f"{source_id}_to_{driving_id}.mp4"
            output_path = os.path.join(opt.output_dir, output_filename)

            # Skip if output already exists
            if os.path.exists(output_path):
                print(f"Skipping existing file: {output_path}")
                continue

            # Load source image
            source_image = imageio.imread(row['source_frame'])
            source_image = resize(source_image, (256, 256))[..., :3]

            # Load driving video frames
            driving_frames = load_image_sequence(row['driving_video'])

            # Generate animation
            predictions = make_animation(source_image, driving_frames, generator, kp_detector,
                                         relative=opt.relative, adapt_movement_scale=opt.adapt_scale,
                                         cpu=opt.cpu)

            # Save video
            imageio.mimsave(output_path, [img_as_ubyte(p) for p in predictions], fps=25)
            print(f"Saved animation to: {output_path}")

        except Exception as e:
            print(f"Error processing pair {idx}: {str(e)}")
            continue