import argparse
import torch
import yaml
from pathlib import Path
from diffusion import Diffusion
from utils import get_id_frame, get_audio_emb, save_video


def update_yaml_config(config_file, input_image, input_audio, output_dir):
    # 如果config文件不存在，创建一个新的字典
    if not Path(config_file).exists():
        config = {}
    else:
        # 加载已有的配置
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f) or {}

    # 更新配置文件
    config['id_frame'] = input_image
    config['audio'] = input_audio
    config['output'] = output_dir

    # 将修改后的配置写回到文件
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def main():
    parser = argparse.ArgumentParser(description="Diffusion-based video generation.")
    parser.add_argument('--input_image', type=str, required=True, help='Path to input image')
    parser.add_argument('--input_audio_text', type=str, required=True, help='Path to input audio')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output video')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--encoder_checkpoint', type=str, required=True, help='Path to encoder checkpoint')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

    # 解析命令行参数
    args = parser.parse_args()

    # 更新 config_crema.yaml 文件
    update_yaml_config('config_crema.yaml', args.input_image, args.input_audio_text, args.output_dir)

    # 设置设备
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

    print('Loading model...')
    unet = torch.jit.load(args.checkpoint)
    diffusion = Diffusion(unet, device).to(device)

    # 加载图像和音频
    id_frame = get_id_frame(args.input_image).to(device)
    audio, audio_emb = get_audio_emb(args.input_audio_text, args.encoder_checkpoint, device)

    # 进行采样
    samples = diffusion.sample(id_frame, audio_emb.unsqueeze(0))

    # 保存视频
    save_video(args.output_dir, samples, audio=audio, fps=25, audio_rate=16000)
    print(f'Results saved at {args.output_dir}')


if __name__ == '__main__':
    main()
