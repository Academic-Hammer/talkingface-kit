import torchaudio
print(torchaudio.list_audio_backends())  # 查看可用的音频后端


import torchaudio
import torch
audio, sample_rate = torchaudio.load('./processed/Jae-in.mp4')
if torch.cuda.is_available():
    audio = audio.cuda()  # 将音频张量移动到 GPU
print(audio.device)  # 打印设备信息，应该显示为 cuda:0 或者 cuda:x
