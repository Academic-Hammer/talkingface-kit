import cv2
# from moviepy import AudioFileClip
import moviepy

# 读取视频并分帧为图片，将视频的音频单独分离出来成为wav文件

video = cv2.VideoCapture("MP4/Obama1.mp4")
save_path = "JPG/Obama1"
index = 0
if video.isOpened():
    f = int(video.get(cv2.CAP_PROP_FPS))  # 读取视频帧率
    print("The video's fps is ", f)  # 显示视频帧率
    rval, frame = video.read()  # 读取视频帧
else:
    rval = False

while rval:
    print(index)
    rval, frame = video.read()
    if frame is None:
        break
    else:
        cv2.imwrite(save_path + "/" + str(index) + ".jpg", frame)
    index += 1

# my_audio_clip = AudioFileClip("MP4/Shaheen.mp4")
# my_audio_clip.write_audiofile("WAV/Shaheen.wav")
