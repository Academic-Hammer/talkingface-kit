import os
import cv2
from moviepy.editor import VideoFileClip
import argparse

def extract_audio(video_path, output_audio_path):
    """提取音频文件"""
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_audio_path)

def extract_and_crop_image(video_path, output_image_path, frame_number=100):
    """截取并裁剪肖像图片"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        height, width, _ = frame.shape
        center_y, center_x = height // 2, width // 2
        cropped_frame = frame[center_y - 128:center_y + 128, center_x - 128:center_x + 128]
        cv2.imwrite(output_image_path, cropped_frame)
    else:
        print(f"Error: Unable to read frame {frame_number} from {video_path}")
    cap.release()

def main(video_path, jpg_output, wav_output, amp_lip_x, amp_lip_y, amp_pos):
    examples_dir = os.path.dirname(jpg_output)
    if not os.path.exists(examples_dir):
        os.makedirs(examples_dir)

    print(f"Extracting audio to {wav_output}")
    extract_audio(video_path, wav_output)
    print(f"Extracting and cropping image to {jpg_output}")
    extract_and_crop_image(video_path, jpg_output)

    if not os.path.exists(jpg_output):
        print(f"Error: Image file {jpg_output} not created.")
        return

    print(f"Successfully created:")
    print(f"  Image file: {jpg_output}")
    print(f"  Audio file: {wav_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从视频中提取音频和肖像图片，并运行 main_end2end.py 脚本")
    parser.add_argument('--video', type=str, required=True, help="输入视频文件路径")
    parser.add_argument('--jpg_output', type=str, default="examples/output.jpg", help="输出肖像图片文件路径")
    parser.add_argument('--wav_output', type=str, default="examples/output.wav", help="输出音频文件路径")
    parser.add_argument('--amp_lip_x', type=float, default=2.0, help="嘴唇运动的 X 轴放大系数")
    parser.add_argument('--amp_lip_y', type=float, default=2.0, help="嘴唇运动的 Y 轴放大系数")
    parser.add_argument('--amp_pos', type=float, default=0.5, help="头部运动位移的放大系数")
    args = parser.parse_args()

    main(args.video, args.jpg_output, args.wav_output, args.amp_lip_x, args.amp_lip_y, args.amp_pos)
