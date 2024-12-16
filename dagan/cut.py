import cv2
import numpy as np


def crop_video(input_path, output_path, target_size=(256, 256)):
    # 读取视频
    cap = cv2.VideoCapture(input_path)

    # 获取原视频的属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算裁剪的起始位置（居中裁剪）
    start_x = (frame_width - target_size[0]) // 2
    start_y = (frame_height - target_size[1]) // 2

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)

    # 处理每一帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 裁剪帧
        cropped_frame = frame[start_y:start_y + target_size[1],
                        start_x:start_x + target_size[0]]

        # 如果裁剪后的尺寸不正确，则调整大小
        if cropped_frame.shape[:2] != target_size:
            cropped_frame = cv2.resize(cropped_frame, target_size)

        # 写入帧
        out.write(cropped_frame)

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# 使用示例
if __name__ == "__main__":
    input_video = "/root/autodl-tmp/videos/Jae-in.mp4"  # 输入视频路径
    output_video = "assets/Jae.mp4"  # 输出视频路径
    crop_video(input_video, output_video)