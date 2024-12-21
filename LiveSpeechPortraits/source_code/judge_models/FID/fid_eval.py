import cv2
import os
import torch
import shutil
from pytorch_fid import fid_score  # 新增引入pytorch-fid

# 获取视频信息：帧数与持续时间
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return frame_count, duration

# 根据目标FPS进行视频帧率调整
def adjust_fps(input_video, output_video, target_fps, target_frame_count=None):
    print(f"Adjusting FPS for {input_video} to {target_fps} FPS...")
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise Exception(f"Cannot open video: {input_video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(output_video, fourcc, target_fps, (width, height))
    frame_interval = int(original_fps / target_fps) if target_fps < original_fps else 1

    frame_count = 0
    extracted_frame_count = 0
    last_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Failed to read frame {frame_count}. Using last valid frame.")
            if last_frame is not None and (target_frame_count is None or extracted_frame_count < target_frame_count):
                out.write(last_frame)
                extracted_frame_count += 1
            if target_frame_count is not None and extracted_frame_count >= target_frame_count:
                break
            frame_count += 1
            continue

        if frame_count % frame_interval == 0:
            out.write(frame)
            extracted_frame_count += 1
            last_frame = frame

        if target_frame_count is not None and extracted_frame_count >= target_frame_count:
            break

        frame_count += 1

    # 如果需要，使用最后一帧填补
    while target_frame_count is not None and extracted_frame_count < target_frame_count:
        if last_frame is not None:
            out.write(last_frame)
            extracted_frame_count += 1
        else:
            print("Error: No valid frames available to pad the output.")

    cap.release()
    out.release()
    print(f"FPS adjustment completed: {input_video} -> {output_video}. Extracted {extracted_frame_count} frames.")
    return extracted_frame_count

# 将视频转为帧
def video_to_frames(video_path, output_dir, target_frame_count=None):
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"Frames already exist in {output_dir}, skipping extraction.")
        return len(os.listdir(output_dir))

    print(f"Extracting frames from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open video: {video_path}")
    os.makedirs(output_dir, exist_ok=True)
    frame_idx = 0
    extracted_frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(frame_path, frame)
        extracted_frame_count += 1
        frame_idx += 1
        if target_frame_count is not None and extracted_frame_count >= target_frame_count:
            break
    cap.release()
    print(f"Frame extraction completed. Total frames: {extracted_frame_count}")
    return extracted_frame_count

# 这里原本有InceptionV3FeatureExtractor和preprocess_image函数，现在不再需要。

# 使用pytorch-fid计算FID
def compute_fid(gt_frames_dir, gen_frames_dir):
    # 使用pytorch_fid提供的calculate_fid_given_paths计算FID
    fid = fid_score.calculate_fid_given_paths(
        [gt_frames_dir, gen_frames_dir],
        batch_size=50,  # 可根据情况调整
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        dims=2048       # 与Inception V3特征维度一致
    )
    return fid

# 主流程，只保留FID计算
def compute_fid_for_videos(gt_video, gen_video, output_dir="./fid_evaluation_output", target_fps=30):
    os.makedirs(output_dir, exist_ok=True)
    # 获取视频信息
    gt_frame_count, gt_duration = get_video_info(gt_video)
    gen_frame_count, gen_duration = get_video_info(gen_video)

    target_frame_count = min(gt_frame_count, gen_frame_count)
    target_fps_gt = target_frame_count / gt_duration if gt_duration > 0 else target_fps
    target_fps_gen = target_frame_count / gen_duration if gen_duration > 0 else target_fps

    # 调整视频帧数和FPS
    adjusted_gt_video = os.path.join(output_dir, "adjusted_gt_video.avi")
    gt_frame_count = adjust_fps(gt_video, adjusted_gt_video, target_fps_gt, target_frame_count)

    adjusted_gen_video = os.path.join(output_dir, "adjusted_gen_video.avi")
    gen_frame_count = adjust_fps(gen_video, adjusted_gen_video, target_fps_gen, target_frame_count)

    # 提取帧
    gt_frames_dir = os.path.join(output_dir, "ground_truth")
    gt_frame_count = video_to_frames(adjusted_gt_video, gt_frames_dir, target_frame_count)

    gen_frames_dir = os.path.join(output_dir, "generated")
    gen_frame_count = video_to_frames(adjusted_gen_video, gen_frames_dir, target_frame_count)

    # 使用pytorch_fid计算FID
    fid = compute_fid(gt_frames_dir, gen_frames_dir)
    shutil.rmtree(output_dir)
    return fid

if __name__ == "__main__":
    gt_video_path = "E:\\Code\\DesktopCode\\LiveSpeechPortraits\\data\\Input\\May_short.mp4"
    gen_video_path = "E:\\Code\\DesktopCode\\LiveSpeechPortraits\\results\\May\\May_short\\May_short.avi"

    fid_value = compute_fid_for_videos(gt_video_path, gen_video_path)
    print(f"FID: {fid_value}")
