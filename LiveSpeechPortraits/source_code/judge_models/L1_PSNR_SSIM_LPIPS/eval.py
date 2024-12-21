import cv2
import os
import numpy as np
import torch
import lpips
from tqdm import tqdm  # Progress bar
import math
from torchmetrics.functional import structural_similarity_index_measure as torch_ssim

# Helper function: get total frame count and duration of a video
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return frame_count, duration

# Helper function: get frame indices to extract based on target FPS
def get_frame_indices(original_fps, target_fps, total_frames):
    if target_fps >= original_fps:
        return list(range(total_frames))
    frame_interval = original_fps / target_fps
    indices = [int(i * frame_interval) for i in range(int(target_fps * (total_frames / original_fps)))]
    return indices

# Helper function: compute L1 Loss
def compute_l1_loss(gt_frame, gen_frame):
    return np.mean(np.abs(gt_frame - gen_frame))

# Helper function: compute PSNR
def compute_psnr(gt_frame, gen_frame):
    mse = np.mean((gt_frame - gen_frame) ** 2)
    return 20 * np.log10(255.0 / math.sqrt(mse)) if mse > 0 else float('inf')

# Helper function: compute SSIM using torchmetrics
def compute_ssim_metric(gt_tensor, gen_tensor, ssim_fn):
    ssim_value = ssim_fn(gt_tensor, gen_tensor)
    return ssim_value.item()

# Helper function: compute LPIPS
def compute_lpips_metric(gt_tensor, gen_tensor, loss_fn):
    return loss_fn(gt_tensor, gen_tensor).item()

# Helper function: compute all metrics for a frame pair
def compute_metrics(gt_frame, gen_frame, loss_fn, ssim_fn, device):
    metrics = {}

    # L1 Loss
    metrics["L1"] = compute_l1_loss(gt_frame, gen_frame)

    # PSNR
    metrics["PSNR"] = compute_psnr(gt_frame, gen_frame)

    # Convert frames to tensors
    gt_tensor = torch.tensor(gt_frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    gen_tensor = torch.tensor(gen_frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    # SSIM
    metrics["SSIM"] = compute_ssim_metric(gt_tensor, gen_tensor, ssim_fn)

    # LPIPS
    metrics["LPIPS"] = compute_lpips_metric(gt_tensor, gen_tensor, loss_fn)

    return metrics

def compute_all_metrics(gt_video, gen_video, target_fps=30, debug=False):
    # Step 1: Get total frame count and duration of videos
    print(f"Loading ground truth video: {gt_video}")
    print(f"Loading generated video: {gen_video}")
    gt_frame_count, gt_duration = get_video_info(gt_video)
    gen_frame_count, gen_duration = get_video_info(gen_video)

    print(f"Ground truth video: {gt_frame_count} frames, {gt_duration:.2f} seconds")
    print(f"Generated video: {gen_frame_count} frames, {gen_duration:.2f} seconds")

    # Step 2: Determine target frame count based on FPS and duration
    target_frame_count = min(int(target_fps * gt_duration), int(target_fps * gen_duration))
    print(f"Target frame count based on {target_fps} FPS: {target_frame_count}")

    # Step 3: Calculate frame indices to extract for both videos
    cap_gt = cv2.VideoCapture(gt_video)
    cap_gen = cv2.VideoCapture(gen_video)

    original_fps_gt = cap_gt.get(cv2.CAP_PROP_FPS)
    original_fps_gen = cap_gen.get(cv2.CAP_PROP_FPS)

    frame_indices_gt = get_frame_indices(original_fps_gt, target_fps, gt_frame_count)
    frame_indices_gen = get_frame_indices(original_fps_gen, target_fps, gen_frame_count)

    # Ensure both frame lists have the same number of frames
    min_frames = min(len(frame_indices_gt), len(frame_indices_gen), target_frame_count)
    frame_indices_gt = frame_indices_gt[:min_frames]
    frame_indices_gen = frame_indices_gen[:min_frames]

    print(f"Number of frames to process: {min_frames}")

    # Step 4: Initialize LPIPS and SSIM once
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    loss_fn = lpips.LPIPS(net='alex').to(device)
    ssim_fn = torch_ssim

    # Step 5: Read and process frames
    metrics_list = []

    # Preload frame indices into sets for faster access
    frame_set_gt = set(frame_indices_gt)
    frame_set_gen = set(frame_indices_gen)

    current_frame_gt = 0
    current_frame_gen = 0
    target_idx = 0

    frames_gt = []
    print("Reading ground truth frames into memory...")
    for desired_frame in tqdm(frame_indices_gt, desc="Loading GT frames"):
        while current_frame_gt <= desired_frame:
            ret, frame = cap_gt.read()
            if not ret:
                print(f"Warning: Reached end of ground truth video at frame {current_frame_gt}.")
                frames_gt.append(None)
                break
            if current_frame_gt == desired_frame:
                frames_gt.append(frame)
                break
            current_frame_gt += 1

    # Reset for generated video
    target_idx = 0
    current_frame_gen = 0
    frames_gen = []
    print("Reading generated video frames into memory...")
    for desired_frame in tqdm(frame_indices_gen, desc="Loading Gen frames"):
        while current_frame_gen <= desired_frame:
            ret, frame = cap_gen.read()
            if not ret:
                # print(f"Warning: Reached end of generated video at frame {current_frame_gen}.")
                frames_gen.append(None)
                break
            if current_frame_gen == desired_frame:
                frames_gen.append(frame)
                break
            current_frame_gen += 1

    cap_gt.release()
    cap_gen.release()

    print(f"Total frames loaded: {len(frames_gt)}")

    # Step 6: Compute metrics for each frame pair
    for i in tqdm(range(len(frames_gt)), desc="Computing metrics"):
        gt_frame = frames_gt[i]
        gen_frame = frames_gen[i]

        if gt_frame is None or gen_frame is None:
            # print(f"Skipping frame {i} due to read error.")
            continue

        metrics = compute_metrics(gt_frame, gen_frame, loss_fn, ssim_fn, device)
        metrics_list.append(metrics)

    # Step 7: Calculate and print average metrics
    metrics_array = np.array([list(m.values()) for m in metrics_list])
    metrics_names = list(metrics_list[0].keys())
    average_metrics = np.mean(metrics_array, axis=0)
    print("\n=== Average Metrics ===")
    for name, value in zip(metrics_names, average_metrics):
        print(f"{name}: {value:.4f}")

    return average_metrics

# Example usage
if __name__ == "__main__":
    output_dir = "output_metrics"
    gt_video = "E:\\Code\\DesktopCode\\LiveSpeechPortraits\\data\\Input\\May_short.mp4"
    gen_video = "E:\\Code\\DesktopCode\\LiveSpeechPortraits\\results\\May\\May_short\\May_short.avi"
    lists = compute_all_metrics(gt_video, gen_video)
    print(lists)
