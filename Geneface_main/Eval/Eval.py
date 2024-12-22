import cv2
import numpy as np
from skimage.util import img_as_float

# Function to extract frames from a video
def extract_frames(video_path):
    """Extract frames from a video and convert to grayscale."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return frames

# Resize frame to match target dimensions
def resize_frame(frame, target_shape):
    """Resize a frame to match the target dimensions."""
    return cv2.resize(frame, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

# Placeholder NIQE calculation function
def calculate_niqe(frame):
    """Calculate NIQE score for a frame (placeholder implementation)."""
    return np.random.uniform(4, 10)  # Replace with actual NIQE implementation

# Calculate PSNR for two frames
def calculate_psnr(frame1, frame2):
    """Calculate PSNR between two frames."""
    mse = np.mean((frame1 - frame2) ** 2)
    if mse == 0:
        return float('inf')
    data_range = frame1.max() - frame1.min()
    result = (data_range ** 2) / mse
    psnr = np.empty_like(result)
    np.log10(result, out=psnr, where=result > 0)
    return 10 * psnr

# Main function to calculate metrics
def calculate_metrics(video1_path, video2_path):
    """Calculate average PSNR and NIQE metrics for two videos."""
    frames1 = extract_frames(video1_path)
    frames2 = extract_frames(video2_path)

    frame_count = min(len(frames1), len(frames2))
    if len(frames1) != len(frames2):
        print("Warning: Videos have different number of frames. Metrics will be calculated up to the shorter one.")

    psnr_values = []
    niqe_values = []

    for i in range(frame_count):
        frame1 = img_as_float(frames1[i])
        frame2 = img_as_float(frames2[i])

        # Resize frames to the same dimensions if necessary
        if frame1.shape != frame2.shape:
            frame2 = resize_frame(frame2, frame1.shape)

        # Calculate PSNR
        psnr_values.append(calculate_psnr(frame1, frame2))

        # Calculate NIQE for the second frame
        niqe_values.append(calculate_niqe(frame2))

    avg_psnr = np.mean(psnr_values)
    avg_niqe = np.mean(niqe_values)

    return avg_psnr, avg_niqe

# Paths to videos
video1_path = "May_org.mp4"
video2_path = "May_radnerf_torso_smo.mp4"

# Calculate metrics
if __name__ == "__main__":
    psnr, niqe = calculate_metrics(video1_path, video2_path)
    print(f"Average PSNR: {psnr:.2f}")
    print(f"Average NIQE: {niqe:.2f}")

