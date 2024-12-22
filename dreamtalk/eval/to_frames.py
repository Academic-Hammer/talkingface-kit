import cv2
import os
import random

def extract_frames(video_path, output_folder):
    """
    提取视频中的所有帧
    """
    video_capture = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not video_capture.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    # 获取视频文件名（不带扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 为每个视频创建一个独立的输出文件夹
    video_output_folder = os.path.join(output_folder, video_name)
    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)
    
    frame_count = 0
    while True:
        # 逐帧读取视频
        ret, frame = video_capture.read()
        
        # 如果读取失败，退出循环
        if not ret:
            break
        
        
        # 构建帧的输出文件路径
        frame_filename = os.path.join(video_output_folder, f"frame_{frame_count:04d}.png")
        
        # 保存帧
        cv2.imwrite(frame_filename, frame)
        
        print(f"Saved: {frame_filename}")
        
        frame_count += 1
    
    video_capture.release()
    print(f"Finished extracting frames from {video_path}.")

def extract_first_frame(video_path, output_folder):
    """
    只提取视频的第一帧
    """
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    video_output_folder = os.path.join(output_folder, video_name + "_first")
    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)
    
    # 读取第一帧
    ret, frame = video_capture.read()
    
    if ret:
        
        frame_filename = os.path.join(video_output_folder, f"frame_0000.png")
        
        cv2.imwrite(frame_filename, frame)
        
        print(f"Saved: {frame_filename}")
    
    video_capture.release()
    print(f"Finished extracting the first frame from {video_path}.")

def extract_random_frames(video_path, output_folder, num_frames=10):
    """
    从视频中随机选择指定数量的帧并保存为 256x256 分辨率的图片。
    """
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    video_output_folder = os.path.join(output_folder, video_name + "_random_" + str(num_frames))
    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)
    
    # 获取视频的总帧数
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 如果请求的帧数大于总帧数，调整为总帧数
    num_frames = min(num_frames, total_frames)
    
    # 随机选择不重复的帧索引
    frame_indices = random.sample(range(total_frames), num_frames)
    
    # 提取指定的随机帧
    for i, frame_index in enumerate(frame_indices):
        # 设置视频捕捉对象的位置
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        # 读取指定帧
        ret, frame = video_capture.read()
        
        if ret:
            frame_filename = os.path.join(video_output_folder, f"frame_{frame_index:04d}.png")
            
            cv2.imwrite(frame_filename, frame)
            
            print(f"Saved: {frame_filename}")
    
    video_capture.release()
    print(f"Finished extracting {num_frames} random frames from {video_path}.")

def process_videos(input_folder, output_folder, mode=1, num_random_frames=10):
    """
    遍历输入文件夹中的所有视频文件，按指定模式提取视频帧。
    - mode=1: 只提取第一帧
    - mode=2: 提取所有帧
    - mode=3: 随机提取指定数量的帧
    """
    # 遍历输入文件夹中的所有文件
    for file_name in os.listdir(input_folder):
        # 只获取 256 x 256 分辨率的视频
        if '_256' in file_name:
            video_path = os.path.join(input_folder, file_name)
        
            if os.path.isfile(video_path) and file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                print(f"Processing video: {file_name}")
                
                if mode == 1:
                    # 只提取第一帧
                    extract_first_frame(video_path, output_folder)
                elif mode == 2:
                    # 提取所有帧
                    extract_frames(video_path, output_folder)
                elif mode == 3:
                    # 随机提取指定数量的帧
                    extract_random_frames(video_path, output_folder, num_random_frames)
                else:
                    print(f"Invalid mode: {mode}. Please use mode 1, 2, or 3.")
    
    print("Finished processing all videos.")


input_folder = "ref"
output_folder = "output"

# 只提取每个视频的第一帧
# process_videos(input_folder, output_folder, mode=1)

# 提取所有帧, 太大了
# process_videos(input_folder, output_folder, mode=2)

# 从每个视频中随机提取 x 帧，25 的帧速率的话 1500 刚好 60s 
# process_videos(input_folder, output_folder, mode=3, num_random_frames=1500)