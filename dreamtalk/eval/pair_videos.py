import cv2
import os
import random

def extract_random_frames_pair(video_paths1, video_paths2, output_folder, num_frames=10):
    """
    从两组视频中随机选择指定数量的帧并保存为 256x256 分辨率的图片，
    确保两组视频提取的随机帧在时间序号上是相同的。
    """
    # 确保两组视频路径长度相同
    if len(video_paths1) != len(video_paths2):
        print("Error: The two video groups must have the same length.")
        return

    # 获取随机选择的帧索引
    total_frames = min(
        int(cv2.VideoCapture(video_paths1[0]).get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cv2.VideoCapture(video_paths2[0]).get(cv2.CAP_PROP_FRAME_COUNT))
    )

    # 如果请求的帧数大于总帧数，调整为总帧数
    num_frames = min(num_frames, total_frames)
    
    # 随机选择不重复的帧索引
    frame_indices = random.sample(range(total_frames), num_frames)

    # 遍历视频路径组，处理每个视频
    for idx, (video_path1, video_path2) in enumerate(zip(video_paths1, video_paths2)):
        # 获取视频名称并创建输出文件夹
        video_name1 = os.path.splitext(os.path.basename(video_path1))[0]
        video_name2 = os.path.splitext(os.path.basename(video_path2))[0]
        
        video_output_folder1 = os.path.join(output_folder, video_name1 + "_random_" + str(num_frames))
        video_output_folder2 = os.path.join(output_folder, video_name2 + "_random_" + str(num_frames))
        
        if not os.path.exists(video_output_folder1):
            os.makedirs(video_output_folder1)
        if not os.path.exists(video_output_folder2):
            os.makedirs(video_output_folder2)

        video_capture1 = cv2.VideoCapture(video_path1)
        video_capture2 = cv2.VideoCapture(video_path2)
        
        # 逐帧处理并保存
        for i, frame_index in enumerate(frame_indices):
            # 设置视频捕捉对象的位置
            video_capture1.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            video_capture2.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            
            # 读取指定帧
            ret1, frame1 = video_capture1.read()
            ret2, frame2 = video_capture2.read()
            
            if ret1 and ret2:
                # 构建帧的输出文件路径
                frame_filename1 = os.path.join(video_output_folder1, f"frame_{frame_index:04d}_ref.png")
                frame_filename2 = os.path.join(video_output_folder2, f"frame_{frame_index:04d}_eval.png")
                
                # 保存帧为图片
                cv2.imwrite(frame_filename1, frame1)
                cv2.imwrite(frame_filename2, frame2)
                
                print(f"Saved: {frame_filename1}")
                print(f"Saved: {frame_filename2}")
        
        video_capture1.release()
        video_capture2.release()

    print(f"Finished extracting {num_frames} random frames for each video pair.")

def process_videos(input_folder1, input_folder2, output_folder, num_random_frames=10):
    """
    遍历两个输入文件夹中的所有同名视频文件，提取指定数量的随机帧。
    """
    # 获取两个文件夹中的所有视频文件名
    videos1 = set(os.listdir(input_folder1))
    videos2 = set(os.listdir(input_folder2))
    
    # 只选择两个文件夹中都有的同名视频
    common_videos = videos1.intersection(videos2)
    
    # 遍历所有同名视频
    for file_name in common_videos:
        # 只获取 256 x 256 分辨率的视频
        if '_256' in file_name:
            video_path1 = os.path.join(input_folder1, file_name)
            video_path2 = os.path.join(input_folder2, file_name)

            if os.path.isfile(video_path1) and os.path.isfile(video_path2) and file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                print(f"Processing video: {file_name}")

                # 从两组视频中提取共同的随机帧
                extract_random_frames_pair([video_path1], [video_path2], output_folder, num_random_frames)
    
    print("Finished processing all videos.")

input_folder1 = "ref"  # 第一个视频文件夹
input_folder2 = "eval"  # 第二个视频文件夹
output_folder = "output"  # 输出文件夹
num_random_frames = 1500  # 提取的随机帧数

# 从每对同名视频中提取随机帧
process_videos(input_folder1, input_folder2, output_folder, num_random_frames)

