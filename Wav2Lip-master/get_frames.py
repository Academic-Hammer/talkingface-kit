import cv2  # 导入OpenCV库，用于视频处理
import os  # 导入os库，用于文件和目录操作
import concurrent.futures  # 导入并发库，用于多线程处理
import time  # 导入时间库，用于计算时间
import argparse  # 导入argparse库，用于命令行参数解析


# 定义保存帧的函数
def save_frame(frame, frame_filename):
    cv2.imwrite(frame_filename, frame)  # 使用OpenCV保存帧到文件


# 定义提取视频帧的函数
def extract_frames(video_path, output_folder, frame_interval, max_threads):
    cap = cv2.VideoCapture(video_path)  # 打开视频文件
    frame_count = 0  # 初始化帧计数器
    futures = []  # 初始化futures列表，用于存储线程任务

    start_time = time.time()  # 记录开始时间

    # 使用ThreadPoolExecutor创建一个线程池，最大线程数为max_threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        while cap.isOpened():  # 当视频文件打开时
            ret, frame = cap.read()  # 读取一帧
            if not ret:  # 如果读取失败，跳出循环
                break

            if frame_count % frame_interval == 0:  # 每隔frame_interval帧处理一次
                seconds = frame_count // frame_interval  # 计算当前帧对应的秒数
                frame_filename = os.path.join(output_folder, f'frame_{seconds:04d}.jpg')  # 构建帧文件名
                futures.append(executor.submit(save_frame, frame, frame_filename))  # 提交保存帧的任务到线程池

            frame_count += 1  # 增加帧计数器

        cap.release()  # 释放视频文件

        for future in concurrent.futures.as_completed(futures):  # 等待所有线程完成
            future.result()  # 获取线程结果

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算执行时间

    print(f'提取完成，共提取了 {frame_count // frame_interval} 帧')  # 输出提取的帧数
    print(f'程序执行时间: {elapsed_time:.2f} 秒')  # 输出程序执行时间


# 创建文件夹保存帧
output_folder = 'video_frames'
os.makedirs(output_folder, exist_ok=True)  # 如果文件夹不存在，则创建

# 使用argparse解析命令行参数
parser = argparse.ArgumentParser(description='Extract frames from a video file.')
parser.add_argument('video_path', type=str, help='Path to the video file.')
parser.add_argument('--output_folder', type=str, default='video_frames', help='Folder to save extracted frames.')
parser.add_argument('--frame_interval', type=int, default=1, help='Interval (in seconds) between frames to extract.')
parser.add_argument('--max_threads', type=int, default=8, help='Maximum number of threads to use.')

args = parser.parse_args()

# 获取视频帧率
cap = cv2.VideoCapture(args.video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
cap.release()  # 释放视频文件
frame_interval = int(args.frame_interval * fps)  # 每秒提取一帧

# 提取视频帧
extract_frames(args.video_path, args.output_folder, frame_interval, args.max_threads)  # 调用提取帧的函数