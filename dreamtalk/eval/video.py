import cv2
import os

input_folder = 'videos'  # 改为你的文件夹路径

def resize_videos(path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(path):
        if filename.endswith('.mp4'):  # 仅处理 MP4 文件
            input_video_path = os.path.join(path, filename)
            
            # 创建输出视频文件的路径，添加 "_256" 后缀
            output_video_path = os.path.join(path, filename.replace('.mp4', '_256.mp4'))
            
            cap = cv2.VideoCapture(input_video_path)
            
            # 获取输入视频的帧率和原始分辨率
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 设置输出视频的编码方式、帧率和分辨率
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (256, 256))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                resized_frame = cv2.resize(frame, (256, 256))
                
                out.write(resized_frame)
            
            cap.release()
            out.release()
            print(f"Processed video: {input_video_path} -> {output_video_path}")
    print("All videos processed!")

resize_videos(input_folder)
