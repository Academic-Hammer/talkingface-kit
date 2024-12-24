import cv2
import numpy as np
from scipy.spatial import distance

# 使用 dlib 提取关键点
import dlib

# 初始化 dlib 的人脸检测器和关键点检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 需要下载权重文件

def extract_landmarks(image):
    """提取单帧图像的面部关键点"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    shape = predictor(gray, faces[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    return landmarks

def compute_lmd(reference_video_path, generated_video_path):
    """计算 LMD 指标"""
    # 打开参考视频和生成视频
    ref_cap = cv2.VideoCapture(reference_video_path)
    gen_cap = cv2.VideoCapture(generated_video_path)
    
    ref_landmarks_list = []
    gen_landmarks_list = []
    frame_count = 0

    while True:
        ref_ret, ref_frame = ref_cap.read()
        gen_ret, gen_frame = gen_cap.read()
        
        if not ref_ret or not gen_ret:
            break

        ref_landmarks = extract_landmarks(ref_frame)
        gen_landmarks = extract_landmarks(gen_frame)
        
        if ref_landmarks is not None and gen_landmarks is not None:
            ref_landmarks_list.append(ref_landmarks)
            gen_landmarks_list.append(gen_landmarks)
            frame_count += 1
    
    # 检查关键点数据是否对齐
    if len(ref_landmarks_list) != len(gen_landmarks_list):
        raise ValueError("Mismatch in landmark detection between reference and generated videos.")
    
    # 计算 LMD
    lmd = 0
    for ref, gen in zip(ref_landmarks_list, gen_landmarks_list):
        distances = [distance.euclidean(r, g) for r, g in zip(ref, gen)]
        lmd += np.mean(distances)

    lmd /= frame_count
    # print(f"LMD: {lmd}")
    return lmd

# 示例：运行 LMD 计算
# reference_video = "../MP4/Source/Jae-in.mp4"
# generated_video = "../MP4/Hallo/Jae-in.mp4"

# compute_lmd(reference_video, generated_video)