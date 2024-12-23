import os
import numpy as np
import cv2
import dlib

def get_face_detector():
    """
    初始化 dlib 的人脸检测器和关键点预测器
    """
    # 初始化人脸检测器
    detector = dlib.get_frontal_face_detector()
    # 加载关键点预测模型（需要下载模型文件）
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    return detector, predictor

def extract_head_pose(image, detector, predictor):
    """
    使用 dlib 提取图像中的头部姿态 (Pitch, Yaw, Roll)
    如果没有检测到人脸，返回 None
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 检测人脸
    faces = detector(gray)
    if len(faces) == 0:
        return None
    
    # 获取第一个人脸的关键点
    face = faces[0]
    landmarks = predictor(gray, face)
    
    # 提取关键点坐标
    nose_tip = (landmarks.part(30).x, landmarks.part(30).y)  # 鼻尖
    chin = (landmarks.part(8).x, landmarks.part(8).y)        # 下巴
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)  # 左眼外角
    right_eye = (landmarks.part(45).x, landmarks.part(45).y) # 右眼外角
    
    # 计算简单的姿态参数
    # 注意：这是一个简化的计算方法，实际应用中可能需要更复杂的 3D 模型
    image_height, image_width = image.shape[:2]
    
    # 归一化坐标
    pitch = (nose_tip[1] - chin[1]) / image_height     # 俯仰角
    yaw = (right_eye[0] - left_eye[0]) / image_width   # 偏航角
    roll = (right_eye[1] - left_eye[1]) / image_height # 翻滚角
    
    return [pitch, yaw, roll]

def extract_poses_from_folder(folder_path, detector, predictor):
    """
    从文件夹中提取每一帧的头部姿态
    """
    poses = []
    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        image = cv2.imread(file_path)
        if image is not None:
            pose = extract_head_pose(image, detector, predictor)
            if pose is not None:
                poses.append(pose)
    return np.array(poses)

def calculate_apd(poses_generated, poses_driven):
    """
    计算生成视频和驱动视频的平均姿态距离 (APD)
    """
    min_length = min(len(poses_generated), len(poses_driven))
    poses_generated = poses_generated[:min_length]
    poses_driven = poses_driven[:min_length]
    distances = np.linalg.norm(poses_generated - poses_driven, axis=1)
    return np.mean(distances)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calculate Average Pose Distance (APD) between generated and driving videos',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--driving_folder', type=str, required=True,
                      help='Directory containing driving images/videos')
    parser.add_argument('--generated_folder', type=str, required=True,
                      help='Directory containing generated images/videos')
    
    args = parser.parse_args()
    
    # 初始化人脸检测器和关键点预测器
    detector, predictor = get_face_detector()
    
    # 提取姿态
    poses_driven = extract_poses_from_folder(args.driving_folder, detector, predictor)
    poses_generated = extract_poses_from_folder(args.generated_folder, detector, predictor)
    
    if len(poses_driven) == 0 or len(poses_generated) == 0:
        print("未能提取到姿态，请检查输入图像！")
    else:
        apd = calculate_apd(poses_generated, poses_driven)
        print(f"平均姿态距离 (APD): {apd:.4f}")