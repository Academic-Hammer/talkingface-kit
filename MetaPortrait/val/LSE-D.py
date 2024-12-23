import os
import cv2
import dlib
import numpy as np

def extract_keypoints(image):
    """
    从图像中提取关键点。
    :param image: 输入图像 (BGR 格式)
    :return: 关键点坐标数组 [68, 2]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None  # 没检测到人脸
    face = faces[0]  # 假设只处理第一个检测到的人脸
    landmarks = predictor(gray, face)
    keypoints = np.array([[p.x, p.y] for p in landmarks.parts()])
    return keypoints

def process_frames(directory):
    """
    处理文件夹中的所有帧，提取关键点。
    :param directory: 文件夹路径
    :return: 所有帧的关键点列表
    """
    keypoints_list = []
    for frame_file in sorted(os.listdir(directory)):
        if frame_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            frame_path = os.path.join(directory, frame_file)
            image = cv2.imread(frame_path)
            keypoints = extract_keypoints(image)
            if keypoints is not None:
                keypoints_list.append(keypoints)
    return np.array(keypoints_list)  # 返回 [num_frames, 68, 2]

def calculate_lse_d(generated_keypoints, driven_keypoints):
    """
    计算 LSE-D（动态准确性）。
    :param generated_keypoints: 生成视频的关键点数组 [num_frames, 68, 2]
    :param driven_keypoints: 驱动视频的关键点数组 [num_frames, 68, 2]
    :return: LSE-D 值
    """
    diff = generated_keypoints - driven_keypoints
    lse_d = np.mean(np.linalg.norm(diff, axis=-1))  # 平均 L2 范数
    return lse_d

if __name__ == "__main__":
    import argparse

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='Calculate LSE-D (Dynamic Accuracy) for generated and driving video frames')

    # 添加命令行参数
    parser.add_argument('--generated_folder', type=str, required=True,
                        help='Directory containing generated video frames')
    parser.add_argument('--driving_folder', type=str, required=True, help='Directory containing driving video frames')
    parser.add_argument('--model_path', type=str, default='shape_predictor_68_face_landmarks.dat',
                        help='Path to the Dlib shape predictor model')

    # 解析命令行参数
    args = parser.parse_args()

    # 加载 Dlib 的人脸检测器和 68 点模型
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.model_path)

    # 提取生成视频和驱动视频的关键点
    generated_keypoints = process_frames(args.generated_folder)
    driven_keypoints = process_frames(args.driving_folder)

    # 计算 LSE-D
    lse_d = calculate_lse_d(generated_keypoints, driven_keypoints)

    # 输出 LSE-D 结果
    print(f"LSE-D (动态准确性): {lse_d}")

