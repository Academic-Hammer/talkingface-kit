import os
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

# 初始化模型
def initialize_model(model_path):
    # 初始化模型
    model = InceptionResnetV1(pretrained=None)

    # 加载权重
    state_dict = torch.load(model_path)  # 替换为您的权重文件路径

    # 加载权重
    model.load_state_dict(state_dict, strict=False)  # strict=False 忽略未匹配的键
    model.eval()
    return model

# 图片预处理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 增加 batch 维度

# 计算 ID Loss
def compute_id_loss(feature_extractor, input_image, frame_image):
    with torch.no_grad():
        # 提取特征
        input_features = feature_extractor(input_image)
        frame_features = feature_extractor(frame_image)

        # 归一化特征向量
        input_features = torch.nn.functional.normalize(input_features, p=2, dim=1)
        frame_features = torch.nn.functional.normalize(frame_features, p=2, dim=1)

        # 计算余弦相似度
        cosine_similarity = (input_features * frame_features).sum(dim=1)
        id_loss = 1 - cosine_similarity.mean().item()
    return id_loss

# 对所有帧计算平均 ID Loss
def compute_average_id_loss(feature_extractor, input_image, frames_folder):
    total_loss = 0
    frame_count = 0
    for frame_file in os.listdir(frames_folder):
        frame_path = os.path.join(frames_folder, frame_file)
        frame_image = preprocess_image(frame_path)
        loss = compute_id_loss(feature_extractor, input_image, frame_image)
        total_loss += loss
        frame_count += 1
    return total_loss / frame_count

# 主函数
if __name__ == "__main__":
    import argparse
    import os

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='Compute Average ID Loss between input image and generated video frames',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 添加命令行参数
    parser.add_argument('--input_image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--generated_folder', type=str, required=True,
                        help='Directory containing generated video frames')
    parser.add_argument('--model_path', type=str, default='vggface2.pt', help='Path to the model for ID loss calculation')

    # 解析命令行参数
    args = parser.parse_args()

    # 初始化模型
    feature_extractor = initialize_model(args.model_path)

    # 预处理输入图像
    input_image = preprocess_image(args.input_image_path)

    # 计算平均 ID Loss
    average_loss = compute_average_id_loss(feature_extractor, input_image, args.generated_folder)
    print(f"平均 ID Loss: {average_loss}")
