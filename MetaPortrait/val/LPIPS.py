import lpips
import torch
from torchvision import transforms
from PIL import Image
import os

# 1. 加载和预处理图片
def load_and_preprocess_image(image_path):
    """
    加载并预处理图片，确保它是RGB格式，并归一化到[-1, 1]范围
    """
    img = Image.open(image_path).convert('RGB')  # 转为RGB格式
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 可以调整大小
        transforms.ToTensor(),  # 转换为Tensor，范围是[0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 归一化到[-1, 1]
    ])
    img_tensor = transform(img).unsqueeze(0)  # 增加batch维度 (1, 3, 64, 64)
    return img_tensor

# 2. 计算一对图片之间的LPIPS损失
def calculate_lpips_for_image_pair(img0_path, img1_path):
    """
    计算一对图片之间的LPIPS损失
    """
    img0 = load_and_preprocess_image(img0_path)
    img1 = load_and_preprocess_image(img1_path)
    
    # 计算LPIPS损失
    d = loss_fn_alex(img0, img1)
    return d.item()  # 返回损失值

# 3. 批量处理两个图像集
def process_image_sets(original_folder, generated_folder):
    """
    遍历原始图像集和生成图像集中的每一对图片，计算它们之间的LPIPS损失
    """
    original_images = os.listdir(original_folder)
    generated_images = os.listdir(generated_folder)
    
    results = []
    
    # 假设原始图像集和生成图像集的图片是按文件名对应的
    for img_name in original_images:
        if img_name in generated_images:  # 确保两个文件夹中的图像对是对应的
            img0_path = os.path.join(original_folder, img_name)
            img1_path = os.path.join(generated_folder, img_name)
            
            # 计算LPIPS损失
            lpips_loss = calculate_lpips_for_image_pair(img0_path, img1_path)
            results.append((img_name, lpips_loss))
    
    return results

if __name__ == "__main__":
    import argparse
    import os

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Calculate LPIPS loss between driving and generated image sets',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 添加命令行参数
    parser.add_argument('--driving_folder', type=str, required=True, help='Directory containing driving images')
    parser.add_argument('--generated_folder', type=str, required=True, help='Directory containing generated images')
    parser.add_argument('--model_path', type=str, default='alex.pth', help='Path to the LPIPS pre-trained model')

    # 解析命令行参数
    args = parser.parse_args()

    # 初始化LPIPS损失函数（使用AlexNet）
    loss_fn_alex = lpips.LPIPS(net='alex', model_path=args.model_path)

    # 计算所有图像对的LPIPS损失
    results = process_image_sets(args.driving_folder, args.generated_folder)

    # 累加LPIPS损失并计算平均值
    sum_loss = 0
    for img_name, lpips_loss in results:
        sum_loss += lpips_loss

    average_loss = sum_loss / len(results)
    print(f'LPIPS loss: {average_loss:.4f}')

