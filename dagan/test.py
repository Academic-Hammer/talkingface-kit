import os
from skimage import io

# 检查数据集路径
dataset_path = "/root/autodl-tmp/vox-png"
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")

print("数据集基本信息:")
print(f"数据集根目录: {dataset_path}")
print(f"训练集目录: {train_path}")
print(f"测试集目录: {test_path}")
print("-" * 50)

# 验证训练集
train_ids = os.listdir(train_path)
print(f"训练数据统计:")
print(f"训练标识数量: {len(train_ids)}")
print(f"前5个训练标识: {train_ids[:5]}")
print("-" * 50)

# 检查图像读取
sample_id = train_ids[0]
print(f"示例ID信息:")
print(f"选中的示例ID: {sample_id}")
print(f"示例ID完整路径: {os.path.join(train_path, sample_id)}")
print("-" * 50)

try:
    sample_video = os.listdir(os.path.join(train_path, sample_id))[0]
    print(f"视频信息:")
    print(f"示例视频名称: {sample_video}")
    print(f"示例视频完整路径: {os.path.join(train_path, sample_id, sample_video)}")
    print("-" * 50)

    sample_frame = os.listdir(os.path.join(train_path, sample_id, sample_video))[0]
    print(f"帧信息:")
    print(f"示例帧名称: {sample_frame}")
    print(f"示例帧完整路径: {os.path.join(train_path, sample_id, sample_video, sample_frame)}")
    print("-" * 50)

    sample_path = os.path.join(train_path, sample_id, sample_video, sample_frame)
    image = io.imread(sample_path)
    print(f"图像信息:")
    print(f"图像形状: {image.shape}")
    print(f"图像类型: {image.dtype}")
except Exception as e:
    print(f"错误信息: {str(e)}")