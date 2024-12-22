import os
import shutil
import pandas as pd


def create_directories(source_dir, driving_dir):
    """创建必要的目录"""
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(driving_dir, exist_ok=True)


def process_image_pairs(csv_path, base_path, source_output_dir, driving_output_dir):
    """处理图像对并保存到指定目录"""
    # 读取CSV文件
    with open(csv_path, 'r') as f:
        csv_content = f.read()

    # 将CSV内容分割成行
    pairs = csv_content.strip().split(' ')

    # 创建空列表来存储source和driving
    sources = []
    drivings = []

    # 处理每一对
    for pair in pairs:
        source, driving = pair.split(',')
        sources.append(source)
        drivings.append(driving)

    # 创建DataFrame
    df = pd.DataFrame({
        'source_self': sources,
        'driving': drivings
    })

    # 处理每一对图像
    for idx, row in df.iterrows():
        # 构建源文件路径
        source_path = os.path.join(base_path, row['source_self'])
        driving_path = os.path.join(base_path, row['driving'])

        # 构建目标文件路径
        source_output_path = os.path.join(source_output_dir, f'img{idx:05d}.png')
        driving_output_path = os.path.join(driving_output_dir, f'img{idx:05d}.png')

        # 复制文件
        try:
            shutil.copy2(source_path, source_output_path)
            print(f"Copied source_self image {idx}: {source_path} -> {source_output_path}")
        except Exception as e:
            print(f"Error copying source_self image {idx}: {e}")

        try:
            shutil.copy2(driving_path, driving_output_path)
            print(f"Copied driving image {idx}: {driving_path} -> {driving_output_path}")
        except Exception as e:
            print(f"Error copying driving image {idx}: {e}")


def main():
    # 设置路径
    base_path = r"E:\grade3-1\voice\dataset\vox\face-video-preprocessing\vox-png\test"
    source_output_dir = r"E:\grade3-1\dagan_docker\evaluation_set\source_cross"
    driving_output_dir = r"E:\grade3-1\dagan_docker\evaluation_set\gt_cross"
    csv_path = r"E:\grade3-1\dagan_docker\data\cross_identity_pairs.csv"

    # 创建必要的目录
    create_directories(source_output_dir, driving_output_dir)

    # 处理图像对
    process_image_pairs(csv_path, base_path, source_output_dir, driving_output_dir)


if __name__ == "__main__":
    main()