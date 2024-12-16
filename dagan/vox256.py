import os
import csv
import random
from pathlib import Path
import numpy as np


def create_vox_csv(dataset_path, output_file='vox256.csv', num_pairs=8000):
    """
    为 PNG 格式的数据集生成配置文件

    参数:
        dataset_path: 数据集根目录路径（包含 PNG 文件）
        output_file: 输出的 CSV 文件路径
        num_pairs: 需要生成的配对数量
    """
    # 获取当前工作目录
    current_dir = os.getcwd()
    output_path = os.path.join(current_dir, output_file)

    # 获取所有 PNG 文件
    png_files = []
    if os.path.exists(dataset_path):
        for file in os.listdir(dataset_path):
            if file.lower().endswith('.png'):
                full_path = os.path.join(dataset_path, file)
                png_files.append(full_path)

    print(f"找到 {len(png_files)} 个 PNG 文件")

    if len(png_files) < 2:
        raise ValueError(f"在 {dataset_path} 中找到的 PNG 文件数量不足，至少需要2个文件")

    # 写入 CSV 文件
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'driving', 'frame'])  # 写入标题行

        pairs_created = 0
        max_attempts = num_pairs * 2
        attempts = 0

        while pairs_created < num_pairs and attempts < max_attempts:
            attempts += 1
            try:
                # 随机选择两个不同的 PNG 文件作为源和驱动
                source_file, driving_file = random.sample(png_files, 2)

                # 获取相对路径（如果需要）
                source_relative = os.path.relpath(source_file, current_dir)
                driving_relative = os.path.relpath(driving_file, current_dir)

                # 生成随机帧号（这里使用0，因为每个文件都是单独的帧）
                frame_number = 0

                # 写入 CSV
                writer.writerow([source_relative, driving_relative, frame_number])
                pairs_created += 1

                if pairs_created % 1000 == 0:
                    print(f"已生成 {pairs_created} 个配对...")

            except Exception as e:
                print(f"生成配对时发生错误: {str(e)}")
                continue

    print(f"配置文件已生成: {output_path}")
    print(f"共生成 {pairs_created} 个配对")


if __name__ == "__main__":
    # 设置数据集路径
    dataset_path = "/root/autodl-tmp/dataset/vox-png"
    create_vox_csv(dataset_path)