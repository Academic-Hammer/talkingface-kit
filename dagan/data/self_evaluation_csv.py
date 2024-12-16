import os
import random
import csv
from pathlib import Path


def generate_self_reenactment_pairs(data_root, output_path, total_pairs=2000):
    """
    为每个视频生成自身重构的图像对,总数限制在2000对
    Args:
        data_root: 数据根目录
        output_path: 输出csv文件路径
        total_pairs: 需要生成的总对数
    """
    pairs = []
    video_dirs = list(Path(data_root).glob("id*"))

    # 计算每个视频需要生成的对数
    pairs_per_video = total_pairs // len(video_dirs)
    remaining_pairs = total_pairs % len(video_dirs)

    for video_dir in video_dirs:
        if not video_dir.is_dir():
            continue

        # 获取该视频下所有图片
        images = sorted([str(x.relative_to(data_root)) for x in video_dir.glob("*.png")])
        if len(images) < 2:
            continue

        # 确定这个视频需要生成的对数
        num_pairs = pairs_per_video
        if remaining_pairs > 0:
            num_pairs += 1
            remaining_pairs -= 1

        # 随机生成图像对
        for _ in range(num_pairs):
            # 从同一视频中随机选择两帧
            source, driving = random.sample(images, 2)
            pairs.append([source, driving])

        if len(pairs) >= total_pairs:
            break

    # 如果生成的对数超过了要求,随机删除多余的
    if len(pairs) > total_pairs:
        pairs = random.sample(pairs, total_pairs)

    # 写入CSV文件
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source_self', 'driving'])  # 写入header
        writer.writerows(pairs)

    print(f"Generated {len(pairs)} pairs in total")
    return pairs

if __name__ == "__main__":
    data_root = "/root/autodl-tmp/vox-png/test"  # 替换为你的数据目录
    output_path = "self_reenactment_pairs.csv"
    generate_self_reenactment_pairs(data_root, output_path, total_pairs=2000)