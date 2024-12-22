import os
import csv
import numpy as np
from tqdm import tqdm


def create_evaluation_csv(dataset_path, output_file='data/vox_evaluation_test.csv'):
    """
    创建用于模型评估的CSV文件，生成从根目录开始的绝对路径

    这个函数会生成完整的绝对路径，格式如：
    /root/autodl-tmp/test/test/id11248#ULBH3A8DjPM#00004.txt#000.mp4/0000111.png

    参数:
        dataset_path: 数据集根目录的路径（例如：/root/autodl-tmp/test/test）
        output_file: CSV文件的输出路径
    """
    # 确保使用绝对路径
    dataset_path = os.path.abspath(dataset_path)
    output_file = os.path.abspath(output_file)

    if not os.path.exists(dataset_path):
        print(f"错误: 数据集路径 {dataset_path} 不存在!")
        return False

    print(f"正在检查数据集路径: {dataset_path}")
    print(f"输出文件将保存到: {output_file}")

    try:
        # 获取所有视频文件夹
        test_videos = [d for d in os.listdir(dataset_path)
                       if os.path.isdir(os.path.join(dataset_path, d))]
        print(f"找到 {len(test_videos)} 个视频文件夹")

        if len(test_videos) < 2:
            print("错误: 至少需要2个视频文件夹才能创建评估对!")
            return False

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'driving'])

            print("开始创建2000对评估图像组合...")
            valid_pairs = 0

            for i in tqdm(range(2000), desc="生成评估对"):
                try:
                    # 随机选择两个不同的视频
                    video_pair = np.random.choice(test_videos, size=2, replace=False)

                    # 构建第一个视频的完整路径
                    source_video_path = os.path.join(dataset_path, video_pair[0])
                    source_frames = [f for f in os.listdir(source_video_path)
                                     if f.lower().endswith('.png')]

                    if not source_frames:
                        continue

                    # 构建第二个视频的完整路径
                    driving_video_path = os.path.join(dataset_path, video_pair[1])
                    driving_frames = [f for f in os.listdir(driving_video_path)
                                      if f.lower().endswith('.png')]

                    if not driving_frames:
                        continue

                    # 随机选择帧并构建完整的绝对路径
                    source_frame = np.random.choice(source_frames)
                    driving_frame = np.random.choice(driving_frames)

                    # 构建完整的绝对路径
                    source_abs_path = os.path.join(dataset_path, video_pair[0], source_frame)
                    driving_abs_path = os.path.join(dataset_path, video_pair[1], driving_frame)

                    # 验证文件存在性
                    if os.path.exists(source_abs_path) and os.path.exists(driving_abs_path):
                        writer.writerow([source_abs_path, driving_abs_path])
                        valid_pairs += 1
                    else:
                        if not os.path.exists(source_abs_path):
                            print(f"\n源文件不存在: {source_abs_path}")
                        if not os.path.exists(driving_abs_path):
                            print(f"\n驱动文件不存在: {driving_abs_path}")

                except Exception as e:
                    print(f"\n处理第 {i + 1} 对时出错: {str(e)}")
                    continue

            print(f"\n成功创建了 {valid_pairs} 个有效的评估数据对")
            print(f"评估CSV文件已保存到: {output_file}")

            if valid_pairs < 100:
                print("\n警告: 生成的有效数据对数量过少，可能会影响评估效果")

            return valid_pairs > 0

    except Exception as e:
        print(f"创建CSV文件时出错: {str(e)}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='创建评估数据集的CSV文件')
    parser.add_argument('--dataset_path', type=str,
                        default='/root/autodl-tmp/vox-png/test',
                        help='数据集根目录的绝对路径')
    parser.add_argument('--output_file', type=str,
                        default='data/vox_evaluation.csv',
                        help='输出CSV文件的路径')

    args = parser.parse_args()

    print("开始创建评估数据集...")
    success = create_evaluation_csv(args.dataset_path, args.output_file)

    if success:
        print("评估数据集创建完成!")
    else:
        print("评估数据集创建失败!")