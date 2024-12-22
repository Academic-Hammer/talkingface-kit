import os
import csv
import random

# 数据集路径（根据实际情况修改）
test_dir = '/root/autodl-tmp/vox-png/test'  # 包含以 idxxx#xxx#xxx.txt#xxx.mp4 格式命名的文件夹
output_dir = '/root/autodl-tmp/dagan/data'  # 输出 CSV 文件的目录

# 获取所有文件夹内的 PNG 文件，并按 ID 分类
def get_files(directory):
    file_dict = {}
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):  # 确保是文件夹
            files = []
            for file in os.listdir(folder_path):
                if file.endswith('.png'):  # 只考虑 PNG 文件
                    files.append(file)
            if len(files) >= 2:  # 只处理每个文件夹下有至少两张图片的情况
                file_dict[folder] = files
    return file_dict

# 获取 test 文件夹中的图像文件
file_dict = get_files(test_dir)

# 获取所有有效的文件夹 ID（即每个 ID 下至少有两张图像）
valid_ids = list(file_dict.keys())

pairs = []

while len(pairs) < 2000:
    # 随机选择两个不同的 ID
    source_id, driving_id = random.sample(valid_ids, 2)

    # 获取 source_self 和 driving 的图像文件列表
    source_files = file_dict[source_id]
    driving_files = file_dict[driving_id]

    # 随机选择图像
    source_file = random.choice(source_files)
    driving_file = random.choice(driving_files)

    # 生成相对路径并添加到图像对列表
    source_path = os.path.join(source_id, source_file)
    driving_path = os.path.join(driving_id, driving_file)
    pairs.append([source_path, driving_path])


# 创建输出目录
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 将图像对保存到 CSV 文件
def save_to_csv(pairs, output_dir):
    create_dir(output_dir)
    file_path = os.path.join(output_dir, 'cross_identity_pairs.csv')

    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['source_self', 'driving'])  # 写入 CSV 的标题行
        for pair in pairs:
            writer.writerow(pair)

    print(f"交叉验证集的 CSV 文件已生成：{file_path}")

# 保存选定的 2000 对数据到 CSV
save_to_csv(pairs, output_dir)
