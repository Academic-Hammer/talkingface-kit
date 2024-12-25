import numpy as np

# 定义距离文件名
distances_file = '65diff.npy'

# 加载距离数据
distances = np.load(distances_file)

# 确保距离数据的长度为 146
if distances.shape[0] != 146:
    raise ValueError("The distance data must contain exactly 146 entries.")

# 计算平均值
average_distance = np.mean(distances)

# 输出平均值
print("Average projected distance of the first 146 entries:", average_distance)