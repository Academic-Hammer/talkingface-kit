import numpy as np

# 定义距离文件名
distances_file = 'distance_67.npy'

# 加载距离数据
distances = np.load(distances_file)

# 确保距离数据的长度为 98
if distances.shape[0] != 390:
    raise ValueError("The distance data must contain exactly 390 entries.")

# 计算平均值
average_distance = np.mean(distances)

# 输出平均值
print("Average projected distance of the first 390 entries:", average_distance)