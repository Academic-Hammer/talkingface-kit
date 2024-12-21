import numpy as np

# 定义文件名
differences_file = 'differences_test.npy'

# 加载 .npy 文件
differences_data = np.load(differences_file)

# 确保数据的列数正确
if differences_data.shape[1] < 3:
    raise ValueError("The data must have at least three columns.")

# 计算每行的距离（第一个和第二个元素的平方和的平方根）
distances = np.sqrt(differences_data[:, 0]**2 + differences_data[:, 1]**2)

# 输出为 .npy 文件
np.save('calculated_distances.npy', distances)

# 输出为 .txt 文件
np.savetxt('calculated_distances.txt', distances, delimiter=',', fmt='%.18e')

print("Distances calculated and saved to calculated_distances.npy and calculated_distances.txt")
