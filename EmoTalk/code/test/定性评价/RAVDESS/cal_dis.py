import numpy as np

# 定义文件名
annotation_file = 'test67.npy'
test_file = 'anno67.npy'

# 加载 .npy 文件
annotation_data = np.load(annotation_file)
test_data = np.load(test_file)

# 确保数据有足够的行
if annotation_data.shape[0] < 146 or test_data.shape[0] < 146:
    raise ValueError("Both files must have at least 98 rows.")

# 取前 146 行
annotation_subset = annotation_data[:146]
test_subset = test_data[:146]

# 提取 x 和 z 坐标 (假设 x 在第 0 列, z 在第 1 列)
x_coords_annotation = annotation_subset[:, 0]
y_coords_annotation = annotation_subset[:, 1]
x_coords_test = test_subset[:, 0]
y_coords_test = test_subset[:, 1]

# 计算投影距离 (只考虑 x 和 z 坐标)
# 距离计算公式: distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)
distances = np.sqrt((x_coords_test - x_coords_annotation) ** 2 + (y_coords_test - y_coords_annotation) ** 2)

#y_coords_annotation = annotation_subset[:, 1]
#y_coords_test = test_subset[:, 1]
#distances = np.sqrt((y_coords_test - y_coords_annotation) ** 2)
# 输出为 .npy 文件
np.save('67diff.npy', distances)

# 输出为 .txt 文件
np.savetxt('distance_test.txt', distances, delimiter=',', fmt='%.18e')

print("Projected distance calculations saved to projected_distances.npy and projected_distances.txt")

