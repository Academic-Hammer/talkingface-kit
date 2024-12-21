import numpy as np

# 定义文件名
test_file_1 = 'annotation65.npy'
test_file_2 = 'annotation67.npy'

# 加载 .npy 文件
test_data_1 = np.load(test_file_1)
test_data_2 = np.load(test_file_2)

# 确保两个数据的行数相同
if test_data_1.shape[0] != test_data_2.shape[0]:
    raise ValueError("Both files must have the same number of rows.")

# 计算每一行的差异
# 假设 test_data_1 和 test_data_2 都是 n 行 3 列的数组
differences = test_data_1 - test_data_2  # 每一元素相减

# 输出为 .npy 文件
np.save('differences_anno.npy', differences)

# 输出为 .txt 文件
np.savetxt('differences_anno.txt', differences, delimiter=',', fmt='%.18e')

print("Differences calculated and saved to differences.npy and differences.txt")
