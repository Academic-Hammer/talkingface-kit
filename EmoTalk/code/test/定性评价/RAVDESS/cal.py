import numpy as np
import pandas as pd

# 设置文件路径
npy_file_path = './test.npy'  # .npy 文件路径
excel_file_path = './65.xlsx'  # Excel 文件路径

# 读取 .npy 文件
data_npy = np.load(npy_file_path)

# 读取 Excel 文件，确保 header=None，因为没有标题行
data_excel = pd.read_excel(excel_file_path, header=None)  # 不使用标题行

# 将 Excel 数据转换为 NumPy 数组
data_excel_np = data_excel.to_numpy()

# 确保矩阵的维度是正确的
assert data_npy.shape == (147, 52), "npy 文件的形状应该是 (146, 52)"
assert data_excel_np.shape == (52, 3), "Excel 文件应该有 52 行和 3 列"

# 矩阵乘法
result = np.dot(data_npy, data_excel_np)

# 输出结果
print("Result of the matrix multiplication (shape: {0}):".format(result.shape))
print(result)

# 如果需要将结果保存到文件中
output_file_path = './test65.npy'
np.save(output_file_path, result)  # 保存结果为 .npy 文件
print(f"Result saved to {output_file_path}")
