import numpy as np
import os

# 定义要转换的 .npy 文件名
npy_files = ['anno65.npy', 'anno67.npy', 'test65.npy', 'test67.npy']

# 遍历每个 .npy 文件并进行转换
for npy_file in npy_files:
    # 加载 .npy 文件
    data = np.load(npy_file)
    
    # 生成对应的 txt 文件名
    txt_file = os.path.splitext(npy_file)[0] + '.txt'
    
    # 将数据保存为 .txt 文件
    np.savetxt(txt_file, data, delimiter=',', fmt='%.18e')  # 使用科学计数法格式输出

    print(f"Converted {npy_file} to {txt_file}")
