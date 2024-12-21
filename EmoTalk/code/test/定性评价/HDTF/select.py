import numpy as np

# 加载原始的.npy文件
data = np.load('mouth_test.npy')

# 确保数据至少有400行，否则无法从第15行开始取376个数
if data.shape[0] < 390:
    raise ValueError("数据不足400行，无法取出376个数。")

# 从第15行开始取376个数
start_index = 14  # 因为索引是从0开始的
end_index = start_index + 376
selected_data = data[start_index:end_index]

# 将选取的数据保存到新的.npy文件
np.save('selected_mouth_test.npy', selected_data)
