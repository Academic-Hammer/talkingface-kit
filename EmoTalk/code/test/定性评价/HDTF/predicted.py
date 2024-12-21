import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 加载预测值和实际值
predictions = np.load('selected_mouth_test.npy')  # 预测值
actuals = np.load('selected_mouth_anno.npy')      # 实际值

# 计算指标
mae = mean_absolute_error(actuals, predictions)  # 平均绝对误差
mse = mean_squared_error(actuals, predictions)    # 均方误差
rmse = np.sqrt(mse)                               # 均方根误差
r2 = r2_score(actuals, predictions)               # R^2决定系数

# 输出结果
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R^2 Score: {r2}')
