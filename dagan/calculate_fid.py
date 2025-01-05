from pytorch_fid import fid_score

# 计算 FID
fid_value = fid_score.calculate_fid_given_paths(
    ['/root/autodl-tmp/dagan/evaluation_set/source_cross', '/root/autodl-tmp/dagan/evaluation_set/generate_form'],  # 路径列表
    batch_size=50,  # 批量大小
    device='cuda',  # 使用 GPU
    dims=2048       # Inception v3 的特征维度
)

print(f"FID: {fid_value}")