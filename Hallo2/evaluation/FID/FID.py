import torch
from pytorch_fid import fid_score
# import logging

# # 配置日志记录
# logging.basicConfig(
#     filename='fid_score.log',  # 日志文件名
#     filemode='a',             # 追加模式
#     level=logging.INFO,       # 日志记录级别
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

def calculate_fid(real_images_folder='./original_frames', generated_images_folder='./generated_frames'):
    # 设置真实数据和生成数据文件夹路径
    # real_images_folder = 'raw_results'
    # generated_images_folder = 'final_results'

    # 设置参数
    new_batch_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dims = 2048  # 使用 Inception 模型的默认特征维度

    # try:
    # 计算 FID 值
    fid_value = fid_score.calculate_fid_given_paths(
        [real_images_folder, generated_images_folder],
        batch_size=new_batch_size,
        device=device,
        dims=dims
    )
    # logging.info(f'FID value: {fid_value}')
    # print(f'FID value: {fid_value}')

    print(__file__)

    return fid_value
    # except Exception as e:
    #     logging.error(f'Error occurred while calculating FID: {str(e)}')

    

if __name__ == '__main__':
    calculate_fid()
