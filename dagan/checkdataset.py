import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   filename='dataset_check.log',
                   filemode='w')

def check_dataset_structure(root_dir):
    """检查数据集的基本结构"""
    logging.info(f"Checking dataset structure in {root_dir}")
    
    # 检查train和test文件夹是否存在
    required_dirs = ['train', 'test']
    for dir_name in required_dirs:
        dir_path = os.path.join(root_dir, dir_name)
        if not os.path.exists(dir_path):
            logging.error(f"Missing required directory: {dir_path}")
            return False
        
    return True

def check_video_frames(video_dir):
    """检查视频帧的完整性"""
    try:
        frames = sorted(os.listdir(video_dir))
        if len(frames) == 0:
            logging.error(f"Empty directory: {video_dir}")
            return False
            
        # 检查每一帧
        for frame in frames:
            frame_path = os.path.join(video_dir, frame)
            if not frame.lower().endswith(('.png', '.jpg', '.jpeg')):
                logging.error(f"Invalid frame format: {frame_path}")
                return False
                
            # 尝试读取图像
            img = cv2.imread(frame_path)
            if img is None:
                logging.error(f"Cannot read frame: {frame_path}")
                return False
                
            # 检查图像尺寸
            if img.shape[:2] != (256, 256):  # 假设需要256x256
                logging.warning(f"Incorrect image size in {frame_path}: {img.shape}")
                
        return True
    except Exception as e:
        logging.error(f"Error checking {video_dir}: {str(e)}")
        return False

def check_dataset_samples(root_dir):
    """检查所有样本"""
    logging.info("Starting dataset samples check...")
    
    total_videos = 0
    corrupted_videos = []
    
    for split in ['train', 'test']:
        split_dir = os.path.join(root_dir, split)
        videos = os.listdir(split_dir)
        
        logging.info(f"Checking {split} split: {len(videos)} videos")
        
        for video in tqdm(videos, desc=f"Checking {split}"):
            video_dir = os.path.join(split_dir, video)
            if os.path.isdir(video_dir):
                total_videos += 1
                if not check_video_frames(video_dir):
                    corrupted_videos.append(video_dir)
                    
    logging.info(f"Dataset check completed:")
    logging.info(f"Total videos: {total_videos}")
    logging.info(f"Corrupted videos: {len(corrupted_videos)}")
    
    if corrupted_videos:
        logging.info("Corrupted videos list:")
        for video in corrupted_videos:
            logging.info(video)
            
    return corrupted_videos

def check_dataloader(dataset_path, batch_size=4, num_workers=2):
    """测试数据加载器"""
    try:
        from frames_dataset import FramesDataset  # 导入你的数据集类
        
        logging.info("Testing DataLoader...")
        
        dataset = FramesDataset(dataset_path)  # 可能需要添加其他参数
        dataloader = DataLoader(dataset, 
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
        
        # 测试几个batch的加载
        for i, batch in enumerate(tqdm(dataloader, desc="Testing DataLoader")):
            if i >= 10:  # 测试10个batch
                break
                
        logging.info("DataLoader test completed successfully")
        return True
    except Exception as e:
        logging.error(f"DataLoader test failed: {str(e)}")
        return False

if __name__ == "__main__":
    dataset_root = "root/autodl-tmp/vox-png"  # 修改为你的数据集路径
    
    # 1. 检查数据集结构
    if not check_dataset_structure(dataset_root):
        logging.error("Dataset structure check failed!")
        exit(1)
    
    # 2. 检查样本完整性
    corrupted_videos = check_dataset_samples(dataset_root)
    
    # 3. 测试数据加载器
    if not check_dataloader(dataset_root):
        logging.error("DataLoader test failed!")
        exit(1)
    
    if not corrupted_videos:
        logging.info("All checks passed successfully!")
    else:
        logging.warning("Some videos are corrupted. Check the log file for details.")