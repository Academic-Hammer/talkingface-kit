import os
import sys
import shutil
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_paths():
    reference_path = None

    # 检查参数数量
    if len(sys.argv) <= 2:
        logging.error("Usage: python cpbd.py --reference <reference_file>")
        sys.exit(1)

    # 遍历参数
    for i, arg in enumerate(sys.argv):
        if arg == "--reference":
            # 检查是否有下一个参数，即文件路径
            if i + 1 < len(sys.argv):
                reference_path = sys.argv[i + 1]
                # 检查路径是否是.mp4文件
                if not reference_path.endswith((".mp4", ".avi", ".mov", ".mkv")) or not os.path.isfile(reference_path):
                    logging.error(f"The provided reference path is not a valid .mp4 file: {reference_path}")
                    sys.exit(1)
            else:
                logging.error("No file path provided after --reference argument.")
                sys.exit(1)

    if reference_path is None:
        logging.error("--reference arguments must be provided.")
        sys.exit(1)

    logging.info(f"Reference path: {reference_path}")
    return reference_path

path = get_paths()

if os.path.exists('tmp_dir'):
    shutil.rmtree('tmp_dir')
os.makedirs('tmp_dir', exist_ok=True)

logging.info('Computing LSE calculation...')

os.system('python run_pipeline.py --videofile ' + path + ' --reference wav2lip --data_dir tmp_dir')
os.system('python calculate_scores_real_videos.py --videofile ' + path + ' --reference wav2lip --data_dir tmp_dir')

# python lse.py --reference ..\reference\May.mp4