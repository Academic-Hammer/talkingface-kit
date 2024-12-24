#!/bin/bash

# 执行第一个命令
python run_pipeline.py --videofile eval/data/demo.mp4 --reference t1 --data_dir tmp

# 执行第二个命令并追加输出到 all_scores.txt
python calculate_scores_real_videos.py --videofile eval/data/demo.mp4 --reference t1 --data_dir tmp >> all_scores.txt