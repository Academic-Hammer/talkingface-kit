#!/bin/bash

# Ensure the script exits on any error
set -e

# Define output log file
OUTPUT_LOG="evaluate.log"

# Start logging

# Run PSNR_SSIM calculation
python psnr_ssim.py >> "$OUTPUT_LOG" 2>&1

# Run FID calculation
python fid.py

# Run pipeline script
python run_pipeline.py --videofile May_out.mp4 --reference May --data_dir output

# Run score calculation
python calculate_scores_real_videos.py --videofile May_out.mp4 --reference May --data_dir output

# Run NIQE calculation
python niqe.py >> "$OUTPUT_LOG" 2>&1

