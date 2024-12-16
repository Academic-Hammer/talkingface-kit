#!/bin/bash
set -e

# Check if necessary files exist
if [ ! -f "/app/depth/models/weights_19/depth.pth" ]; then
    echo "Error: Depth model checkpoint not found!"
    exit 1
fi

if [ ! -f "/app/depth/models/weights_19/encoder.pth" ]; then
    echo "Error: DaGAN model checkpoint not found!"
    exit 1
fi

# Check input directories
if [ -z "$(ls -A /app/input_image)" ]; then
    echo "Error: No source image found in input_image directory!"
    exit 1
fi

if [ -z "$(ls -A /app/input_audio_text)" ]; then
    echo "Error: No driving video found in input_audio_text directory!"
    exit 1
fi

# Run inference
echo "Starting inference..."
python demo.py \
    --config config/vox-adv-256.yaml \
    --driving_video /app/input_audio_text/$(ls /app/input_audio_text | head -n 1) \
    --source_image /app/input_image/$(ls /app/input_image | head -n 1) \
    --checkpoint /app/checkpoints/00000289-checkpoint.pth.tar \
    --relative \
    --adapt_scale \
    --kp_num 15 \
    --generator DepthAwareGenerator \
    --result_video /app/output_dir/result.mp4

echo "Inference completed! Result saved to /app/output_dir/result.mp4"