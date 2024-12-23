#!/bin/bash

# 激活 conda 环境
source /root/anaconda3/etc/profile.d/conda.sh
conda activate meta_portrait_base


# 检查第一个参数
case "$1" in
    "aed")
        cd val
        python AED.py --generated_folder "$2" --driving_folder "$3"
        ;;
    "apd")
        cd val
        python APD.py --generated_folder "$2" --driving_folder "$3"
        ;;
    "id_loss")
        cd val
        python ID_loss.py --input_image_path "$2" --generated_folder "$3"
        ;;
    "lpips")
        cd val
        python LPIPS.py --driving_folder "$2" --generated_folder "$3"
        ;;
    "lse_c")
        cd val
        python LSE-C.py --generated_folder "$2"
        ;;
    "lse_d")
        cd val
        python LSE-D.py --generated_folder "$2" --driving_folder "$3"
        ;;
    "ssim")
        cd val
        python SSIM.py --generated_folder "$2" --driving_folder "$3"
        ;;
    "fid")
        cd val
        python -m pytorch_fid "$2" "$3"
        ;;
    "niqe")
        cd val
        python niqe-master/niqe.py --generated_folder "$2"
        ;;
    *)
        exec "$@"
        ;;
esac