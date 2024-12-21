#!/bin/bash
args=("$@")

if [ "$1" = '-cpbd' ]; then
    python cpbd.py "${args[@]:1}";
elif [ "$1" = '-ssim' ]; then
    python ssim.py "${args[@]:1}";
elif [ "$1" = '-fid' ]; then
    python fid.py "${args[@]:1}";
elif [ "$1" = '-niqe' ]; then
    python niqe.py "${args[@]:1}";
elif [ "$1" = '-psnr' ]; then
    python psnr.py "${args[@]:1}";
elif [ "$1" = '-lse' ]; then
    python lse.py "${args[@]:1}";
elif [ "$1" = '-list' ]; then
    echo 'Available arguments: -cpbd, -ssim, -fid, -niqe, -psnr, -lse. ';
    echo '-cpbd: CPBD (Color Preserving Brightness Distortion) metric';
    echo '    -cpbd --reference <your_reference_path>';
    echo '-niqe: NIQE (No Reference Image Quality Evaluator) metric';
    echo '    -niqe --reference <your_reference_path>';
    echo '-lse: LSE (Local Structural Similarity) metric';
    echo '    -lse --reference <your_reference_path>';
    echo '-ssim: SSIM (Structural Similarity Index Measure) metric';
    echo '    -ssim --origin <your_origin_path> --reference <your_reference_path>';
    echo '-psnr: PSNR (Peak Signal-to-Noise Ratio) metric';
    echo '    -psnr --origin <your_origin_path> --reference <your_reference_path>';
    echo '-fid: FID (Fréchet Inception Distance) metric';
    echo '    -fid --origin <your_origin_path> --reference <your_reference_path>';
    exit 1;
else 
    echo 'Invalid argument. Use one of the following: -cpbd, -ssim, -fid, -niqe, -psnr, -lse. Or use following to list all arguments: -list. '
    exit 1;
fi

