import sys
import subprocess
import os
import argparse

def compute_L1_PSNR_SSIM_LPIPS(gt_video, gen_video):
    # 调用./L1_PSNR_SSIM_LPIPS目录下的eval.py文件
    try:
        from L1_PSNR_SSIM_LPIPS.eval import compute_all_metrics
        metrics = compute_all_metrics(gt_video, gen_video)
        return metrics
    except ImportError:
        print("Error: Could not import the function 'compute_all_metrics' from eval.py.")
        sys.exit(1)

def compute_FID(gt_video, gen_video):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    script_directory = os.path.join(script_directory, 'fid_tmp')
    try:
        from FID.fid_eval import compute_fid_for_videos
        value = compute_fid_for_videos(gt_video, gen_video, output_dir=script_directory)
        return value
    except ImportError:
        print("Error: Could not import the function 'compute_all_metrics' from eval.py.")
        sys.exit(1)

def compute_LSE(gen_video):
    # 调用./LSE-C-D目录下的judge_lse.py文件
    try:
        subprocess.run(['python', './LSE-C-D/judge_lse.py', gen_video], check=True)
        # 从结果文件中读取LSE-C和LSE-D
        with open('./LSE-C-D/all_scores.txt', 'r') as file:
            scores = file.readline().split()
            lse_c, lse_d = float(scores[0]), float(scores[1])
            return lse_c, lse_d
    except subprocess.CalledProcessError as e:
        print(f"Error during LSE calculation: {e}")
        sys.exit(1)

def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description="Evaluate generated video against ground truth.")

    # 添加参数
    parser.add_argument('--gt_video', type=str, required=True, help="Path to the ground truth video")
    parser.add_argument('--gen_video', type=str, required=True, help="Path to the generated video")

    # 解析命令行参数
    args = parser.parse_args()

    # 获取参数值
    gt_video = args.gt_video
    gen_video = args.gen_video
    
    # Step 1: 计算FID
    print("=================================================")
    print("[Computing] FID")
    fid = compute_FID(gt_video, gen_video)
    print("=================================================")
    print("[Finished]  FID")

    # Step 2: 计算L1、PSNR、SSIM、LPIPS
    print("=================================================")
    print("[Computing] L1 && PSNR && SSIM && LPIPS")
    metrics = compute_L1_PSNR_SSIM_LPIPS(gt_video, gen_video)
    print("=================================================")
    print("[Finished]  L1 && PSNR && SSIM && LPIPS")
    l1, psnr, ssim, lpips = metrics[0], metrics[1], metrics[2], metrics[3]    
    
    # Step 3: 计算LSE-C和LSE-D
    print("=================================================")
    print("[Computing] LSE-C && LSE-D")
    print("=================================================")
    print("This will take some time...")
    lse_c, lse_d = compute_LSE(gen_video)
    print("=================================================")
    print("[Finished]  LSE-C && LSE-D")

    # Step 3: 整合所有评估结果
    result = {
        "L1": l1,
        "PSNR": psnr,
        "SSIM": ssim,
        "LPIPS": lpips,
        "LSE-C": lse_c,
        "LSE-D": lse_d,
        "FID": fid
    }

    # 输出结果
    print("=================================================")
    print("Evaluation Results:")
    print("=================================================")
    for metric, value in result.items():
        print(f"{metric}: {value}")
    print("=================================================")
    return result

if __name__ == "__main__":
    main()