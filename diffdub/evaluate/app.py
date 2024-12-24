import argparse
import os

import FID
import LMD
import NIQE
import calculate_scores_real_videos
import evaluate
import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="evaluate the output video")
    parser.add_argument("--generated_video", type=str, required=True, help="path to the generated video")
    parser.add_argument("--original_video", type=str, required=True,  help="path to the original video")
    parser.add_argument("--temp_dir", type=str, default="temp", help="path to the temporary directory")
    parser.add_argument("--output_file", type=str, default="", help="path to the output file, default is stdout")

    args = parser.parse_args()

    os.makedirs(args.temp_dir, exist_ok=True)

    print("Evaluating the generated video...")

    ssim, lpips, psnr = evaluate.main(args.original_video, args.generated_video, args.temp_dir)
    fid = FID.main(args.original_video, args.generated_video, args.temp_dir)
    niqe = NIQE.main(args.original_video, args.generated_video, args.temp_dir)
    lmd = LMD.main(args.original_video, args.generated_video, args.temp_dir)

    # lse
    opt = argparse.Namespace()

    setattr(opt, "data_dir", args.temp_dir)
    setattr(opt, "videofile", args.generated_video)
    setattr(opt, "reference", "wav2lip")
    setattr(opt, "facedet_scale", 0.25)
    setattr(opt, "crop_scale", 0.40)
    setattr(opt, "min_track", 100)
    setattr(opt, "frame_rate", 25)
    setattr(opt, "num_failed_det", 25)
    setattr(opt, "min_face_size", 100)

    run_pipeline.main(opt)

    opt1 = argparse.Namespace()
    setattr(opt1, "data_dir", args.temp_dir)
    setattr(opt1, "videofile", args.generated_video)
    setattr(opt1, "reference", "wav2lip")
    setattr(opt1, "initial_model", "data/syncnet_v2.model")
    setattr(opt1, "batch_size", 20)
    setattr(opt1, "vshift", 15)

    res = calculate_scores_real_videos.main(opt1)
    assert len(res) == 1

    print("Evaluation completed.")

    output: str = args.output_file
    if len(output) == 0:
        print("SSIM:", ssim)
        print("LPIPS:", lpips)
        print("PSNR:", psnr)
        print("FID:", fid)
        print("NIQE:", niqe)
        print("LMD:", lmd)
        print("LSE-C:", res[0][0])
        print("LSE-D:", res[0][1])
    else:
        with open(output, "a+", encoding='utf-8') as f:
            f.write(f"SSIM: {ssim}\n")
            f.write(f"LPIPS: {lpips}\n")
            f.write(f"PSNR: {psnr}\n")
            f.write(f"FID: {fid}\n")
            f.write(f"NIQE: {niqe}\n")
            f.write(f"LMD: {lmd}\n")
            f.write(f"LSE-C: {res[0][0]}\n")
            f.write(f"LSE-D: {res[0][1]}\n")
        print(f"Results saved to {output}")

if __name__ == "__main__":
    main()








