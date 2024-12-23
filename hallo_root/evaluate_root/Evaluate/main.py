import niqe, psnr_ssim, fid, lmd
import cv2
from skimage import io
from argparse import ArgumentParser
from inception import InceptionV3
import os
import subprocess

def calculate_FID(path1, path2):
    # 构建计算FID的解释器，用来传递计算所需的参数       
    parser = ArgumentParser()
    parser.add_argument('--path1', type=str, default=path1)
    parser.add_argument('--path2', type=str, default=path2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM))
    parser.add_argument('-c', '--gpu', default='0', type=str)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    FID = fid.calculate_fid_given_paths(args.path1,
                                            args.path2,
                                            args.batch_size,
                                            args.gpu != '',
                                            args.dims)
    return FID

def run_pipeline(videofile, reference, data_dir):
    # python run_pipeline.py --videofile /path/to/your/video --reference wav2lip --data_dir tmp_dir
    target_dir = "syncnet_python"
    command = [
        "python3", "run_pipeline.py", 
        "--videofile", videofile, 
        "--reference", reference, 
        "--data_dir", data_dir
    ]
    try:
        result = subprocess.run(command, cwd=target_dir, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
         print(e.stderr)


def calculate_LSE(videofile, reference, data_dir):
    # python calculate_scores_real_videos.py --videofile /path/to/you/video --reference wav2lip --data_dir tmp_dir >> all_scores.txt
    target_dir = "syncnet_python"
    command = [
        "python3", "calculate_scores_real_videos.py", 
        "--videofile", videofile, 
        "--reference", reference, 
        "--data_dir", data_dir,
    ]
    # result = subprocess.run(command, cwd=target_dir, check=True, capture_output=True, text=True)
    try:
        result = subprocess.run(command, cwd=target_dir, check=True, capture_output=True, text=True)
        scores = list(map(float, result.stdout.strip().split()))
        return scores
    except subprocess.CalledProcessError as e:
         print(e.stderr)
    return None


if __name__ == "__main__":
    example_video_name = 'Shaheen.mp4'
    print("The video for demonstration is " + example_video_name)
    
    example_source_video_path = '../MP4/Source'
    example_hallo_video_path = '../MP4/Hallo'

    example_source_video = cv2.VideoCapture(example_source_video_path + "/" + example_video_name)
    example_hallo_video = cv2.VideoCapture(example_hallo_video_path + "/" + example_video_name)

    example_FID_source_img_path = '../ImgsForFIDCalcu/source'
    example_FID_hallo_img_path = '../ImgsForFIDCalcu/hallo'
    
    params_path = 'pre-train-models/'   # 计算NIQE所需要的变量
    index = 0   # 帧序号

    niqe_source = 0.0
    niqe_hallo = 0.0
    PSNR = 0.0
    SSIM = 0.0

    if example_source_video.isOpened() and example_hallo_video.isOpened():
        rval_source, frame_source = example_source_video.read()  # 读取视频帧
        rval_hallo, frame_hallo = example_hallo_video.read()
    else:
        rval_source = False
        rval_hallo = False

    while rval_source and rval_hallo:
        if index == 30:
            break
        # 对视频的每一帧进行处理
        rval_source, frame_source = example_source_video.read()
        img_source = niqe.img_scissors(frame_source, 720, 512)  # 对源视频的帧图像进行尺寸统一处理
        rval_hallo, frame_hallo = example_hallo_video.read()
        img_hallo = frame_hallo
        if img_source is None or img_hallo is None:
            print("Loop End.")
            break
        else:
            cv2.imwrite(example_FID_source_img_path + "/" + str(index) + ".jpg", img_source)
            cv2.imwrite(example_FID_hallo_img_path + "/" + str(index) + ".jpg", img_hallo)
            
            #计算source与hallo的NIQE值
            niqe_source += niqe.calculate_niqe(img_source, crop_border=0, params_path=params_path)
            niqe_hallo += niqe.calculate_niqe(img_hallo, crop_border=0, params_path=params_path)
            #计算PSNR值
            PSNR += psnr_ssim.calculate_psnr(img_source, img_hallo, crop_border=0)
            #计算SSIM值
            SSIM += psnr_ssim.calculate_ssim(img_source, img_hallo, crop_border=0)
            
            print(index)
            index += 1
    
    niqe_source /= index
    niqe_hallo /= index

    PSNR /= index
    SSIM /= index

    FID = calculate_FID(example_FID_source_img_path, example_FID_hallo_img_path)
    
    run_pipeline("../" + example_source_video_path + "/" + example_video_name, "wav2lip", "tmp_dir")    
    scores_source = calculate_LSE("../" + example_source_video_path + "/" + example_video_name, "wav2lip", "tmp_dir")
    LSE_D_source = scores_source[0]
    LSE_C_source = scores_source[1]
    
    run_pipeline("../" + example_hallo_video_path + "/" + example_video_name, "wav2lip", "tmp_dir")    
    scores_hallo = calculate_LSE("../" + example_hallo_video_path + "/" + example_video_name, "wav2lip", "tmp_dir")
    LSE_D_hallo = scores_hallo[0]
    LSE_C_hallo = scores_hallo[1]
    
    LMD = lmd.compute_lmd(example_source_video_path + "/" + example_video_name, example_hallo_video_path + "/" + example_video_name)
    
    print("source NIQE: " + str(niqe_source) + ", hallo NIQE: " + str(niqe_hallo))
    print("PSNR: " + str(PSNR))
    print("SSIM: " + str(SSIM))
    print("FID: " + str(FID))
    print("source LSE_C: " + str(LSE_C_source) + ", hallo LSE_C: " + str(LSE_C_hallo))
    print("source LSE_D: " + str(LSE_D_source) + ", hallo LSE_D: " + str(LSE_D_hallo))
    print("LMD: " + str(LMD))
