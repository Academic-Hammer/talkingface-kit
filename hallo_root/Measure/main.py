import niqe, psnr_ssim, fid
import cv2

example_source_video_path = '../MP4/Source'
example_hallo_video_path = '../MP4/Hallo'

example_source_video = cv2.VideoCapture(example_source_video_path + "/Jae-in.mp4")
example_hallo_video = cv2.VideoCapture(example_hallo_video_path + "/Jae-in.mp4")

example_FID_source_img_path = '../ImgsForFIDCalcu/Jae-in/source'
example_FID_hallo_img_path = '../ImgsForFIDCalcu/Jae-in/hallo'

# 输出示例的NIQE值
print(niqe.NIQE(example_source_video, example_hallo_video))
 
#