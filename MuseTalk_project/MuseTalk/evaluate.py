from scripts import inference
import argparse
import os
import cv2
import skimage
from PIL import Image
import numpy as np
from metrics.niqe import niqe
from skimage.metrics import peak_signal_noise_ratio
from metrics.fid import fid
from skimage.metrics import structural_similarity
import subprocess
import re
from tqdm import tqdm

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--inference_config", type=str, default="configs/inference/test_img.yaml")
	parser.add_argument("--bbox_shift", type=int, default=0)
	parser.add_argument("--result_dir", default='./results', help="path to output")

	parser.add_argument("--fps", type=int, default=25)
	parser.add_argument("--batch_size", type=int, default=8)
	parser.add_argument("--output_vid_name", type=str, default=None)
	parser.add_argument("--use_saved_coord",
                        action="store_true",
                        help='use saved coordinate to save time')
	parser.add_argument("--use_float16",
                        action="store_true",
                        help="Whether use float16 to speed up inference",
	)
	parser.add_argument("--input_image",default="data/video/yongen.mp4")
	parser.add_argument("--input_audio",default="data/audio/yongen.wav")
	parser.add_argument("--output_dir",default="./result")
	parser.add_argument("--ground_truth",default=None)

	args = parser.parse_args()
	result=args.output_dir
	input_basename = os.path.basename(args.input_image).split('.')[0]
	audio_basename  = os.path.basename(args.input_audio).split('.')[0]
	output_basename = f"{input_basename}_{audio_basename}"
	video_path=os.path.join(args.output_dir,output_basename+'.mp4')
	frame_dir=os.path.join(args.output_dir,output_basename)
	truth_dir=os.path.join(args.output_dir,output_basename+'_truth')
	inference.main(args)
	
	os.makedirs(frame_dir,exist_ok=True)
	os.makedirs(truth_dir,exist_ok=True)
	capture=cv2.VideoCapture(video_path)
	if args.ground_truth==None:
		capture_t=cv2.VideoCapture(args.input_image)
	else:
		print("ground_truth")
		capture_t=cv2.VideoCapture(args.ground_truth)
	real,gen=[],[]
	cnt=0
	while True:
		ret,img=capture.read()
		ret_t,img_t=capture_t.read()
		
		if (not ret) or (not ret_t):
			break
		
		cv2.imwrite(f"{frame_dir}/{str(cnt).zfill(8)}.png",img)
		cv2.imwrite(f"{truth_dir}/{str(cnt).zfill(8)}.png",img_t)
		cnt=cnt+1
		
		img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		img_t=cv2.cvtColor(img_t,cv2.COLOR_BGR2RGB)
		gen.append(img)
		real.append(img_t)
	
	FID=fid(truth_dir,frame_dir)
	print(FID)
	# def extract_floats(text):
	# 	return [float(s) for s in re.findall(r'-?\d+\.\d+', text)]
	# subprocess.call(['python','syncnet/run_pipeline.py','--videofile',video_path,"--reference",'wav2lip','--data_dir','tmp_dir'])
	# result=subprocess.check_output(['python','syncnet/calculate_scores_real_videos.py','--videofile',video_path,'--reference','wav2lip','--data_dir','tmp_dir'])
	# result=result.decode('utf-8')
	# floats=extract_floats(result)
	# LSE_D,LSE_C=floats[0],floats[1]
	# print(LSE_D,LSE_C)

	NIQE,PSNR,SSIM=0,0,0
	for i in tqdm(range(len(gen))):
		img,img_t=gen[i],real[i]
		image = Image.fromarray(img)
		image_t=Image.fromarray(img_t)
		image_g = np.array(image.convert('LA'))[:, :, 0]
		N,P,S=niqe(image_g),peak_signal_noise_ratio(img_t/255.0,img/255.0),structural_similarity(img,img_t,channel_axis=2)
		#print(f'{i}: {N} {P} {S}')
		NIQE=NIQE+N
		PSNR=PSNR+P #DB
		SSIM=SSIM+S
	cnt=len(gen)
	NIQE,PSNR,SSIM=NIQE/cnt,PSNR/cnt,SSIM/cnt

	result_file=f'evaluation/{output_basename}.txt'
	with open(result_file,'w') as file:
		file.write(f'NIQE: {NIQE}\n')
		file.write(f'PSNR: {PSNR}\n')
		file.write(f'FID: {FID}\n')
		file.write(f'SSIM: {SSIM}\n')

