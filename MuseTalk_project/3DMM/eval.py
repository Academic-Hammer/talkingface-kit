import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm, trange
import torch
import face_alignment
import deep_3drecon
from moviepy.editor import VideoFileClip
import copy
import psutil
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, network_size=4, device='cuda')
face_reconstructor = deep_3drecon.Reconstructor()

# landmark detection in Deep3DRecon
def lm68_2_lm5(in_lm):
    # in_lm: shape=[68,2]
    lm_idx = np.array([31,37,40,43,46,49,55]) - 1
    # 将上述特殊角点的数据取出，得到5个新的角点数据，拼接起来。
    lm = np.stack([in_lm[lm_idx[0],:],np.mean(in_lm[lm_idx[[1,2]],:],0),np.mean(in_lm[lm_idx[[3,4]],:],0),in_lm[lm_idx[5],:],in_lm[lm_idx[6],:]], axis = 0)
    # 将第一个角点放在了第三个位置
    lm = lm[[1,2,0,3,4],:2]
    return lm

def process_video(fname, out_name=None):
    assert fname.endswith(".mp4")
    print(fname)
    if out_name is None:
        out_name = fname[:-4] + '.npy'
    tmp_name = out_name[:-4] + '.doi'
    if os.path.exists(out_name):
        print("out exisit, skip")
        return
    os.system(f"touch {tmp_name}")
    cap = cv2.VideoCapture(fname)
    print(f"loading video ...")
    
    # cap.subclip()
    
    # 获取视频相关参数
    num_frames = int(cap.get(7)) 
    h = int(cap.get(4))
    w = int(cap.get(3))
    # 检测系统资源是否充足
    mem = psutil.virtual_memory()
    a_mem = mem.available
    min_mem=num_frames*68*2 + num_frames*5*2 + num_frames*h*w*3
    if a_mem < min_mem:
        print(f"WARNING: The physical memory is insufficient, which may result in memory swapping. Available Memory: {a_mem/1000000:.3f}M, the minimum memory required is:{min_mem/1000000:.3f}M.")
    # 初始化矩阵
    lm68_arr=np.empty((num_frames, 68, 2),dtype=np.float32)
    lm5_arr=np.empty((num_frames, 5, 2),dtype=np.float32)
    video_rgb=np.empty((num_frames, h, w, 3),dtype=np.uint8)
    cnt=0
    i=0
    while cap.isOpened():
        i += 1
        ret, frame_bgr = cap.read()
        if frame_bgr is None:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # if fa.get_landmarks(frame_rgb) is None:
            # continue
        video_rgb[cnt]=frame_rgb
        cnt += 1
    
    # print("rua")
    # num_frames = cnt
    for i in trange(num_frames, desc="extracting 2D facial landmarks ..."):
        
        try:
            lm68 = fa.get_landmarks(video_rgb[i])[0] # 识别图片中的人脸，获得角点, shape=[68,2]
        except:
            print(f"WARNING: Caught errors when fa.get_landmarks, maybe No face detected at frame {i} in {fname}!")
            raise ValueError("")
        # lm68 = fa.get_landmarks(video_rgb[i]) # 识别图片中的人脸，获得角点, shape=[68,2]
        
        # if lm68 is None:
        #     lm68_arr[i]=lm68
        #         lm5_arr[i]=lm5
        #     continue
        
        # lm68 = lm68[0]
        lm5 = lm68_2_lm5(lm68)
        lm68_arr[i]=lm68
        lm5_arr[i]=lm5
    batch_size = 32
    iter_times = num_frames // batch_size
    last_bs = num_frames % batch_size
    coeff_lst = []
    for i_iter in range(iter_times):
        start_idx = i_iter * batch_size
        batched_images = video_rgb[start_idx: start_idx + batch_size]
        batched_lm5 = lm5_arr[start_idx: start_idx + batch_size]
        coeff, align_img = face_reconstructor.recon_coeff(batched_images, batched_lm5, return_image = True)
        coeff_lst.append(coeff)
    # print(last_bs)
    if last_bs != 0:
        batched_images = video_rgb[-last_bs:]
        batched_lm5 = lm5_arr[-last_bs:]
        coeff, align_img = face_reconstructor.recon_coeff(batched_images, batched_lm5, return_image = True)
        coeff_lst.append(coeff)
    return lm68_arr


def split_wav(mp4_name):
    wav_name = mp4_name[:-4] + '.wav'
    if os.path.exists(wav_name):
        return
    video = VideoFileClip(mp4_name,verbose=False)
    dur = video.duration
    audio = video.audio 
    assert audio is not None
    audio.write_audiofile(wav_name,fps=16000,verbose=False,logger=None)

if __name__ == '__main__':
    ### Process Single Long video for NeRF dataset
    # video_id = 'May'
    # video_fname = f"data/raw/videos/{video_id}.mp4"
    # out_fname = f"data/processed/videos/{video_id}/coeff.npy"
    # process_video(video_fname, out_fname)

    ### Process short video clips for LRS3 dataset
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--gt_path', type=str, default='Shaheen.mp4', help='')
    parser.add_argument('--infer_path', type=str, default='Shaheen_30_Shaheen.mp4', help='')
    parser.add_argument('--process_id', type=int, default=0, help='')
    parser.add_argument('--total_process', type=int, default=1, help='')
    args = parser.parse_args()

    import os, glob
    mp4_name1 = args.gt_path
    mp4_name2 = args.infer_path
    lm_68_1 = process_video(mp4_name1)
    lm_68_2 = process_video(mp4_name2)
    if len(lm_68_1) != len(lm_68_2):
        print("Warning: the frame of videos is not equal.")
    Len = min(len(lm_68_1),len(lm_68_2))
    loss = 0
    for i in range(Len):
        frame1 = lm_68_1[i]
        frame2 = lm_68_2[i]
        sum = 0
        for j in range(68):
            l = 1.0
            if j >= 48:
                l = 1.2
            sum += l*((frame1[j][0]-frame2[j][0])**2 + (frame1[j][1]-frame2[j][1])**2)
        sum += 5
        loss += math.log10(sum)
            
    loss /= Len
    print("Loss:",loss)
