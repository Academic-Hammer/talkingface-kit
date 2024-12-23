# #!/usr/bin/python
# #-*- coding: utf-8 -*-

# import time, pdb, argparse, subprocess
# import glob
# import os
# from tqdm import tqdm
# from moviepy.editor import VideoFileClip
# from shutil import rmtree

# from LSE.SyncNetInstance_calc_scores import *

# def calculate_scores(video_path):
# 	# 视频切割
# 	def split_video(input_folder, output_folder, segment_duration=30):
# 		# 获取 /data/ 文件夹下的所有视频文件
# 		video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
		
# 		# 如果 /merge/ 文件夹不存在，则创建
# 		if os.path.exists(output_folder):
# 			rmtree(output_folder)
# 		os.makedirs(output_folder)

# 		for video_file in video_files:
# 			# print(video_file)
# 			# 构建输入视频文件的完整路径
# 			video_path = os.path.join(input_folder, video_file)
			
# 			# 使用 VideoFileClip 读取视频文件
# 			with VideoFileClip(video_path) as video:
# 				# 获取视频的总时长（秒）
# 				total_duration = video.duration
				
# 				# 计算分割的视频段数
# 				num_segments = int(total_duration // segment_duration) + 1
				
# 				# 对视频进行切割
# 				for i in range(num_segments):
# 					# 计算每个片段的起始时间和结束时间
# 					start_time = i * segment_duration
# 					end_time = min((i + 1) * segment_duration, total_duration)
					
# 					# 从视频中切割出指定时间段
# 					clip = video.subclip(start_time, end_time)
					
# 					# 构建输出视频文件的路径
# 					output_filename = f"{os.path.splitext(video_file)[0]}_segment_{i + 1:04d}.mp4"
# 					output_path = os.path.join(output_folder, output_filename)
					
# 					# 保存切割出的片段
# 					clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
					
# 					print(f"Saved {output_filename} to {output_folder}")

# 			# 只执行第一个视频
# 			break

# 	# ==================== LOAD PARAMS ====================


# 	parser = argparse.ArgumentParser(description = "SyncNet")

# 	parser.add_argument('--initial_model', type=str, default="./LSE/data/syncnet_v2.model", help='')
# 	parser.add_argument('--batch_size', type=int, default='20', help='')
# 	parser.add_argument('--vshift', type=int, default='15', help='')
# 	parser.add_argument('--data_input', type=str, default=f"{video_path}", help='')
# 	parser.add_argument('--data_root', type=str, default="./LSE/data/merge/", help='')
# 	parser.add_argument('--tmp_dir', type=str, default="./LSE/data/work/pytmp", help='')
# 	parser.add_argument('--reference', type=str, default="demo", help='')

# 	opt = parser.parse_args()


# 	# ==================== RUN EVALUATION ====================

# 	s = SyncNetInstance()

# 	s.loadParameters(opt.initial_model)

# 	split_video(opt.data_input,opt.data_root)

# 	#print("Model %s loaded."%opt.initial_model)
# 	merge_path = os.path.join(opt.data_root, "*.mp4")

# 	all_videos = glob.glob(merge_path)

# 	prog_bar = tqdm(range(len(all_videos)))
# 	avg_confidence = 0.
# 	avg_min_distance = 0.


# 	for videofile_idx in prog_bar:
# 		videofile = all_videos[videofile_idx]
# 		# print(videofile)
# 		offset, confidence, min_distance = s.evaluate(opt, videofile=videofile)
# 		avg_confidence += confidence
# 		avg_min_distance += min_distance
# 		prog_bar.set_description('Avg Confidence: {}, Avg Minimum Dist: {}'.format(round(avg_confidence / (videofile_idx + 1), 3), round(avg_min_distance / (videofile_idx + 1), 3)))
# 		prog_bar.refresh()

# 	print('Average Confidence: {}'.format(avg_confidence/len(all_videos)))
# 	print('Average Minimum Distance: {}'.format(avg_min_distance/len(all_videos)))

# 	return avg_confidence/len(all_videos), avg_min_distance/len(all_videos)

#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess
import glob
import os
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from shutil import rmtree

from LSE.SyncNetInstance_calc_scores import *

def calculate_scores(video_path):
	# 视频切割
	def split_video(input_video_path, output_dir='./data/merge', segment_duration=15):
		# 如果 /merge/ 文件夹不存在，则创建
		if os.path.exists(output_dir):
			rmtree(output_dir)
		os.makedirs(output_dir)

		# 打开视频文件
		video = VideoFileClip(input_video_path)

		# 获取视频的总时长
		video_duration = video.duration  # 视频总时长（秒）

		# 切割视频并保存
		segment_start = 0
		segment_count = 0

		while segment_start < video_duration:
			# 计算切割的结束时间
			segment_end = min(segment_start + segment_duration, video_duration)

			# 切割出一个片段
			video_segment = video.subclip(segment_start, segment_end)

			# 保存切割后的片段
			segment_filename = f"{output_dir}/segment_{segment_count:03d}.mp4"
			video_segment.write_videofile(segment_filename, codec='libx264')

			print(f"Saved segment {segment_count:03d} from {segment_start} to {segment_end}")
			
			# 更新切割的开始时间和片段计数器
			segment_start = segment_end
			segment_count += 1

		video.close()
		

	# ==================== LOAD PARAMS ====================

	class SyncNetConfig:
		def __init__(self, 
					initial_model="./LSE/data/syncnet_v2.model",
					batch_size=20, 
					vshift=15, 
					data_input=None, 
					data_root="./LSE/data/merge/", 
					tmp_dir="./LSE/data/work/pytmp", 
					reference="demo"):
			self.initial_model = initial_model
			self.batch_size = batch_size
			self.vshift = vshift
			self.data_input = data_input
			self.data_root = data_root
			self.tmp_dir = tmp_dir
			self.reference = reference

	# parser = argparse.ArgumentParser(description = "SyncNet")

	# parser.add_argument('--initial_model', type=str, default="./LSE/data/syncnet_v2.model", help='')
	# parser.add_argument('--batch_size', type=int, default='20', help='')
	# parser.add_argument('--vshift', type=int, default='15', help='')
	# parser.add_argument('--data_input', type=str, default=f"{video_path}", help='')
	# parser.add_argument('--data_root', type=str, default="./LSE/data/merge/", help='')
	# parser.add_argument('--tmp_dir', type=str, default="./LSE/data/work/pytmp", help='')
	# parser.add_argument('--reference', type=str, default="demo", help='')

	opt_new = SyncNetConfig(data_input=f"{video_path}")


	# ==================== RUN EVALUATION ====================

	s = SyncNetInstance()

	s.loadParameters(opt_new.initial_model)

	split_video(opt_new.data_input,opt_new.data_root)

	#print("Model %s loaded."%opt.initial_model)
	merge_path = os.path.join(opt_new.data_root, "*.mp4")

	all_videos = glob.glob(merge_path)

	prog_bar = tqdm(range(len(all_videos)))
	avg_confidence = 0.
	avg_min_distance = 0.


	for videofile_idx in prog_bar:
		videofile = all_videos[videofile_idx]
		# print(videofile)
		offset, confidence, min_distance = s.evaluate(opt_new, videofile=videofile)
		avg_confidence += confidence
		avg_min_distance += min_distance
		prog_bar.set_description('Avg Confidence: {}, Avg Minimum Dist: {}'.format(round(avg_confidence / (videofile_idx + 1), 3), round(avg_min_distance / (videofile_idx + 1), 3)))
		prog_bar.refresh()

	print(f"{video_path}:")
	print('Average Confidence: {}'.format(avg_confidence/len(all_videos)))
	print('Average Minimum Distance: {}'.format(avg_min_distance/len(all_videos)))

	return avg_confidence/len(all_videos), avg_min_distance/len(all_videos)

if __name__ == '__main__':
	video_paths = ['./data/input/Obama.mp4','./data/input/Jae-in.mp4','./data/input/Lieu.mp4','./data/input/Macron.mp4','./data/input/May.mp4',
				'./data/input/Shaheen.mp4']
	# 填入需要进行计算的video路径
	# for video_path in video_paths:
	video_path = './data/input/Lieu.mp4'
	calculate_scores(video_path)

