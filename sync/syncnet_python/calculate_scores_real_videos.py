#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess, pickle, os, gzip, glob

from SyncNetInstance_calc_scores import *

# ==================== PARSE ARGUMENT ====================

parser = argparse.ArgumentParser(description = "SyncNet");
parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--data_dir', type=str, default='data/work', help='');
parser.add_argument('--videofile', type=str, default='', help='');
parser.add_argument('--reference', type=str, default='', help='');
opt = parser.parse_args();

setattr(opt,'avi_dir',os.path.join(opt.data_dir,'pyavi'))
setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
setattr(opt,'work_dir',os.path.join(opt.data_dir,'pywork'))
setattr(opt,'crop_dir',os.path.join(opt.data_dir,'pycrop'))


# ==================== LOAD MODEL AND FILE LIST ====================

s = SyncNetInstance();

s.loadParameters(opt.initial_model);
#print("Model %s loaded."%opt.initial_model);

flist = glob.glob(os.path.join(opt.crop_dir,opt.reference,'0*.avi'))
flist.sort()

# ==================== GET OFFSETS ====================
import logging

# 配置 logging，输出到 output.log 文件
logging.basicConfig(filename='evaluate.log', 
                    level=logging.INFO,  # 设置日志级别为 INFO
                    format='%(message)s', 
                    filemode='a')  # 使用追加模式，确保日志内容不会被覆盖

# 假设这是你的代码
dists = []
for idx, fname in enumerate(flist):
    offset, conf, dist = s.evaluate(opt, videofile=fname)
    # 使用 logging 输出信息，而不是 print
    logging.info(f"LSE-C: {dist} LSE-D: {conf}")  # 这将把信息追加到 output.log 中
    
# ==================== PRINT RESULTS TO FILE ====================

#with open(os.path.join(opt.work_dir,opt.reference,'activesd.pckl'), 'wb') as fil:
#    pickle.dump(dists, fil)
