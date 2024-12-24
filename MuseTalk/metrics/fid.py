import subprocess
import re

def extract_floats(text):
    return [float(s) for s in re.findall(r'-?\d+\.\d+', text)]

def fid(real,gen):
    result=subprocess.check_output(['python','pytorch_fid/fid_score.py',real,gen,'--device','cuda:0'])
    result=result.decode('utf-8')
    floats=extract_floats(result)
    return floats[0]