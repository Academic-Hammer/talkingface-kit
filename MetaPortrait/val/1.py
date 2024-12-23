# run_from_txt.py
with open("D:/data_process/新建文件夹/val/python_512.txt", "r", encoding='utf-8') as file:
    # 读取所有脚本路径并按行分割
    python_script_paths = file.read().strip().split('\n')

# 打印出所有读取的路径，检查它们是否正确
print("Python script paths:", python_script_paths)

for python_script_path in python_script_paths:
    # 检查每个路径是否存在，如果存在，则执行该脚本
    try:
        # print(f"Running script: {python_script_path}")
        with open(python_script_path, "r", encoding='utf-8') as script_file:
            exec(script_file.read())  # 执行该 Python 脚本
    except FileNotFoundError:
        print(f"Error: {python_script_path} not found!")
    except Exception as e:
        print(f"Error running {python_script_path}: {e}")
