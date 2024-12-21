import subprocess
import argparse

def run_lspmodel(args):
    # 构建命令，包含 lspmodel 后面的所有参数
    command = ["python", "demo.py"] + args
    subprocess.run(command)

def run_eval(args):
    # 构建命令，先进入 judge_models 目录，然后运行 run_judge.py
    command = ["cd", "judge_models", "&&", "python", "run_judge.py"] + args
    subprocess.run(command, shell=True)

def main():
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="运行不同模式的脚本")
    
    # --lspmodel 和 --eval 是选项
    parser.add_argument("--lspmodel", action="store_true", help="运行 lspmodel 模式")
    parser.add_argument("--eval", action="store_true", help="运行 eval 模式")
    
    # 使用 parse_known_args 解析已知参数和多余参数
    args, unknown_args = parser.parse_known_args()

    # 根据 --lspmodel 或 --eval 执行不同的命令
    if args.lspmodel:
        run_lspmodel(unknown_args)
    elif args.eval:
        run_eval(unknown_args)
    else:
        print("Please specify --lspmodel or --eval to decide the mode.")

if __name__ == "__main__":
    main()
