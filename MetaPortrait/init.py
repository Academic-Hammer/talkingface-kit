import os
import shutil
from pathlib import Path

def copy_files(source_dir, target_dir):
    """
    复制文件到指定目录
    source_dir: Resources文件夹的路径
    target_dir: MetaPortrait项目根目录的路径
    """
    # 确保源目录和目标目录存在
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # 定义复制映射关系
    copy_mapping = {
        'ckpt_base.pth.tar': 'base_model/checkpoint/',
        'shape_predictor_68_face_landmarks.dat': 'val/',
        'vggface2.pt': 'val/',
        'GFPGANv1.3.pth': 'sr_model/pretrained_ckpt/',
        'temporal_gfpgan.pth': 'sr_model/pretrained_ckpt/',
        'HDTF_warprefine': 'sr_model/data/',
        'big-lama-20241220T131240Z-001.zip': 'Inpaint-Anything/pretrained_models/'
    }
    
    # 执行复制操作
    for source_name, target_path in copy_mapping.items():
        source_path = source_dir / source_name
        full_target_path = target_dir / target_path
        
        # 创建目标目录（如果不存在）
        os.makedirs(full_target_path, exist_ok=True)
        
        # 复制文件或目录
        try:
            if source_path.is_dir():
                # 如果是目录，使用copytree
                if (full_target_path / source_name).exists():
                    shutil.rmtree(full_target_path / source_name)
                shutil.copytree(source_path, full_target_path / source_name)
            else:
                # 如果是文件，使用copy2
                shutil.copy2(source_path, full_target_path)
            print(f"成功复制 {source_name} 到 {full_target_path}")
        except FileNotFoundError:
            print(f"错误：找不到源文件 {source_path}")
        except Exception as e:
            print(f"复制 {source_name} 时发生错误: {str(e)}")

if __name__ == "__main__":
    # 设置源目录和目标目录的路径
    # 脚本在MetaPortrait目录下运行
    current_dir = Path(os.getcwd())
    resources_dir = current_dir / "Resources"  # Resources在当前目录下
    
    # 执行复制
    copy_files(resources_dir, current_dir)