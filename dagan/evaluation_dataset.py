import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SelfReenactmentDataset(Dataset):
    def __init__(self, dataroot, pairs_list, transform=None, max_samples=None):
        self.dataroot = dataroot

        # 读取CSV文件
        with open(pairs_list, 'r') as f:
            lines = f.readlines()[1:]  # 跳过header
            self.pairs = [line.strip().split(',') for line in lines]

        if max_samples is not None:
            self.pairs = self.pairs[:max_samples]

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]) if transform is None else transform

    def __getitem__(self, idx):
        source_path = os.path.join(self.dataroot, self.pairs[idx][0].strip())
        driving_path = os.path.join(self.dataroot, self.pairs[idx][1].strip())

        source = Image.open(source_path).convert('RGB')
        driving = Image.open(driving_path).convert('RGB')

        if self.transform:
            source = self.transform(source)
            driving = self.transform(driving)

        return {
            'source_self': source,
            'driving': driving,
            'source_path': source_path,
            'driving_path': driving_path
        }

    def __len__(self):
        return len(self.pairs)


class EvaluationDataset(Dataset):
    def __init__(self, dataroot, pairs_list, transform=None, max_samples=None):
        self.dataroot = dataroot

        # 读取CSV文件
        with open(pairs_list, 'r') as f:
            lines = f.readlines()[1:]  # 跳过header
            self.pairs = [line.strip().split(',') for line in lines]

        if max_samples is not None:
            self.pairs = self.pairs[:max_samples]

        # 检查生成图像列是否存在
        self.has_generated = len(self.pairs[0]) >= 3

        # 过滤掉没有生成图像的样本（如果有）
        if self.has_generated:
            self.pairs = [pair for pair in self.pairs if len(pair) >= 3 and pair[2].strip()]

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]) if transform is None else transform

    def __getitem__(self, idx):
        source_path = os.path.join(self.dataroot, self.pairs[idx][0].strip())
        driving_path = os.path.join(self.dataroot, self.pairs[idx][1].strip())

        print(f"加载图像索引 {idx}：")
        print(f"  源图像路径: {source_path}")
        print(f"  驱动图像路径: {driving_path}")

        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"源图像文件不存在: {source_path}")
        if not os.path.isfile(driving_path):
            raise FileNotFoundError(f"驱动图像文件不存在: {driving_path}")

        source = Image.open(source_path).convert('RGB')
        driving = Image.open(driving_path).convert('RGB')

        if self.has_generated:
            generated_path = self.pairs[idx][2].strip()
            print(f"  生成图像路径: {generated_path}")
            if not os.path.isfile(generated_path):
                raise FileNotFoundError(f"生成图像文件不存在: {generated_path}")
            generated = Image.open(generated_path).convert('RGB')
        else:
            generated = None
            generated_path = None

        if self.transform:
            source = self.transform(source)
            driving = self.transform(driving)
            if generated:
                generated = self.transform(generated)

        return {
            'source_self': source,
            'driving': driving,
            'generated': generated,
            'source_path': source_path,
            'driving_path': driving_path,
            'generated_path': generated_path
        }

    def __len__(self):
        return len(self.pairs)