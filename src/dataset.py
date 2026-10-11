"""图片预处理与 PyTorch 数据集。运行：python -m src.dataset --help。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.download import CLASS_NAMES


def calculate_statistics(root: Path, manifest: pd.DataFrame, image_size=32):
    """只用训练集计算 RGB 均值和标准差，避免验证集和测试集参与预处理参数。"""
    if image_size < 1:
        raise ValueError('统计图片尺寸必须大于 0。')
    train_rows = manifest.loc[manifest['split'] == 'train']
    if train_rows.empty:
        raise ValueError('训练集为空，无法计算归一化参数。')
    total = np.zeros(3, dtype=np.float64)
    squared_total = np.zeros(3, dtype=np.float64)
    pixels = 0
    for relative_path in train_rows['path']:
        with Image.open(root / relative_path) as image:
            array = np.asarray(image.convert('RGB').resize((image_size, image_size)), dtype=np.float64) / 255.0
        total += array.sum(axis=(0, 1))
        squared_total += (array ** 2).sum(axis=(0, 1))
        pixels += array.shape[0] * array.shape[1]
    mean = total / pixels
    std = np.sqrt(np.maximum(squared_total / pixels - mean ** 2, 1e-8))
    return mean.tolist(), std.tolist()


def make_transform(image_size, mean, std, training=False):
    """建立图片变换：训练集增加随机翻转，其他集合只做确定性变换。"""
    if image_size < 32:
        raise ValueError('模型输入尺寸至少为 32。')
    if len(mean) != 3 or len(std) != 3 or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError('mean 和 std 必须包含三个有限的 RGB 通道数值。')
    if any(value <= 0 for value in std):
        raise ValueError('标准差必须大于 0。')
    operations = [transforms.Resize((image_size, image_size))]
    if training:
        operations.append(transforms.RandomHorizontalFlip(p=0.5))
    operations.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(operations)


class FlowerDataset(Dataset):
    """根据 manifest 读取一个集合的图片和整数标签。"""

    def __init__(self, root: Path, manifest: Path | pd.DataFrame, split: str,
                 image_size=96, mean=None, std=None, limit_per_class=None):
        self.root = Path(root)
        self.frame = pd.read_csv(manifest, encoding='utf-8-sig') if isinstance(manifest, (str, Path)) else manifest.copy()
        if split not in {'train', 'val', 'test'}:
            raise ValueError('split 必须是 train、val 或 test。')
        self.frame = self.frame.loc[self.frame['split'] == split].copy()
        if limit_per_class is not None:
            if limit_per_class < 1:
                raise ValueError('每类样本上限必须大于 0。')
            self.frame = self.frame.groupby('label', sort=False).head(limit_per_class)
        self.frame = self.frame.reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f'{split} 集合为空。')
        self.classes = list(CLASS_NAMES)
        self.class_to_index = {name: index for index, name in enumerate(self.classes)}
        if not self.frame['label'].isin(self.classes).all():
            raise ValueError('manifest 中包含未知类别。')
        if mean is None or std is None:
            raise ValueError('必须传入只由训练集计算的 mean 和 std。')
        self.transform = make_transform(image_size, mean, std, split == 'train')

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.root / row['path']) as image:
            tensor = self.transform(image.convert('RGB'))
        label = self.class_to_index[row['label']]
        return tensor, label


def create_loaders(root: Path, manifest: Path, image_size=96, batch_size=32,
                   num_workers=0, limit_per_class=None):
    """计算训练统计量，并创建训练、验证、测试三个 DataLoader。"""
    if batch_size < 1:
        raise ValueError('batch_size 必须大于 0。')
    if num_workers < 0:
        raise ValueError('num_workers 不能为负数。')
    frame = pd.read_csv(manifest, encoding='utf-8-sig')
    mean, std = calculate_statistics(root, frame)
    datasets = {
        split: FlowerDataset(root, frame, split, image_size, mean, std,
                             limit_per_class if split == 'train' else None)
        for split in ['train', 'val', 'test']
    }
    loaders = {
        split: DataLoader(dataset, batch_size=batch_size, shuffle=split == 'train',
                          num_workers=num_workers, pin_memory=torch.cuda.is_available())
        for split, dataset in datasets.items()
    }
    return loaders, datasets, {'mean': mean, 'std': std, 'classes': list(CLASS_NAMES)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/flower_photos'))
    parser.add_argument('--manifest', type=Path, default=Path('data/splits/manifest.csv'))
    parser.add_argument('--image-size', type=int, default=96)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args(argv)
    loaders, datasets, stats = create_loaders(args.data, args.manifest, args.image_size, args.batch_size)
    images, labels = next(iter(loaders['train']))
    print(json.dumps({'classes': stats['classes'], 'mean': stats['mean'], 'std': stats['std'],
                      'dataset_sizes': {key: len(value) for key, value in datasets.items()},
                      'batch_shape': list(images.shape), 'label_shape': list(labels.shape),
                      'label_range': [int(labels.min()), int(labels.max())]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
