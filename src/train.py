"""训练手写 ResNet-18 并保存最佳验证模型。运行：python -m src.train --help。"""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import create_loaders
from src.model import ResNet18


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_epoch(model, loader, device, optimizer=None):
    """有优化器时训练，否则只验证；指标按实际处理的样本数加权。"""
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError('损失出现非有限数值，请检查数据和学习率。')
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += len(labels)
    if count == 0:
        raise ValueError('当前批次加载器未提供样本。')
    return {'loss': total_loss / count, 'accuracy': correct / count, 'samples': count}


def save_curves(history, path):
    os.environ.setdefault('MPLCONFIGDIR', str(Path('runs/.mpl-cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ['loss', 'accuracy']):
        for split in ['train', 'val']:
            ax.plot(history['epoch'], history[f'{split}_{metric}'], marker='o', label=split)
        ax.set(xlabel='Epoch', ylabel=metric, title=metric.title())
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train(args):
    if args.epochs < 1 or args.batch_size < 2 or args.image_size < 32:
        raise ValueError('epochs 至少为 1，batch-size 至少为 2，image-size 至少为 32。')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('学习率必须是大于 0 的有限数值。')
    if args.seed < 0 or args.seed >= 2 ** 32 or args.num_threads < 1:
        raise ValueError('seed 必须在 0 到 2**32-1 之间，num-threads 至少为 1。')
    if args.output.exists():
        raise ValueError(f'{args.output} 已存在，请选择新的 --output，避免覆盖原实验。')
    device_name = args.device
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('当前没有可用 CUDA，请使用 --device cpu 或安装匹配的 GPU 版本。')
    device = torch.device(device_name)
    set_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    manifest_bytes = args.manifest.read_bytes()
    loaders, datasets, stats = create_loaders(args.data, args.manifest, args.image_size,
                                             args.batch_size, limit_per_class=args.limit_per_class)
    # 小尺寸下，训练末批只有一张时，最深处 BatchNorm 可能仅剩一个值。
    # 只在这种情况下舍弃最后一张；验证时 eval 模式不受此限制。
    drop_singleton = len(datasets['train']) % args.batch_size == 1
    if drop_singleton:
        loaders['train'] = DataLoader(datasets['train'], batch_size=args.batch_size,
                                      shuffle=True, num_workers=0, drop_last=True)
    model = ResNet18(num_classes=len(stats['classes'])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({'resolved_device': device_name, 'optimizer': 'Adam', 'weight_decay': 0.0001,
                   'torch_version': str(torch.__version__), 'drop_singleton_train_batch': drop_singleton})
    args.output.mkdir(parents=True)
    (args.output / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'manifest.csv').write_bytes(manifest_bytes)
    print(f'设备：{device}；训练：{len(datasets["train"])} 张；验证：{len(datasets["val"])} 张', flush=True)
    if drop_singleton:
        print('每轮训练舍弃末批的单张图片，实际样本数会记录在 history.csv 中。', flush=True)
    records, best_accuracy = [], -1.0
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        training = run_epoch(model, loaders['train'], device, optimizer)
        validation = run_epoch(model, loaders['val'], device)
        record = {'epoch': epoch, **{f'train_{k}': v for k, v in training.items()},
                  **{f'val_{k}': v for k, v in validation.items()},
                  'seconds': time.perf_counter() - started}
        records.append(record)
        history = pd.DataFrame(records)
        history.to_csv(args.output / 'history.csv', index=False)
        if validation['accuracy'] > best_accuracy:
            best_accuracy = validation['accuracy']
            # 保存预处理和类别顺序，使之后的推理能够恢复相同设置。
            torch.save({'model_state': model.state_dict(), **stats, 'config': config,
                        'epoch': epoch, 'val_accuracy': best_accuracy,
                        'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest()}, args.output / 'best.pt')
        save_curves(history, args.output / 'training_curves.png')
        print(f'第 {epoch}/{args.epochs} 轮：训练 loss={training["loss"]:.4f} acc={training["accuracy"]:.2%}；'
              f'验证 loss={validation["loss"]:.4f} acc={validation["accuracy"]:.2%}；'
              f'耗时 {record["seconds"]:.1f} 秒', flush=True)
    print(f'最佳验证准确率：{best_accuracy:.2%}；模型：{args.output / "best.pt"}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/flower_photos'))
    parser.add_argument('--manifest', type=Path, default=Path('data/splits/manifest.csv'))
    parser.add_argument('--output', type=Path, default=Path('runs/train'))
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--image-size', type=int, default=96)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--limit-per-class', type=int, help='每类训练图片上限，默认使用全部训练集')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-threads', type=int, default=4)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    train(parser.parse_args(argv))


if __name__ == '__main__':
    main()
