"""清洗花卉图片记录并划分数据集。运行：python -m src.prepare。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analyze import find_duplicates, scan_images
from src.download import CLASS_NAMES


def clean_images(frame):
    """排除整个标签冲突组；同标签重复文件按路径排序后保留第一份。"""
    ordered = frame.sort_values('path').reset_index(drop=True)
    _, conflicts = find_duplicates(ordered)
    without_conflicts = ordered.loc[~ordered['sha256'].isin(conflicts['sha256'])].copy()
    duplicate_mask = without_conflicts.duplicated('sha256', keep='first')
    removed_duplicates = without_conflicts.loc[duplicate_mask].copy()
    clean = without_conflicts.loc[~duplicate_mask].reset_index(drop=True)
    return clean, conflicts, removed_duplicates


def split_images(frame, seed=42):
    """每个类别独立打乱，约 70% 训练、15% 验证、15% 测试。"""
    if seed < 0:
        raise ValueError('随机种子必须是非负整数。')
    if frame['sha256'].duplicated().any():
        raise ValueError('划分前必须先去除重复文件和标签冲突。')
    if not frame['label'].isin(CLASS_NAMES).all():
        raise ValueError('发现未知类别，请检查标签。')

    manifest = frame.sort_values('path').reset_index(drop=True).copy()
    manifest['split'] = ''
    rng = np.random.default_rng(seed)
    for label in CLASS_NAMES:
        indices = manifest.index[manifest['label'] == label].to_numpy()
        if len(indices) < 7:
            raise ValueError(f'{label} 清洗后仅有 {len(indices)} 张图片，至少需要 7 张。')
        indices = rng.permutation(indices)
        # 向下取整；剩余样本全部归入训练集，因此各类比例略有差异。
        n_val = max(1, int(len(indices) * 0.15))
        n_test = max(1, int(len(indices) * 0.15))
        manifest.loc[indices[:n_val], 'split'] = 'val'
        manifest.loc[indices[n_val:n_val + n_test], 'split'] = 'test'
        manifest.loc[indices[n_val + n_test:], 'split'] = 'train'
    return manifest


def prepare(root: Path, output: Path, seed=42):
    # 重新检查原图，避免使用已经过期的分析报告。
    readable, invalid = scan_images(root)
    clean, conflicts, removed_duplicates = clean_images(readable)
    manifest = split_images(clean, seed)
    counts = pd.crosstab(manifest['label'], manifest['split'])
    counts = counts.reindex(index=CLASS_NAMES, columns=['train', 'val', 'test'], fill_value=0)
    summary = {
        'seed': seed,
        'readable_images': len(readable),
        'invalid_images': len(invalid),
        'conflicting_label_images_excluded': len(conflicts),
        'same_label_duplicates_removed': len(removed_duplicates),
        'clean_images': len(manifest),
        'classes': list(CLASS_NAMES),
        'class_to_index': {label: index for index, label in enumerate(CLASS_NAMES)},
        'split_counts': {split: int(counts[split].sum()) for split in counts.columns},
    }
    # 清洗与划分验证成功后才输出，原始图片不移动、不删除。
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        'manifest.csv': manifest,
        'invalid_images.csv': invalid,
        'excluded_conflicts.csv': conflicts,
        'removed_duplicates.csv': removed_duplicates,
    }
    for name, table in tables.items():
        table.to_csv(output / name, index=False, encoding='utf-8-sig')
    counts.to_csv(output / 'class_counts.csv', encoding='utf-8-sig')
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(counts.to_string())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'清洗与划分记录已保存至：{output}')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/flower_photos'), help='花卉图片目录')
    parser.add_argument('--output', type=Path, default=Path('data/splits'), help='划分记录目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，默认 42')
    args = parser.parse_args(argv)
    prepare(args.data, args.output, args.seed)


if __name__ == '__main__':
    main()
