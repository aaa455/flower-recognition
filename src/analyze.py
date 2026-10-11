"""检查花卉图片并生成统计表与图表。运行：python -m src.analyze。"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.download import CLASS_NAMES

IMAGE_COLUMNS = ['path', 'label', 'width', 'height', 'sha256']


def scan_images(root: Path):
    """返回可读图片表和损坏图片表，不修改或删除原始图片。"""
    missing = [name for name in CLASS_NAMES if not (root / name).is_dir()]
    if missing:
        raise ValueError(f'数据目录缺少类别：{missing}，请先下载数据并检查 --data 路径。')

    records, invalid = [], []
    for label in CLASS_NAMES:
        for path in sorted((root / label).rglob('*')):
            if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            relative_path = path.relative_to(root).as_posix()
            try:
                with Image.open(path) as image:
                    image.load()  # 实际解码图片，不能只根据扩展名判断它是否可用。
                    width, height = image.size
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                records.append({'path': relative_path, 'label': label, 'width': width,
                                'height': height, 'sha256': digest})
            except (OSError, ValueError) as error:
                invalid.append({'path': relative_path, 'label': label, 'error': str(error)})

    frame = pd.DataFrame(records, columns=IMAGE_COLUMNS)
    if frame.empty:
        raise ValueError('没有找到可读图片，请检查数据目录。')
    invalid_frame = pd.DataFrame(invalid, columns=['path', 'label', 'error'])
    return frame, invalid_frame


def find_duplicates(frame):
    """哈希相同代表文件字节相同；不同标签的相同文件属于标签冲突。"""
    duplicates = frame.loc[frame.duplicated('sha256', keep=False)].copy()
    labels_per_hash = frame.groupby('sha256')['label'].nunique()
    conflict_hashes = labels_per_hash.index[labels_per_hash > 1]
    conflicts = frame.loc[frame['sha256'].isin(conflict_hashes)].copy()
    return duplicates, conflicts


def save_plots(frame, counts, output):
    # 缓存放在项目内；使用无窗口模式，让脚本只生成图片文件。
    os.environ.setdefault('MPLCONFIGDIR', str(Path('runs/.mpl-cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts['label'], counts['count'])
    ax.set(title='Flower class distribution', ylabel='Number of images')
    for index, count in enumerate(counts['count']):
        ax.text(index, count, str(count), ha='center', va='bottom')
    fig.tight_layout()
    fig.savefig(output / 'class_distribution.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(frame['width'], frame['height'], alpha=0.3, s=8)
    ax.set(title='Image dimensions', xlabel='Width (pixels)', ylabel='Height (pixels)')
    fig.tight_layout()
    fig.savefig(output / 'image_sizes.png', dpi=150)
    plt.close(fig)


def analyze(root: Path, output: Path):
    frame, invalid = scan_images(root)
    duplicates, conflicts = find_duplicates(frame)
    counts = frame.groupby('label').size().reindex(CLASS_NAMES, fill_value=0)
    counts = counts.rename('count').reset_index()
    counts['proportion'] = counts['count'] / len(frame)
    # NumPy 分别按宽、高这两列统计；axis=0 表示沿图片这一维进行计算。
    dimensions = frame[['width', 'height']].to_numpy()
    summary = {
        'readable_images': len(frame), 'invalid_images': len(invalid),
        'duplicate_groups': int(duplicates['sha256'].nunique()),
        'images_in_duplicate_groups': len(duplicates),
        'conflicting_label_groups': int(conflicts['sha256'].nunique()),
        'images_with_conflicting_labels': len(conflicts),
        'mean_size': dict(zip(['width', 'height'], np.mean(dimensions, axis=0).tolist())),
        'median_size': dict(zip(['width', 'height'], np.median(dimensions, axis=0).tolist())),
    }
    output.mkdir(parents=True, exist_ok=True)
    tables = {'images.csv': frame, 'invalid_images.csv': invalid, 'class_counts.csv': counts,
              'duplicate_images.csv': duplicates, 'conflicting_labels.csv': conflicts}
    for name, table in tables.items():
        table.to_csv(output / name, index=False, encoding='utf-8-sig')
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    save_plots(frame, counts, output)
    print(counts.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'分析结果已保存至：{output}')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/flower_photos'), help='花卉图片目录')
    parser.add_argument('--output', type=Path, default=Path('data/analysis'), help='分析结果目录')
    args = parser.parse_args(argv)
    analyze(args.data, args.output)


if __name__ == '__main__':
    main()
