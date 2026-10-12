"""加载最佳模型并评估测试集。运行：python -m src.evaluate --help。"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import FlowerDataset
from src.download import CLASS_NAMES
from src.model import ResNet18


def load_checkpoint(path: Path, device):
    """重建网络，恢复类别顺序与预处理参数，并切换到评估模式。"""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint['classes'] != list(CLASS_NAMES):
        raise ValueError('模型的类别顺序与当前数据加载器不一致。')
    model = ResNet18(num_classes=len(checkpoint['classes'])).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, checkpoint


def verify_manifest(path: Path, expected_hash):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError('数据清单与训练时不一致，请使用训练目录中保存的 manifest.csv。')


def predict_batches(model, loader, device):
    """按加载器顺序预测全部样本，不修改参数或运行统计量。"""
    model.eval()
    true_labels, predictions, scores = [], [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            if not torch.isfinite(logits).all():
                raise ValueError('模型输出出现非有限数值，无法评估。')
            confidence, indices = logits.softmax(dim=1).max(dim=1)
            true_labels.extend(labels.tolist())
            predictions.extend(indices.cpu().tolist())
            scores.extend(confidence.cpu().tolist())
    if not true_labels:
        raise ValueError('测试加载器没有提供样本。')
    return true_labels, predictions, scores


def calculate_metrics(frame, classes):
    """矩阵行是真实类别，列是预测类别；无预测或无样本时相应指标记为 0。"""
    if frame.empty:
        raise ValueError('没有可评估的预测记录。')
    if not frame['true'].isin(classes).all() or not frame['predicted'].isin(classes).all():
        raise ValueError('预测记录中存在未知类别。')
    matrix = pd.crosstab(frame['true'], frame['predicted'])
    matrix = matrix.reindex(index=classes, columns=classes, fill_value=0)
    rows = []
    for label in classes:
        correct = int(matrix.loc[label, label])
        support = int(matrix.loc[label].sum())
        predicted_count = int(matrix[label].sum())
        precision = correct / predicted_count if predicted_count else 0.0
        recall = correct / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({'label': label, 'support': support, 'precision': precision,
                     'recall': recall, 'f1': f1})
    report = pd.DataFrame(rows)
    metrics = {'test_samples': len(frame), 'accuracy': float((frame['true'] == frame['predicted']).mean()),
               'macro_f1': float(report['f1'].mean())}
    return metrics, matrix, report


def save_matrix_plot(matrix, path):
    os.environ.setdefault('MPLCONFIGDIR', str(Path('runs/.mpl-cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    plot = ax.imshow(matrix.to_numpy(), cmap='Blues')
    classes = matrix.index.tolist()
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes,
           yticklabels=classes, xlabel='Predicted class', ylabel='True class', title='Test confusion matrix')
    threshold = matrix.to_numpy().max() / 2
    for row in range(len(classes)):
        for col in range(len(classes)):
            count = int(matrix.iloc[row, col])
            ax.text(col, row, str(count), ha='center', va='center',
                    color='white' if count > threshold else 'black')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    fig.colorbar(plot, ax=ax, label='Number of images')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(args):
    if args.batch_size < 1 or args.num_threads < 1:
        raise ValueError('batch-size 和 num-threads 必须大于 0。')
    output = args.output if args.output is not None else args.checkpoint.parent / 'test'
    if output.exists():
        raise ValueError(f'{output} 已存在，请选择新的 --output，避免覆盖评估记录。')
    device_name = args.device
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('当前没有可用 CUDA，请使用 --device cpu。')
    device = torch.device(device_name)
    torch.set_num_threads(args.num_threads)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    # 默认使用训练保存的副本，避免无意间更换了划分。
    manifest = args.manifest if args.manifest is not None else args.checkpoint.parent / 'manifest.csv'
    verify_manifest(manifest, checkpoint['manifest_sha256'])
    dataset = FlowerDataset(args.data, manifest, 'test', checkpoint['config']['image_size'],
                            checkpoint['mean'], checkpoint['std'])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f'设备：{device}；测试图片：{len(dataset)} 张', flush=True)
    true_labels, predictions, scores = predict_batches(model, loader, device)
    classes = checkpoint['classes']
    frame = pd.DataFrame({'path': dataset.frame['path'].tolist(),
                          'true': [classes[index] for index in true_labels],
                          'predicted': [classes[index] for index in predictions], 'score': scores})
    metrics, matrix, report = calculate_metrics(frame, classes)
    metrics.update({'selected_epoch': checkpoint['epoch'], 'validation_accuracy': checkpoint['val_accuracy'],
                    'manifest_sha256': checkpoint['manifest_sha256']})
    output.mkdir(parents=True)
    frame.to_csv(output / 'predictions.csv', index=False, encoding='utf-8-sig')
    frame.loc[frame['true'] != frame['predicted']].to_csv(output / 'misclassified.csv', index=False, encoding='utf-8-sig')
    matrix.to_csv(output / 'confusion_matrix.csv', encoding='utf-8-sig')
    report.to_csv(output / 'classification_report.csv', index=False, encoding='utf-8-sig')
    (output / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    save_matrix_plot(matrix, output / 'confusion_matrix.png')
    print(report.to_string(index=False))
    print(f'测试准确率：{metrics["accuracy"]:.2%}；宏平均 F1：{metrics["macro_f1"]:.4f}')
    print(f'结果已保存至：{output}')
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='训练生成的 best.pt')
    parser.add_argument('--data', type=Path, default=Path('data/flower_photos'))
    parser.add_argument('--manifest', type=Path, help='默认使用模型目录中的 manifest.csv')
    parser.add_argument('--output', type=Path, help='默认保存到模型目录中的 test 文件夹')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-threads', type=int, default=4)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    evaluate(parser.parse_args(argv))


if __name__ == '__main__':
    main()
