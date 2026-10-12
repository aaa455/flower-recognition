"""加载模型预测一张花卉图片。运行：python -m src.predict --help。"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from src.dataset import make_transform
from src.evaluate import load_checkpoint

CHINESE_NAMES = {'daisy': '雏菊', 'dandelion': '蒲公英', 'roses': '玫瑰',
                 'sunflowers': '向日葵', 'tulips': '郁金香'}


def predict_image(model, checkpoint, image_path: Path, device, top_k=3):
    """沿用权重中记录的类别与预处理设置，返回按分数排序的候选结果。"""
    classes = checkpoint['classes']
    if not 1 <= top_k <= len(classes):
        raise ValueError(f'top-k 必须在 1 到 {len(classes)} 之间。')
    transform = make_transform(checkpoint['config']['image_size'],
                               checkpoint['mean'], checkpoint['std'], training=False)
    with Image.open(image_path) as image:
        # 一张图片是 [3,H,W]；网络需要批次维度，故添加为 [1,3,H,W]。
        tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(tensor)
        if logits.shape != (1, len(classes)) or not torch.isfinite(logits).all():
            raise ValueError('模型输出形状错误或包含非有限数值。')
        # softmax 将原始分数归一化；它不代表经过校准的正确概率。
        scores, indices = logits.softmax(dim=1)[0].topk(top_k)
    return [{'rank': rank, 'label': classes[index],
             'name': CHINESE_NAMES.get(classes[index], classes[index]), 'score': score}
            for rank, (score, index) in enumerate(zip(scores.cpu().tolist(), indices.cpu().tolist()), start=1)]


def predict(args):
    if args.num_threads < 1:
        raise ValueError('num-threads 必须大于 0。')
    if args.output is not None and args.output.exists():
        raise ValueError(f'{args.output} 已存在，请使用新的输出文件。')
    device_name = args.device
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('当前没有可用 CUDA，请使用 --device cpu。')
    device = torch.device(device_name)
    torch.set_num_threads(args.num_threads)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    results = predict_image(model, checkpoint, args.image, device, args.top_k)
    report = {'image': str(args.image), 'checkpoint': str(args.checkpoint),
              'model_epoch': checkpoint['epoch'], 'predictions': results}
    print(f'图片：{args.image}')
    for result in results:
        print(f'{result["rank"]}. {result["name"]}（{result["label"]}）：{result["score"]:.2%}')
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'预测记录已保存至：{args.output}')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='训练生成的 best.pt')
    parser.add_argument('--image', type=Path, required=True, help='待预测的图片路径')
    parser.add_argument('--top-k', type=int, default=3, help='显示候选数量，默认 3，可选 1 到 5')
    parser.add_argument('--output', type=Path, help='可选的 JSON 输出文件路径')
    parser.add_argument('--num-threads', type=int, default=4)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    predict(parser.parse_args(argv))


if __name__ == '__main__':
    main()
