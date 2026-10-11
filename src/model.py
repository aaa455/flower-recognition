"""手动搭建 ResNet-18。运行：python -m src.model 查看各层输出形状。"""
import argparse

import torch
from torch import nn

from src.download import CLASS_NAMES


class BasicBlock(nn.Module):
    """两个 3×3 卷积组成主分支，再与捷径分支相加。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 只有通道数和空间尺寸均不变时，捷径才可以直接使用原输入。
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 逐元素相加，两个分支的 [N,C,H,W] 必须完全一致。
        out = out + shortcut
        return self.relu(out)


class ResNet18(nn.Module):
    """四个阶段各包含两个残差块，默认输出五个花卉类别的分数。"""

    def __init__(self, num_classes=len(CLASS_NAMES)):
        super().__init__()
        if num_classes < 1:
            raise ValueError('类别数量必须大于 0。')
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self.make_stage(64, 64, stride=1)
        self.layer2 = self.make_stage(64, 128, stride=2)
        self.layer3 = self.make_stage(128, 256, stride=2)
        self.layer4 = self.make_stage(256, 512, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    @staticmethod
    def make_stage(in_channels, out_channels, stride):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels, stride=1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)  # [N,512,1,1] → [N,512]
        return self.classifier(x)  # 原始分数 logits；不在网络中做 softmax。


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-size', type=int, default=96, help='模拟输入的高和宽')
    parser.add_argument('--batch-size', type=int, default=2, help='模拟批次大小')
    args = parser.parse_args(argv)
    if args.image_size < 32 or args.batch_size < 1:
        parser.error('图片尺寸至少为 32，批次大小至少为 1。')
    torch.set_num_threads(4)
    model = ResNet18().eval()
    x = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
    print(f'输入：{list(x.shape)}')
    # 本命令只检查网络结构，随机输入与随机参数不能用于评价分类效果。
    with torch.inference_mode():
        for name in ['stem', 'layer1', 'layer2', 'layer3', 'layer4', 'pool']:
            x = getattr(model, name)(x)
            print(f'{name}：{list(x.shape)}')
        x = torch.flatten(x, start_dim=1)
        print(f'展平：{list(x.shape)}')
        print(f'分类输出：{list(model.classifier(x).shape)}')
    print(f'参数总量：{sum(parameter.numel() for parameter in model.parameters()):,}')


if __name__ == '__main__':
    main()
