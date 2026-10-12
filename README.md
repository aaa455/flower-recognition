# flower-recognition

基于 Python 与 PyTorch 的五类花卉图像分类项目。

## 项目目标

- 统计花卉图片的类别分布，检查数据并划分训练集、验证集和测试集。
- 手动搭建 ResNet-18，完成模型训练与评估。
- 加载保存的模型，对单张花卉图片进行分类预测。

## 环境要求

- Python 3.12
- Git

## 安装依赖

在 PowerShell 中执行：

```powershell
git clone https://github.com/aaa455/flower-recognition.git
cd flower-recognition
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

使用虚拟环境的 Python 路径执行命令，无需激活环境。

## 下载数据

```powershell
.\.venv\Scripts\python.exe -m src.download
```

脚本下载并解压约 218 MiB 的花卉数据，保存到 `data/flower_photos/`。
已成功解压的数据会跳过重复操作；已有压缩包时会直接使用。
数据集类别、目录结构和许可见 [数据说明](docs/data-source.md)。

使用其他保存目录：

```powershell
.\.venv\Scripts\python.exe -m src.download --root "D:/datasets/flowers"
```

## 检查与统计图片

下载完成后执行：

```powershell
.\.venv\Scripts\python.exe -m src.analyze
```

脚本检查图片是否可读，统计每类数量与图片尺寸，查找完全重复文件和标签冲突。
结果保存在 `data/analysis/`，包括 CSV 表格、JSON 摘要、类别分布图和图片尺寸图。
本步骤只生成报告，保留原始图片。输出含义见 [数据说明](docs/data-source.md)。

## 清洗与划分数据

```powershell
.\.venv\Scripts\python.exe -m src.prepare
```

脚本排除损坏图片和整个标签冲突组，同标签完全重复图片只保留一份。
随后按类别划分约 70% 训练、15% 验证、15% 测试，结果保存在 `data/splits/`。
原始图片不会被移动或删除；`manifest.csv` 的 `split` 列记录各图片所属集合。
固定种子为 42，相同数据和种子可复现划分。更换种子时用 `--output` 保存到其他目录，避免覆盖已有实验的记录。

检查清洗和划分规则：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 图片预处理与数据加载

完成划分后，运行下面的命令检查一个批次：

```powershell
.\.venv\Scripts\python.exe -m src.dataset
```

处理流程是：读取图片 → 调整为统一尺寸 → 转换为张量 → 按训练集统计量归一化。
训练集增加随机水平翻转，验证集和测试集使用确定性处理。输出会显示三个集合的数量、均值、标准差和批次形状。

默认每批 16 张、输入尺寸 96×96。修改参数：

```powershell
.\.venv\Scripts\python.exe -m src.dataset --image-size 128 --batch-size 32
```

## 手写 ResNet-18

```powershell
.\.venv\Scripts\python.exe -m src.model
```

命令显示模拟输入经过各阶段后的形状和模型参数量。
`src/model.py` 手动实现残差块及四个网络阶段，默认将 `[N,3,96,96]` 图片批次转换为 `[N,5]` 分类分数。
当前命令只检查网络结构，不展示已训练的预测结果。网络原理见 [模型说明](docs/model.md)。

## 训练与验证

先用少量训练图片检查流程：

```powershell
.\.venv\Scripts\python.exe -m src.train --epochs 1 --limit-per-class 10 --output runs/first-check
```

使用全部训练图片训练 8 轮：

```powershell
.\.venv\Scripts\python.exe -m src.train --epochs 8 --output runs/experiment-01
```

默认输入 96×96，每批 16 张，自动使用可用的 CUDA 或 CPU。
每轮输出训练和验证指标，并在输出目录保存曲线、记录和最佳验证模型 `best.pt`。
每次使用新的 `--output` 目录，已有目录不会覆盖。参数详见 `python -m src.train --help`，原理见 [训练说明](docs/training.md)。
本步骤不评估测试集；训练与方案选择完成后再执行下方评估命令。

## 测试集评估

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --checkpoint runs/experiment-01/best.pt
```

将路径替换为实际训练生成的 `best.pt`。脚本使用模型目录中的数据划分副本，核对后加载测试图片。
输出测试准确率、各类精确率/召回率/F1、混淆矩阵和错误预测记录，默认保存到模型目录下的 `test/`。
可用 `--output` 指定新的结果目录，已有目录不会覆盖。指标与图表含义见 [评估说明](docs/evaluation.md)。
测试集用于最终评估，后续调整方案仍应依据验证集。

## 单张图片预测

```powershell
.\.venv\Scripts\python.exe -m src.predict --checkpoint runs/experiment-01/best.pt --image "C:/photos/flower.jpg"
```

将模型和图片路径替换为实际路径，默认输出前三个候选类别及 softmax 分数。
使用 `--top-k 1` 只显示第一候选；`--output runs/predictions/flower.json` 保存 JSON 结果，已有文件不会覆盖。
预测自动使用训练时的输入尺寸与归一化参数，不需要重新训练或提供数据集。
分数未经概率校准，模型只识别五类花卉。具体流程见 [预测说明](docs/prediction.md)。

## 实测结果

使用全部 2572 张训练图片，从随机参数开始训练 8 轮，输入为 96×96。
按验证准确率选中第 8 轮模型，验证准确率为 **68.92%**。
在 547 张测试图片上，准确率为 **67.46%**，宏平均 F1 为 **0.6584**。

训练曲线、各类表现、混淆矩阵及复现方法见 [实验报告](docs/experiment-report.md)，
原始指标与数据划分记录位于 [results/experiment-01](results/experiment-01/)。
这些结果来自一次固定划分实验。仓库中的结果文件不含模型权重，新克隆后请先训练，再运行预测。

## 运行演示与项目说明

完成训练后，在 PowerShell 中运行：

```powershell
.\scripts\demo.ps1
```

脚本检查网络形状，再用已训练模型预测示例图片，将结果保存到新的 `runs/demo/` 子目录。
使用 `-Checkpoint` 指定权重，`-Image` 指定自己的图片。
录屏顺序和完整操作见 [运行演示](docs/demonstration.md)；
模块说明、结果总结和交付材料见 [项目说明](docs/project-summary.md)。

## 主要依赖

| 依赖 | 用途 |
|---|---|
| PyTorch | 张量计算、网络搭建和模型训练 |
| torchvision | 图片预处理 |
| NumPy | 数组计算与数据统计 |
| Pandas | 数据表格与分组统计 |
| Matplotlib | 绘制数据分布和训练曲线 |
| Pillow | 读取与转换图片 |
