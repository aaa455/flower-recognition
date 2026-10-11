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

## 主要依赖

| 依赖 | 用途 |
|---|---|
| PyTorch | 张量计算、网络搭建和模型训练 |
| torchvision | 图片预处理 |
| NumPy | 数组计算与数据统计 |
| Pandas | 数据表格与分组统计 |
| Matplotlib | 绘制数据分布和训练曲线 |
| Pillow | 读取与转换图片 |
