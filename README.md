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

## 主要依赖

| 依赖 | 用途 |
|---|---|
| PyTorch | 张量计算、网络搭建和模型训练 |
| torchvision | 图片预处理 |
| NumPy | 数组计算与数据统计 |
| Pandas | 数据表格与分组统计 |
| Matplotlib | 绘制数据分布和训练曲线 |
| Pillow | 读取与转换图片 |
