# 运行演示与录屏

演示目标是展示数据处理、手写网络、训练结果和实际预测。下面的流程建议控制在 5～8 分钟，
这是安排建议，考核没有限定视频时长。当前仓库尚未包含运行视频。

## 录屏前准备

按 README 安装依赖、下载数据、分析图片、划分数据并完成训练。
默认演示使用本地 `runs/experiment-01/best.pt`。使用其他训练目录时，通过 `-Checkpoint` 指定。
新克隆的仓库不包含权重，不能直接跳过训练。

准备打开以下内容：

- README 与 [项目说明](project-summary.md)。
- `src/model.py` 和 `src/train.py`。
- 本次训练目录的 `history.csv`、`training_curves.png`。
- 对应测试结果的 `metrics.json`、`confusion_matrix.png`。
- 一张待预测图片。优先增加自己拍摄、未参与数据划分的五类花卉图片。

## 快速运行演示

在 PowerShell 中，从仓库根目录执行：

```powershell
.\scripts\demo.ps1
```

脚本先展示网络各阶段的张量形状，再加载权重预测一张测试集雏菊图片。
每次在 `runs/demo/` 下新建结果目录，并保存 `prediction.json`，不覆盖之前的预测。
结构检查使用随机输入和随机参数，仅验证网络形状；分类步骤才使用已训练的权重。

预测自己拍摄的图片：

```powershell
.\scripts\demo.ps1 -Checkpoint runs/experiment-01/best.pt -Image "C:/photos/my-flower.jpg"
```

将图片路径换成真实文件。所有相对路径都以仓库根目录为基准，含空格路径需加引号。
可传 `-Output runs/demo/my-flower.json` 指定新的结果文件；可传 `-Device cuda` 使用可用的 CUDA。
如果 PowerShell 提示禁止运行脚本，也可直接执行下面两条命令：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.model
.\.venv\Scripts\python.exe -X utf8 -m src.predict --checkpoint runs/experiment-01/best.pt --image "C:/photos/my-flower.jpg"
```

## 推荐展示顺序

| 顺序 | 画面与操作 | 需要讲清楚的内容 |
|---|---|---|
| 1 | README 和项目目录 | 输入是一张花卉图片，输出为五类花卉的候选类别和分数 |
| 2 | 数据统计图与划分表 | 原始 3670 张，清洗后 3666 张；三个集合各自的用途 |
| 3 | `BasicBlock.forward` 和结构检查输出 | 两个分支为什么相加，形状变化时为什么需要 1×1 卷积 |
| 4 | 训练核心代码和已保存的曲线 | 清梯度、前向、交叉熵、反向、更新参数；按验证集选择最佳模型 |
| 5 | 测试指标与混淆矩阵 | 本次测试准确率 67.46%；玫瑰较易漏识别，整体分数不代表每类都好 |
| 6 | 显示图片，再运行预测 | 解释真实类别、预测类别及分数，保留预测错误的情况 |
| 7 | GitHub 提交列表与总结 | 项目各模块怎样连接，当前局限及下一步改进 |

讲解内容参考如下，请根据自己的理解表达：

> 本项目用手写 ResNet-18 识别雏菊、蒲公英、玫瑰、向日葵和郁金香。
> 图片先统一尺寸并归一化，再经过卷积和残差块，最终得到五个类别分数。
> 训练集更新参数，验证集选择模型，测试集用于评估已选模型。
> 这次训练了 8 轮，测试集 547 张中认对 369 张。模型对玫瑰仍有较多漏识别。

如果演示其他实验，请以它的配置和指标为准，不要继续引用 experiment-01 的数值。
展示保存的曲线时说明这是已经完成的训练记录；展示数据集图片时说明它所属的集合。
对新照片的一次预测也不等于已经测得真实场景准确率。

## 如需现场展示训练与评估

使用独立目录运行一轮小样本训练，用于展示程序如何工作：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.train --epochs 1 --limit-per-class 10 --output runs/video-check
```

不要将这次小样本模型的表现当作 8 轮实验结果。已有目录时换一个名字。
完整 8 轮实验的训练和验证计时约 4.9 分钟，实际耗时随机器变化。

如需现场重新计算已选模型的测试指标：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.evaluate --checkpoint runs/experiment-01/best.pt --output runs/video-evaluation
```

这只是重现同一模型的评估，不据此调整模型或训练参数。已有输出目录时换名。

## 视频保存与交付

可用 Windows 自带录屏或 OBS 录制，结束后保存为 MP4。
检查文字能否看清、声音能否听清，以及实际运行的命令和输出是否完整。
若展示数据集原图，按 `data/flower_photos/LICENSE.txt` 保留对应摄影者和来源署名；
自己拍摄的图片可以在视频中说明来源。

将视频与项目说明、仓库链接及必要运行结果一并打包，具体材料见 [项目说明](project-summary.md#交付内容)。
