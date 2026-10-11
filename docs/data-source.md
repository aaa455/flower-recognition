# 花卉数据

使用 TensorFlow 提供的 flower_photos 示例数据，共 3,670 张图片，包含五个类别：

| 目录名称 | 花卉类别 |
|---|---|
| daisy | 雏菊 |
| dandelion | 蒲公英 |
| roses | 玫瑰 |
| sunflowers | 向日葵 |
| tulips | 郁金香 |

下载地址：https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

压缩包约 218 MiB，下载并解压后，目录结构如下：

```text
data/
├─ flower_photos.tgz
└─ flower_photos/
   ├─ daisy/
   ├─ dandelion/
   ├─ roses/
   ├─ sunflowers/
   ├─ tulips/
   └─ LICENSE.txt
```

每个类别目录存放对应花卉的 JPG 图片，文件夹名称作为类别标签。
原始数据仍需检查重复图片、标签冲突和损坏图片，再用于训练。

图片采用 CC BY 2.0 许可，`LICENSE.txt` 逐张记录摄影者及来源。
分享图片或包含图片的预测示例时，应保留相应署名。

## 图片检查与统计

运行 `python -m src.analyze`，默认分析 `data/flower_photos/`，将结果保存到 `data/analysis/`。
如果数据位于其他目录，可以传入 `--data`；`--output` 指定报告目录。

| 输出文件 | 内容 |
|---|---|
| `images.csv` | 可读图片的相对路径、类别、宽高与 SHA-256 哈希 |
| `invalid_images.csv` | 无法读取的图片及错误信息 |
| `class_counts.csv` | 各类可读图片的数量与比例 |
| `duplicate_images.csv` | 完全重复文件的所有记录 |
| `conflicting_labels.csv` | 文件内容相同、类别不同的所有记录 |
| `summary.json` | 图片总量、重复与冲突数量、平均尺寸、中位尺寸 |
| `class_distribution.png` | 类别数量柱状图 |
| `image_sizes.png` | 图片宽高散点图 |

同一重复组可能同时属于标签冲突组，所以两份报告的记录数不能直接相加。
哈希检查只识别字节完全相同的文件，不会识别经过压缩、裁剪或缩放的相似图片。
统计基于当前所有可读图片，重复与冲突文件也包含在类别数量中。
本步骤不删除图片，也不划分数据；后续根据报告进行清洗和划分。

## 清洗与划分

执行 `python -m src.prepare`，重新读取原图，依次处理：

1. 不可读取的图片记录到异常表，不进入划分。
2. 相同内容对应不同标签时，排除整个冲突组，不猜测正确类别。
3. 同标签完全重复文件按路径排序，只保留第一份。
4. 每个类别单独打乱，约 70% 用于训练、15% 用于验证、15% 用于测试。

各类验证、测试数量向下取整，剩余图片归训练集，所以比例略有差异。
每类清洗后至少需要 7 张图片，这是脚本保证三个集合均有样本的最低限制，不代表足够训练模型。

默认输出 `data/splits/`：

| 文件 | 内容 |
|---|---|
| `manifest.csv` | 保留图片的路径、类别、尺寸、哈希及 `split` 列 |
| `class_counts.csv` | 清洗后各类别在三个集合中的数量 |
| `invalid_images.csv` | 不可读取的图片 |
| `excluded_conflicts.csv` | 排除的整个标签冲突组 |
| `removed_duplicates.csv` | 同标签重复组中没有保留的记录 |
| `summary.json` | 清洗数量、种子、类别编号与各集合总量 |

`train` 用于学习参数，`val` 用于比较模型和调整方案，`test` 用于最终评估。
同一文件内容只保留一份，避免它同时出现在不同集合中造成数据泄漏。
这不能排除所有视觉近似图片；若是连续拍摄、同一植株等关联数据，还需按来源分组检查。

随机种子固定打乱顺序。相同数据与种子能够复现本脚本的划分，不保证不同软件版本的结果完全相同。
实验开始后应保留对应的 `manifest.csv`，不要为获得更好结果反复更换测试集。

## 图片预处理

`src/dataset.py` 使用 `FlowerDataset` 实现 PyTorch 的 `Dataset` 接口：

- `__len__()` 返回集合中的图片数量。
- `__getitem__(index)` 读取一张图片，返回图片张量和整数类别。

`DataLoader` 再把多张图片组成一个批次。原始图片的高、宽可能不同，预处理会统一缩放到指定尺寸。
一张 RGB 图片进入模型前的形状是 `[3, 高, 宽]`；批次形状是 `[批次大小, 3, 高, 宽]`。
标签按照 `daisy`、`dandelion`、`roses`、`sunflowers`、`tulips` 映射为 0 到 4。

均值和标准差只由 `train` 集计算，避免验证集和测试集的信息进入训练流程。
统计时将训练图片缩放为 32×32，以减少计算量；这是缩小图片后的统计量。
训练图片随机水平翻转，以增加外观变化；验证和测试不使用随机操作，保证评估稳定。
归一化公式为 `(像素值 - 均值) / 标准差`，三个通道分别计算。

程序不会把图片读成一个巨大的数组，而是按 `__getitem__` 需要逐张读取，`DataLoader` 再按批次提供，降低内存占用。
