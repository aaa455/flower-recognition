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
