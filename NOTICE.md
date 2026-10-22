# 来源与项目范围

项目依据 PyTorch 官方 **Transfer Learning for Computer Vision Tutorial** 的迁移学习方法改编。上游许可和版权声明保留在 `LICENSE`，项目代码继续采用 BSD-3-Clause。

- 教程：https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- 教程源码：https://github.com/pytorch/tutorials/blob/main/beginner_source/transfer_learning_tutorial.py
- 上游许可：https://github.com/pytorch/tutorials/blob/main/LICENSE

改编内容包括六类纸盒数据、手动单盒裁剪、按采集组划分、两阶段 ResNet18 训练、独立测试报告、本地网页预测，以及保持完整盒面的缩放补边预处理。训练骨干从 torchvision 的 ImageNet 预训练 ResNet18 开始。

实际机器人实验使用 Mech-Eye PRO S、Mech-Vision / Mech-DLK、Mech-Viz 与 ABB 机器人。这个教学项目独立讲解图像分类，没有读取或转换 `.dlkpack`，没有使用或声称复现 Mech-DLK 的内部网络，也没有连接机器人。仓库不包含厂商软件、SDK、机器人备份、工业视觉工程或标定文件。

## 素材与许可

`source_frames` 的 84 张图片来自实验室采集原图的按日期抽样，`reference_labels` 的 6 张图来自原实验的类别展示图。发布者已确认可以公开这些素材。原图索引移除了实验室工程路径和相机编号，仍保留相对路径、采集组和 SHA-256。

照片、参考图、包含照片的截图和实验结果采用 CC BY-NC 4.0，详见 [DATA_LICENSE.md](DATA_LICENSE.md)。代码许可不自动涵盖图片内容。参考权重的来源及适用范围见 [docs/WEIGHTS.md](docs/WEIGHTS.md)。

## 实际运行与检查范围

参考实验已在 Windows、Python 3.12、RTX 4060 Laptop 8GB、PyTorch 2.8.0+cu126 上完成标注、训练、测试和预测。保存的结果位于 `results/experiment01`。原始 52 张裁剪、人工标注和固定划分未公开，因此本仓库不支持精确重建该次数据集。

发布整理保留模型结构、预处理和训练算法，增加无需 PyTorch 的数据完整性检查、公开文档、安装参数和自动检查。发布整理阶段验证文件、数据检查与划分行为，没有重新运行 GPU 训练。
