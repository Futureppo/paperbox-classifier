# 参考权重

参考权重作为 [v1.0.0 Release](https://github.com/Futureppo/paperbox-classifier/releases/tag/v1.0.0) 的独立附件提供，文件名为 `paperbox-resnet18-best.pt`。它来自 `experiment01` 的第 15 轮，大小 44,791,691 字节。权重元数据见 [`weights.json`](../results/experiment01/weights.json)。

[直接下载参考权重](https://github.com/Futureppo/paperbox-classifier/releases/download/v1.0.0/paperbox-resnet18-best.pt)

SHA-256：

```text
7c96412a3a3e1ef59af302851aa4cc558905c59d2bc5287f6c7a6dc473f5088e
```

下载后放在项目根目录，校验并移动到默认位置：

```powershell
Get-FileHash .\paperbox-resnet18-best.pt -Algorithm SHA256
New-Item -ItemType Directory -Force .\runs\experiment01 | Out-Null
Move-Item .\paperbox-resnet18-best.pt .\runs\experiment01\best.pt
.\.venv\Scripts\python.exe app.py
```

目标目录已有自己的 `best.pt` 时保留它，使用 `app.py --checkpoint 新下载的权重路径` 指定参考权重即可。加载权重做预测不会额外下载 ImageNet 权重。

参考权重用于体验六类单盒分类。它没有未知类别拒识，也没有验证新商品、新实体盒子或新环境下的泛化。原固定划分不公开，不能拿它与重新标注的数据组合来复测原始测试报告。

## 来源与使用

模型基于 torchvision 的 [ResNet18 ImageNet 预训练模型](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) 微调。代码 BSD-3-Clause 许可不自动替代上游预训练模型及数据的适用条款。

本项目发布者对其有权许可的微调权重内容采用 **CC BY-NC 4.0**，用于署名后的非商业学习和实验。许可链接与署名方式见 [DATA_LICENSE.md](../DATA_LICENSE.md)。使用时还应遵守上游预训练模型适用的许可与条件。
