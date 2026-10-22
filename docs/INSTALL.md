# 安装与常见问题

## 版本

参考实验已在 Windows、Python 3.12、RTX 4060 Laptop 8GB、PyTorch `2.8.0+cu126` 上完成。安装文件固定 torchvision `0.23.0+cu126` 及普通依赖版本，以便提供一致的起点。原实验未导出完整 `pip freeze`，因此不将这份依赖文件冒充当时所有包的精确环境快照。

官方版本组合说明：https://pytorch.org/get-started/previous-versions/#v280

## Windows CPU 安装

CPU 可以运行相同的标注、预测和训练入口，速度取决于电脑。这条 CPU 安装路径没有作为本次 4060 实验的性能测试结果。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-torch-cpu.txt --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_environment.py
```

CPU 训练、测试需分别增加 `--device cpu`。`app.py`、`predict.py` 默认自动选择可用设备。

## Linux

安装 Python 3.12 和 Tk 支持后，在项目根目录执行：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-torch-cu126.txt --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python check_environment.py --require-cuda
```

后续将 README 中的 `.\.venv\Scripts\python.exe` 替换成 `.venv/bin/python`。标注工具需要图形桌面和 Tk。无 GPU 时改用 CPU 依赖文件及其官方源。Linux GPU 路径未在本次发布整理中实机训练验证。

## 常见问题

| 现象 | 处理 |
|---|---|
| `py -3.12` 找不到 Python | 用 `py -0p` 检查，安装 64 位 Python 3.12 或显式传入已有兼容版本的 `-PythonVersion` |
| PyTorch 下载很慢 | CUDA wheel 体积较大，普通 PyPI 镜像不会自动加速它。脚本支持独立指定 `-TorchIndex` |
| `CUDA unavailable` | 先运行 `nvidia-smi`，确认 NVIDIA 驱动，再检查是否误装 CPU 版 torch |
| 标注窗口无法打开 | 检查 Python 的 Tcl/Tk 组件和图形桌面，不需要安装训练框架才能使用 Tk 本身 |
| `Class ... occurs in only ... groups` | 在至少 3 个独立采集组中为该类补充有效标注，不复制图片或随意拆分近邻帧 |
| 输出目录已存在 | 保留原目录，用新的 `--out`；后续 `--data`、`--checkpoint` 与新实验保持一致 |
| `Dataset changed since training` | 权重和当前划分不匹配，使用原固定划分，或重新训练自己的模型 |
| `Checkpoint not found` | 自行训练，或按 [WEIGHTS.md](WEIGHTS.md) 放置 Release 权重 |
| 7860 端口占用 | `app.py --port 7861` |

安装脚本可以重复运行，不会删除已有数据或重新训练模型。显存不足可关闭占用 GPU 的程序，并将 batch size 降为 16 或 8。

首次训练需要 torchvision 下载官方 ResNet18 权重。离线机器可提前准备官方 `resnet18-f37072fd.pth`，运行 `train.py --pretrained-file 文件路径`。`.dlkpack` 不是 PyTorch 权重，不能替代。
