# 参与改进

欢迎改进标注体验、安装说明、数据检查及教学解释。问题报告请包含操作系统、Python/PyTorch 版本、运行命令和完整报错，不要提交 `.venv`、缓存、私人数据或个人路径。

源代码贡献沿用 BSD-3-Clause。图片、截图和数据贡献需有权公开，并明确按 CC BY-NC 4.0 提供。避免把带有类别答案的参考图加入训练数据，也不要把同一采集组拆到多个数据分区来提高分数。

在已安装项目依赖的 Python 环境中提交前运行（Windows 未激活虚拟环境时，将 `python` 换为 `.\.venv\Scripts\python.exe`）：

```bash
python -m unittest discover -s tests -v
python check_data.py --sources source_frames
python scripts/verify_release.py
```

以上检查不训练模型。修改模型、预处理或训练行为时，还需在相应环境实际验证，并注明数据划分、样本数量和测试范围，不覆盖已有参考结果来冒充原实验。

若有意变更已发布的图片或结果素材，在确认来源和实际内容后运行 `python scripts/update_manifest.py` 更新校验清单，并审阅产生的差异。只修改代码或文字说明时不需要刷新清单。
