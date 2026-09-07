# DV-VLN CPU 仿真准备状态

## 当前结论

DV-VLN 的 R2R 评测通过 MatterSim（Matterport3D Simulator）运行。CPU 可以启动 MatterSim 并执行环境动作，但完整模型推理不适合当前无 GPU 环境，且仍需要本地准备 Matterport3D 数据、R2R 导航图、视觉特征和模型权重。

## 目录约定

```text
DV-VLN/
└── datasets/
    ├── R2R/
    │   ├── annotations/
    │   ├── connectivity/
    │   ├── features/
    │   └── exprs/
    └── Matterport3D/
        └── v1_unzip_scans/
```

请将官方数据分别放入上述目录。仓库脚本默认使用 `--root_dir ../datasets`（从 `finetune_src` 运行时）.

## 阻塞项

- MatterSim C++ 核心库和 Python 扩展已使用 OSMesa CPU 模式编译成功，并在现有 `janusvln-env`（Python 3.9）中通过导入及导航图初始化测试。
- 当前机器未检测到完整 Matterport3D 扫描、connectivity 文件或 `vit-16.hdf5` 特征文件。
- `finetune_src/scripts/run_*.sh` 中的模型路径仍是占位符，且默认使用 CUDA；CPU 运行需改为 `--device cpu` 并准备 CPU 可加载的权重。
