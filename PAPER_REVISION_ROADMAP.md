# DV-VLN 论文修订计划（Applied Sciences 投稿版）

## 投稿定位

DV-VLN 的核心贡献是面向语言化视觉导航（vision-to-text VLN）的推理时双重验证：先采样多个候选动作，再用 True–False Verification（TFV）和 Masked-Entity Verification（MEV）进行重排。当前实验验证的是 Matterport3D/Habitat 中的离散 waypoint 导航。

本文实验基于 Matterport3D/Habitat 中的离散 waypoint 导航。论文应明确这是仿真 benchmark 上的模块化 VLN 方法，不应表述为已完成真实机器人部署、连续控制或通用 VLA 验证。

## 必须完成的修订

### 1. 实验结果可信度

- 在完整的 Val Unseen 集上报告主要结果，不只使用 90 条子集。
- 使用至少 3 个随机种子，报告均值和标准差；如计算方便，可补充置信区间。
- 核对数据划分、评价指标、候选数量、全景视角数量和所有实验设置。

### 2. 核心消融与 baseline

至少包含：不使用验证器、仅 TFV、仅 MEV、TFV + MEV，以及 majority vote 或普通 LLM 重排等一个简单 baseline。重点回答性能提升是否确实来自双重验证。

### 3. 方法和论文表述

- 修正序列交叉熵等损失函数写法，统一符号和维度定义。
- 明确候选生成、TFV、MEV 和最终动作选择流程。
- 将 “end-to-end” 改为模块化或 language-mediated VLN。
- 在 limitations 中说明当前仅在仿真和离散 waypoint 设置下验证。

### 4. 可复现性与成本

- 补充代码、数据、模型和运行环境说明。
- 简要报告平均 LLM 调用次数、单条轨迹耗时或平均延迟；可获得时再报告 token 数和 GPU 显存。

## 建议但非必需的增强

根据资源，可增加 candidate recall/验证器准确率分析，或一项输入扰动 / hard-negative 鲁棒性实验。

## 当前阶段不必强求

真实机器人验证、多种 VLM 大规模组合、AUROC/ECE、自适应预算、独立 verifier、sim-to-real 闭环、连续控制及 Habitat 3.0 或通用 VLA 的大范围比较，均非 Applied Sciences 投稿硬性要求。

## 推荐执行顺序

1. 完整 Val Unseen + 3 个随机种子；
2. TFV/MEV 消融和一个简单 baseline；
3. 修正公式、符号、实验设置和限制说明；
4. 补充运行成本与复现信息；
5. 视资源增加一项轻量分析；
6. 根据结果收窄标题、摘要和贡献表述。

