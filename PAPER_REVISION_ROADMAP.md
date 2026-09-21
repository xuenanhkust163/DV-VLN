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

## 重排后的 DV-VLN 执行计划

以下顺序将原论文复现作为主线，并把 `revision/DV-VLN_Revision_Experiment_Checklist.pdf` 所对应的补充实验纳入同一条可审计流水线。正式结果必须与论文设置一致；smoke test 只能用于链路诊断，不得写入主结果表。

### 阶段 0：实验协议冻结与资源门禁

- 固定代码 commit、数据版本、模型 checkpoint、GPU/软件环境和随机种子。
- 按 `experiment/EXPERIMENT_SETTINGS.md` 冻结正式参数：R2R/Matterport3D 离散 waypoint、`cand`、`vit-16-ori`、36 views、候选采样 4、`top_p=0.9`、verification attempts 4、`max_action_len=15`、`--llm_predict --stop_first`。
- 检查 R2R 标注、特征、connectivity、candidate captions、tokenizer 和 checkpoint；资源缺失时不得启动正式评测。
- 清理硬编码 API key，改用环境变量并撤销已暴露密钥。

### 阶段 1：运行环境与模拟器一致性

- 验证 Python/PyTorch/CUDA/MatterSim 组合、单 GPU 初始化和动态库路径。
- 验证 `Simulator.initialize()`、`getState()`、`makeAction()` 与论文代码的动作语义一致。
- 固定 DV-VLN 与当前 RTX 6000D 兼容的运行时；记录所有兼容性改动，不改变方法逻辑。

### 阶段 2：最小链路 smoke test（非正式结果）

- 使用 2～10 条样本验证数据读取、36 视角特征、候选动作、TFV、MEV、轨迹写出和评测脚本。
- 记录每步耗时、LLM 生成耗时、调用次数、非法动作和提前停止原因。
- 仅当链路 `RC=0` 且 checkpoint 加载无未解释的 missing/unexpected keys 后进入正式实验。

### 阶段 3：论文主线基线复现

- 在完整 `val_unseen` 上先运行 DV-VLN 论文的完整方法（TFV+MEV），使用 seed 0/1/2。
- 严格保持论文的候选数量、采样策略、验证次数、动作上限和 stop 规则。
- 产出每个 seed 的预测、日志、SR/SPL/NE（及论文报告的其他指标）、LLM 调用次数和轨迹耗时，并计算均值±标准差。

### 阶段 4：修订补充消融矩阵

- **No verifier**：不使用验证器的 LLM 动作选择。
- **TFV-only**：只启用 True–False Verification。
- **MEV-only**：只启用 Masked-Entity Verification。
- **TFV+MEV**：完整 DV-VLN 方法，与阶段 3 结果交叉核对。
- **Simple reranking baseline**：按 checklist 选择 majority vote 或普通 LLM reranking；不引入额外 VLN-BERT 非 LLM baseline（沿用当前实验范围决定）。
- 所有可比组复用同一 split、seed、候选和预算，避免消融同时改变数据或推理预算。

### 阶段 5：补充分析实验

- 统计 candidate recall、验证器判别正确率及最终动作选择错误类型。
- 在资源允许时增加一项轻量 robustness：输入扰动或 hard-negative 候选；保持原始实验协议不变，仅改变该分析变量。
- 对每组报告平均 LLM 调用次数、每条轨迹延迟、生成 token 数（可得时）和 GPU 显存。

### 阶段 6：性能与可重复性审计

- 在不改变正式协议的前提下，单独优化推理吞吐：prompt/视觉缓存、生成长度控制和 episode 计时。
- 对超时、异常退出、非法动作和空预测建立失败清单；优化前后分别保留配置与结果，不能混合进同一统计组。
- 将实际命令、commit、环境、数据路径、checkpoint、seed、开始/结束时间和结果文件逐条追加到 `experiment/RUN_LOG.md`。

### 阶段 7：论文与 Dashboard 交付

- 汇总主结果、消融、成本、鲁棒性和失败分析，形成均值±标准差表格。
- 修正损失公式、符号、候选/TFV/MEV 流程图和 “end-to-end” 表述；明确仅在仿真离散 waypoint 上验证。
- 将 DV-VLN 指标、实验组、seed、调用成本和失败类型接入 NavCoT Dashboard，区分 smoke、正式 baseline 和 revision supplementary。
- 最后再收窄标题、摘要和贡献表述，确保结论只覆盖实际完成的实验。
