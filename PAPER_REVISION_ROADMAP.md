# DV-VLN 论文修订路线

## 当前定位

DV-VLN 的核心贡献是面向语言化视觉导航（vision-to-text VLN）的推理时双重验证：先采样多个候选动作，再用 True–False Verification（TFV）和 Masked-Entity Verification（MEV）进行重排。当前实验验证的是 Matterport3D/Habitat 中的离散 waypoint 导航。

应避免将系统表述为真正端到端或已完成机器人部署验证的通用 VLA。当前工作没有实车、真实传感器、连续控制或 sim-to-real 闭环验证。

## 必须修订

- 将 “end-to-end” 改为模块化的 vision-to-text / language-mediated navigation 表述。
- 明确声明实验为仿真和离线 benchmark；在 limitations 中说明暂无实车验证。
- 在完整 Val Unseen 上重跑主要结果，而不只依赖 90 条子集。
- 使用至少 3 个随机种子，报告均值、标准差和置信区间。
- 报告推理成本：LLM 调用次数、token 数、GPU 显存、平均延迟和每条轨迹耗时。
- 修正损失函数的序列交叉熵写法，并统一候选数与全景视角数的符号。

## 最有价值的新增实验

### 可靠性

- candidate recall：正确动作是否进入采样候选集；
- TFV/MEV 的 candidate-level accuracy、precision、recall 和 AUROC；
- calibration / ECE；
- 首次错误后的恢复率和轨迹错误数；
- majority vote、LLM log-probability、TFV-only、MEV-only、随机验证器等 baseline。

### 视觉输入

比较 BLIP、BLIP-2 或更强 VLM，以及 caption、物体标签、方向信息的组合。使用 2×2 实验区分性能提升究竟来自更强视觉描述还是双重验证机制。

### 鲁棒性

加入实体删除、方向词替换、无关描述、候选顺序打乱、历史屏蔽、指令改写、图像模糊/遮挡和 hard-negative 候选，比较 direct、sampling 和 DV-VLN。

### 最小现实验证

在 20–50 个真实室内场景上采集候选视角，测试短程方向选择；如果暂时没有机器人，加入图像退化、观测延迟和动作执行误差，并称为 sim-to-real robustness proxy，而非真实机器人验证。

## 方法改进方向

- 自适应验证预算：仅对候选分歧大或置信度低的步骤启动 TFV/MEV；
- 使用归一化或校准后的 TFV/MEV 分数，避免实体数量影响 MEV；
- 引入独立 verifier 或冻结 verifier，减少同一 LLM 自我确认偏差；
- 将候选验证扩展为短期未来状态/地图一致性检查；
- 增加 abstain、重新观察或安全停止策略。

## 相关进展与比较原则

补充讨论强视觉语言模型、VLA、在线地图/空间记忆、world model、test-time scaling 和 Habitat 3.0/VLN-CE。离散 waypoint VLN、视觉语言导航、连续控制 VLN 和通用 VLA 应分组比较，避免不公平横向排名。

## 推荐执行顺序

1. 完整 Val Unseen + 多随机种子 + 成本统计；
2. candidate-level 验证器分析和强 baseline；
3. BLIP/强 VLM 对照；
4. 输入扰动与 hard-negative 鲁棒性；
5. 真实室内图像或仿真到现实代理实验；
6. 根据结果收窄标题、摘要和贡献表述。

