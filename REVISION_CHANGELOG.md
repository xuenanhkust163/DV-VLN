# DV-VLN Revision Change Log

记录本次审稿修订中每个要点的变更内容，格式为：
"要点：将 A，改成了 B。"

## 2026-09-09

### 1. 创新性边界的表述
- 要点：明确区分“通用 self-verification / generate-then-verify 思路”与“面向 VLN 的导航专用设计”。
- 将 A："generic self-verification or generate-then-verify pipelines"。
- 改成 B："a VLN-specific generate–then–verify framework"，并在摘要和贡献列表中强调其针对导航决策、重排序和验证的场景适配。

### 2. 摘要中的贡献定位
- 要点：避免让方法看起来像“泛 LLM 生成-验证框架”，而是突出其在 VLN 中的任务适配。
- 将 A："a first generate-then-verify setup for VLN" / "generic LLM reasoning"。
- 改成 B："a VLN-specific generate–then–verify framework that moves beyond single-shot action selection by sampling multiple structured navigation hypotheses and re-ranking them before execution"。

### 3. 贡献列表表述
- 要点：收紧贡献，强调导航中的候选重排序与验证，而不是泛泛的 reasoning 叙述。
- 将 A：强调“通用 LLM reasoning / generic verification”语气。
- 改成 B：强调“Prediction–View Match–Action + TFV / MEV + verification-guided re-ranking for embodied navigation decision making”。

### 4. cross-modal 比较的表述
- 要点：保留 cross-modal 作为参考上界，但不让它成为本文主创新与主叙事。
- 将 A："competitive performance against several cross-modal systems"。
- 改成 B："representative cross-modal systems as reference upper bounds under a lighter language-only protocol"，并强调方法定位为轻量、可解释、可扩展的语言-only 方案。

### 5. Related Work 的边界澄清
- 要点：使文献综述明确区分通用验证方法与 VLN 专用设计。
- 将 A：把现有方法概括成较为笼统的“LLM-based reasoning / verification”叙述。
- 改成 B：在 Related Work 中区分“通用 LLM reasoning / self-verification”与“VLN-specific chain-of-thought + dual verification”两类方法。

### 6. 方法定义的澄清
- 要点：明确 Prediction–View Match–Action 三个字段的职责，避免读者把它们当成同一个动作决策步骤。
- 将 A："Prediction (what the next scene should be), View Match (which view supports this prediction), and Action (which direction to move)"。
- 改成 B："Prediction describes the expected next-scene state implied by the instruction and current context; View Match identifies the specific panorama or viewpoint that best supports this prediction; Action selects the final navigation direction"。
- 目的：把语义预测、视图匹配和最终动作决策分开说明，增强可解释性和方法清晰度。

### 7. MEV 匹配规则的说明
- 要点：明确 MEV 的判定标准，避免审稿人继续追问“是语义匹配还是严格字符串匹配”。
- 将 A：笼统表述为“recover the masked content”或“semantic recoverability”。
- 改成 B："the recovered entity is compared against the original entity using a normalized lexical match: lowercase, remove punctuation and redundant spaces, and count a recovery only when the resulting surface form matches exactly"。
- 目的：让 MEV 的评价规则清晰、可复现、易于审稿人理解。

### 8. 候选选择、tie-breaking 与验证预算说明
- 要点：明确候选动作如何在同分情况下决策，以及为何使用中等规模的验证预算。
- 将 A：笼统说“if tie, choose the first candidate”或“we use repeated verification for all candidates”而没有说明成本与顺序。
- 改成 B："When multiple candidates tie, we choose the one with the higher TFV score; if still tied, we fall back to the earliest candidate in decoding order. If the sampled candidates are already consistent, the unanimous action is executed directly to save latency. The verification budget is fixed as K action candidates with P repeated checks per candidate, with the default operating point K=4, P=4 in the reported experiments."
- 目的：让方法在工程实现层面更可复现，同时说明为什么不继续增大 K / P 会带来边际收益递减。

### 9. 生成与验证使用同一 LLM Backbone 的说明
- 要点：澄清生成阶段与验证阶段是否使用不同模型，避免审稿人追问“是否存在额外训练 verifier”。
- 将 A：隐含地把验证视为“另一个独立的 checker”或没有说明模型来源。
- 改成 B："The same LLM backbone is reused for both candidate generation and verification; we do not introduce a separate learned verifier. Instead, the same model is prompted with a different judgment task and sampled repeatedly for verification."
- 目的：强调方法是 inference-time check，而不是依赖额外训练的验证器。

### 10. checklist 状态同步
- 要点：每次修订后，同步更新英文和中文版本的 checklist 完成状态。
- 处理方式：将已完成项目标记为 ✅，待处理项目标记为 ⏳，便于后续 Review 和跟踪。

## 2026-09-10

### 11. MEV 边缘场景的简短说明
- 要点：在方法描述中加入一段简短的 caveat，说明 MEV 在缺少明确实体或纯方向指令时可能较弱。
- 将 A：不讨论边缘情况，直接把 MEV 写成一个统一、普适的验证器。
- 改成 B："MEV is most informative when the instruction contains salient entity references; in highly abstract or purely directional instructions, the masked-entity signal may be weaker and the decision then relies more on TFV."
- 目的：保留方法的可解释性和严谨性，同时避免审稿人追问“MEV 是否在所有场景都成立”。

### 12. 训练阶段措辞收紧
- 要点：将过强的“增强准确性”式表述收敛为更稳妥的训练叙述，不夸大推理稳定性的提升。
- 将 A："enhancing the accuracy of its self-guided decision-making processes" 等强宣称。
- 改成 B："This procedure is intended to stabilize action selection by conditioning the model on structured intermediate reasoning, while keeping the training scheme lightweight and compatible with a fully local LLM backbone."
- 目的：适度保守，突出方法是轻量且稳定的，而不是泛化地声称显著提升。

### 13. 实验设置中的子集说明
- 要点：明确区分小规模 Val Unseen Subset 与完整验证集的使用范围。
- 将 A：将 subset 直接混同于主 benchmark 表述。
- 改成 B："the subset is used only for diagnostic ablations, while all benchmark comparisons in the main results table are reported on the full validation set unless otherwise stated."
- 目的：避免实验叙述上的歧义，并满足审稿人对 fairness / protocol 的关注。

### 14. revision log 与 checklist 同步更新
- 要点：每次修订都同步写入 revision log，并更新英文/中文 checklist 的状态。
- 处理方式：采用“随改随记”的方式，记载修改对象、原因和目的，确保版本控制清晰。
- 目的：保证修订过程可追踪，可审核，可推送到 GitHub 时作为备用说明。
