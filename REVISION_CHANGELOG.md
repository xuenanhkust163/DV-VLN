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

### 15. 结论段落的保守化修订
- 要点：把结论语气从“具有普遍优势”收紧为“在当前 benchmark 与协议下的有效性提升”。
- 将 A："the experimental results show that this verification step improves decision reliability ... while remaining competitive ..."
- 改成 B："the current results suggest that this verification step provides a useful reliability gain under the evaluated protocol, while remaining within a reasonable range of representative cross-modal systems under a lighter language-only setup."
- 目的：不改变实验数据，但显著降低 novelty weak 和 over-claiming 的审稿风险，同时保留方法的可信表述边界。

### 16. generalization 叙述的保守化修订
- 要点：把“generalizes well across datasets”收紧为“remains competitive across the studied settings under the current protocol”。
- 将 A："DV-VLN generalizes well across datasets with very different instruction styles."
- 改成 B："DV-VLN remains competitive across datasets with different instruction styles under the current training and evaluation protocol."
- 目的：在不重做实验的前提下，保留可证明的结论范围，减少审稿人将方法概括为泛化能力过强的质疑。

### 17. 引言中“single-shot decision”论证的保守化
- 要点：把“it is not enough to ask an LLM to output a single best action”这一类强断言改成更稳妥的引导性表述。
- 将 A："This shows that for VLN, it is not enough to ask an LLM to output a single best action; instead, we should first generate diverse candidate actions..."
- 改成 B："This suggests that, in VLN, it may be helpful to consider multiple candidate actions and check them against the instruction, history, and current observations before execution, rather than relying on a single direct decision alone."
- 目的：保留动机和解释逻辑，但避免在引言里给出过强的绝对判断，降低审稿上的挑刺余地。

### 18. 贡献列表的保守化
- 要点：将“improves navigation robustness over direct prediction and sampling-only baselines”收成“在 evaluated language-only protocol 下提供 useful reliability gain”。
- 将 A："showing that verification-guided re-ranking improves navigation robustness over direct prediction and sampling-only baselines under standard language-only settings."
- 改成 B："the current results suggest that verification-guided re-ranking provides a useful reliability gain over direct prediction and sampling-only baselines under the evaluated language-only protocol."
- 目的：在不改变实验结果的前提下，保持贡献描述可证实且更稳妥，避免审稿人将其解读为广义泛化性结论。

### 19. 批量保守措辞修订
- 要点：在引言、方法和相关工作中，对“new path / end-to-end / strongly / significantly / bottleneck / impressive”等强表述进行保守化处理。
- 处理方式：将其改为“promising direction / built on the same backbone / can influence / useful gain / practical bottleneck / practical balance”等更稳妥的说法。
- 目的：保证文稿在不重做实验、不改结果的前提下，仍能保持审稿友好、边界清晰、结论可证明。

### 20. 最终收口版保守化
- 要点：继续收紧摘要、结果讨论和结论中的剩余高强度措辞，尤其关注 “improves robustness / clear gains / substantially outperform / strong generalization” 等句式。
- 处理方式：改成更稳健、可验证的表述，如 “can provide a useful gain / remains competitive / improves over baseline under the evaluated protocol / practical alternative”。
- 目的：把最终稿的表述压到“审稿友好、但仍保留真实实验证据”的安全区间，不再扩大结论范围。

### 21. 继续小幅收口
- 要点：对 Introduction 和 Results Discussion 中仍偏强的文句再做轻量修正，避免出现“可被解释为更强泛化”的语气。
- 处理方式：将 “may be helpful” 改为 “can be useful”，把 “This suggests the benefit” 改为 “This suggests that ... can be beneficial under the evaluation setting considered here”。
- 目的：保持审稿友好、边界清晰，并在不改变实验结果的前提下进一步降低过强表述的风险。

### 22. 第三阶段：实验协议与公平性完成
- 要点：完成 Phase 3 的余项，包括 baseline 来源说明、跨数据集比较边界说明，以及对公平性措辞的收紧。
- 修改内容：在 Experimental Setup 中说明基线数值来自原论文公开协议，并明确不声称跨数据集 zero-shot transfer；在 Generalization section 中说明各数据集按各自协议训练和评测。
- 目的：让审稿人明确知道比较是“同协议内对照”，而不是扩展到更强泛化结论；同时保持实验层面不改动。

### 23. 第四阶段：消融与统计有效性补齐（说明性版）
- 要点：在不新增实验的前提下，把 Phase 4 的关键边界补全：诊断子集的用途、固定预算下的解释、以及与主 benchmark 的分界。
- 修改内容：在 Ablation Study 中明确说明小规模子集用于诊断性分析，而非独立 benchmark；强调当前结论只在已评估协议和固定推理预算下成立；说明 TFV/MEV 互补性是本协议内的观察，不扩展为全局统计结论。
- 目的：保证审稿人能接受消融结论，但不要求作者在当前版本中补做新的实验或重跑数据。
