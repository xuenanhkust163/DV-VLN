# DV-VLN Revision Checklist

This checklist tracks the revision plan for the reviewer feedback. Each item will be updated in the separate file `jsen_edit.tex` first, and only after approval will the same change be applied to the main manuscript if needed.

## Phase 1: Clarify contribution and novelty
- ✅ Rephrase the abstract to reduce over-claiming and make the contribution more precise.
- ✅ Clarify the novelty boundary relative to Self-Verification / generate-then-verify methods.
- ✅ State explicitly what is newly designed for VLN, beyond adapting general verification ideas.
- ✅ Tighten the contribution list so it focuses on navigation-specific verification and re-ranking, not generic LLM reasoning.
- ✅ Ensure the Related Work text distinguishes prior general verification methods from VLN-specific design choices.
- ✅ Complete a conservative wording pass across the introduction and related work to remove over-strong claims while keeping the reported results unchanged.

## Phase 2: Method details and technical clarity
- ✅ Clarify the definition and role of Prediction–View Match–Action in the method section.
- ✅ Explain the exact matching rule used in MEV and whether it is exact string match, normalized string match, or semantic matching.
- ✅ State how candidate selection, tie-breaking, and verification budget are handled in practice.
- ✅ Clarify that the same LLM/backbone is used for generation and verification, without introducing a separate learned verifier.
- ✅ Add a concise discussion of edge cases where MEV may be weak or less informative.

## Phase 3: Experimental protocol and fairness
- ✅ Distinguish clearly between the compact ablation subset and the full validation set.
- ✅ State explicitly which tables use the full validation set and which use the subset.
- ✅ Add notes on data sources for baseline numbers and whether they are directly reported or reproduced.
- ✅ Check all cross-dataset comparisons and clarify whether results correspond to zero-shot transfer or separate training.
- ✅ Review the wording around “competitive,” “generalization,” and “significantly improves” to make claims more precise.

## Phase 4: Ablation and statistical validity
- ✅ Clarify that the ablation on the small subset is used only as a diagnostic experiment, not as a standalone benchmark claim.
- ✅ Clarify that the subset is not intended to provide a separate uncertainty estimate or repeated-run statistics in the current revision.
- ✅ Clarify the equal-budget comparison pattern for TFV-only, MEV-only, and TFV+MEV, showing their complementary contribution within the fixed protocol.
- ✅ Clarify the interpretation of the MEV analysis without introducing a new experiment: the current text treats MEV as a diagnostic signal under the evaluated action set rather than a claim of broader dependence on random or non-matching candidates.
- ✅ Separate the subset-based sensitivity analysis from the final benchmark results and state that the main claims remain tied to the full validation protocol.

## Phase 5: Efficiency and reproducibility
- ⏳ Add a latency / inference-time analysis or at least a clear statement that the current paper reports no wall-clock measurement.
- ⏳ Report the computational cost in a more transparent way: candidate count, verification budget, and per-step cost.
- ⏳ Document generation hyperparameters: sampling temperature, top-p, repeated verification budget, etc.
- ⏳ Include the BLIP / CLIP / entity-extraction model details and checkpoints used.
- ⏳ Clarify how duplicated candidates, entity extraction, and same-action candidates are handled.

## Phase 6: Evaluation metric correctness
- ⏳ Re-check the SPL definition and ensure the formula is written correctly.
- ⏳ Verify the REVERIE evaluation description and whether it is navigation-only or includes object grounding.
- ⏳ Check the wording for R2R / RxR / REVERIE benchmark comparison consistency.
- ⏳ Ensure the narrative matches the exact official evaluator definitions.

## Phase 7: Presentation and English polishing
- ✅ Remove or soften over-strong wording such as “significantly improves” unless backed by a narrow, precise comparison.
- ✅ Remove residual over-strong wording in the abstract, results discussion, and conclusion so the claims stay within the evaluated protocol.
- ✅ Perform a final light pass on the introduction and discussion wording to keep the language conservative without altering the actual results.
- ⏳ Re-check all figure/table references and ensure they match the actual printed text.
- ✅ Fix minor grammar and wording issues in the English prose.
- ⏳ Ensure all figure annotations and score explanations are readable, especially Fig. 2 / Fig. 4 style examples.
- ⏳ Confirm the page header/year/template fields are consistent with the current manuscript metadata.

## Phase 8: Final pass before submission
- ⏳ Recompile the revision file and confirm no LaTeX errors.
- ⏳ Check that all reviewer concerns are directly answered in the revised text.
- ⏳ Confirm the final revision file is cleanly separated from the original manuscript.
- ⏳ Prepare a concise summary of the changes for commit / GitHub push.
- ⏳ Ask for approval before applying the accepted edits to the main manuscript.

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
## Proposed order of execution
1. Abstract + contribution list
2. Novelty / Related Work clarification
3. MEV / TFV technical definition and matching rule
4. Experimental protocol and benchmark explanations
5. Ablation and fairness issues
6. Efficiency and reproducibility details
7. Metric correctness and English polishing
8. Final compile and final approval

## Working rule
- Edit only `jsen_edit.tex` first.
- Do not overwrite the original `jsen.tex` unless explicitly approved.
- After each small update, pause for review and approval before proceeding to the next block.
