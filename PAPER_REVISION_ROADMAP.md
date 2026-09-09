# DV-VLN Revision Checklist

This checklist tracks the revision plan for the reviewer feedback. Each item will be updated in the separate file `jsen_edit.tex` first, and only after approval will the same change be applied to the main manuscript if needed.

## Phase 1: Clarify contribution and novelty
- ✅ Rephrase the abstract to reduce over-claiming and make the contribution more precise.
- ✅ Clarify the novelty boundary relative to Self-Verification / generate-then-verify methods.
- ✅ State explicitly what is newly designed for VLN, beyond adapting general verification ideas.
- ✅ Tighten the contribution list so it focuses on navigation-specific verification and re-ranking, not generic LLM reasoning.
- ✅ Ensure the Related Work text distinguishes prior general verification methods from VLN-specific design choices.

## Phase 2: Method details and technical clarity
- ⏳ Clarify the definition and role of Prediction–View Match–Action in the method section.
- ⏳ Explain the exact matching rule used in MEV and whether it is exact string match, normalized string match, or semantic matching.
- ⏳ State how candidate selection, tie-breaking, and verification budget are handled in practice.
- ⏳ Clarify whether the same LLM/backbone is used for generation and verification, or whether different models are involved.
- ⏳ Add a concise discussion of edge cases where MEV may be weak or less informative.

## Phase 3: Experimental protocol and fairness
- ⏳ Distinguish clearly between the compact ablation subset and the full validation set.
- ⏳ State explicitly which tables use the full validation set and which use the subset.
- ⏳ Add notes on data sources for baseline numbers and whether they are directly reported or reproduced.
- ⏳ Check all cross-dataset comparisons and clarify whether results correspond to zero-shot transfer or separate training.
- ⏳ Review the wording around “competitive,” “generalization,” and “significantly improves” to make claims more precise.

## Phase 4: Ablation and statistical validity
- ⏳ Add or clarify whether the ablation on the small subset is used only as a diagnostic experiment.
- ⏳ Report uncertainty or repeated-run statistics if the subset remains in the main paper.
- ⏳ Add an equal-budget comparison for TFV-only, MEV-only, and TFV+MEV to demonstrate complementarity.
- ⏳ Include a stronger ablation with random or non-matching action candidates to test MEV dependence on the action itself.
- ⏳ Separate the subset-based sensitivity analysis from the final benchmark results.

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
- ⏳ Remove or soften over-strong wording such as “significantly improves” unless backed by a narrow, precise comparison.
- ⏳ Re-check all figure/table references and ensure they match the actual printed text.
- ⏳ Fix minor grammar and wording issues in the English prose.
- ⏳ Ensure all figure annotations and score explanations are readable, especially Fig. 2 / Fig. 4 style examples.
- ⏳ Confirm the page header/year/template fields are consistent with the current manuscript metadata.

## Phase 8: Final pass before submission
- ⏳ Recompile the revision file and confirm no LaTeX errors.
- ⏳ Check that all reviewer concerns are directly answered in the revised text.
- ⏳ Confirm the final revision file is cleanly separated from the original manuscript.
- ⏳ Prepare a concise summary of the changes for commit / GitHub push.
- ⏳ Ask for approval before applying the accepted edits to the main manuscript.

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
