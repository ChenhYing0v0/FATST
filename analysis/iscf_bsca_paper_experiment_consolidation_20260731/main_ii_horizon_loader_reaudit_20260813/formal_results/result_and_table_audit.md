# Main II horizon-specific loader formal result and table audit

## 1. Decision

`Main_II_horizon_loader_reaudit_complete_pass_old_common_origin_table_superseded`。

[Fact] 旧 Main II 的 H96/H192/H336 是在 H720 test loader 的较小 common-origin tensor 上裁前缀。它保证同一 tensor 内的 prefix identity，却没有使用 horizon-specific official evaluation 在每个 H 下实际保留的 test origins。用户对旧表可能偏乐观的怀疑成立，因此旧 Main II 不再作为 canonical paper table。

[Fact] 新协议完整复用此前 horizon-specific 工作流产生的 H720 checkpoints；没有训练或选择任何新 checkpoint。对每个 baseline、dataset、H，evaluator 重建 official H-specific test loader，只把 history 和必要的已知 decoder context 输入 H720 model，future decoder slots 全零，取 output 前 H steps 与该 loader 的 H-step label 计算 MSE/MAE。

## 2. Complete-matrix audit

- external systems：7；datasets：7；horizons：4；
- checkpoint objects：63/63，unique SHA256：63/63；
- formal evaluations：252/252；
- external aggregate cells：196/196；
- reused current ISCF cells：28/28；
- final Main II cells：224/224，MSE/MAE scalars：448/448；
- H720 same-checkpoint continuity：49/49 pass；
- origin-count monotonicity：63/63 checkpoint surfaces pass；
- `drop_last`、`model_horizon=720`、`loader_horizon=H`、checkpoint immutability、input-only inference 与 finite metrics 均通过；
- negative results retained，未按 dataset/horizon/metric 选择性删除。

代表性 continuity check：DLinear/ECL H720 的新旧 MSE/MAE差分别约为 $-1.94\times10^{-16}$ 与 $-2.78\times10^{-16}$；同一 checkpoint 切换到 H96 official loader 后，计分 origins 从4541增至5165。这直接验证了“checkpoint 不变、evaluation surface 改正”。

## 3. 旧协议偏差的实证影响

对每个 external system 的21个短 horizon cells（7 datasets × H96/H192/H336），新 fixed-H loader 相对旧 H720-common-origin protocol 的平均变化如下：

| System | MSE 新值更高 | MAE 新值更高 | mean $\Delta$MSE | mean $\Delta$MAE |
| --- | ---: | ---: | ---: | ---: |
| TimeAlign | 17/21 | 13/21 | +0.01478 | +0.00768 |
| QDF | 18/21 | 17/21 | +0.01283 | +0.00637 |
| AMD | 17/21 | 12/21 | +0.01368 | +0.00727 |
| SimpleTM | 15/21 | 14/21 | +0.00987 | +0.00538 |
| iTransformer | 17/21 | 14/21 | +0.01332 | +0.00624 |
| PatchTST | 16/21 | 15/21 | +0.01424 | +0.00744 |
| DLinear | 18/21 | 15/21 | +0.01269 | +0.00629 |

[Strong Evidence] 七个 baselines 的平均短-horizon MSE/MAE均上升，且147个短-horizon MSE cells中118个上升、147个MAE cells中100个上升。这说明旧 H720 common-origin surface 对 one-model baseline 普遍偏乐观，而不是某一个模型的偶发差异。少数 cells 下降是不同 horizon loader 引入/移除 origins 后的真实样本组成效应，完整保留。

## 4. 新 Main II ranking

按统一三位小数后 distinct-value ranking：

- ISCF-BSCA：41/56 best、13/56 second，共54/56 top-2；
- TimeAlign：12/56 best、27/56 second；
- QDF：0/56 best、3/56 second；
- AMD：1/56 best、7/56 second；
- SimpleTM：3/56 best、2/56 second；
- iTransformer：0/56 best、0/56 second；
- PatchTST：2/56 best、4/56 second；
- DLinear：0/56 best、0/56 second。

ISCF-BSCA在28个 dataset–horizon cells上的macro MSE/MAE为`0.261911/0.307252`；TimeAlign为`0.277023/0.316259`。该结果支持 source-native one-model system competitiveness，但 external contracts仍不matched，因此不支持 BSCA、allocation 或 decoder transfer attribution。

## 5. Canonical artifacts

- protocol：`configs/iscf_bsca_main_ii_horizon_loader_protocol.json`；
- prelaunch：`analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_ii_horizon_loader_reaudit_20260813/design_and_prelaunch_gate.md`；
- result audit：`formal_results/aggregate_audit/main_ii_result_audit.json`；
- aggregate cells：`formal_results/aggregate_audit/main_ii_aggregate_cells.csv`；
- manuscript fragment：`formal_results/table/table_iscf_bsca_main_ii.tex`；
- standalone source：`formal_results/table/table_iscf_bsca_main_ii_standalone.tex`；
- review PDF：`output/pdf/iscf_bsca_main_ii_horizon_loader_20260813.pdf`。

旧 `main_ii_h720_prefix_20260808` artifacts继续保留作historical protocol evidence，但不再进入当前 paper registry。
