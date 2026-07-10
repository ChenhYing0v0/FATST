# Phase5 StageB C0 ETTm1 Encoder Control 深度分析

## 研究裁决

[Decision] `patch_num_performance_defect_not_supported`。Matched P5 在全部 16 个 horizon-selector-dropout comparisons 中都输给 P1，因此预注册 gate 不授权追加 seeds。

[Decision] `dropout_0.9_not_a_general_protocol_defect`。将 P1 dropout 降到 0.2 后，mean MSE 变化 +0.79% (last) 与 +0.34% (best-val)，没有改善 unified P1 carrier。

[Decision] `checkpoint_selector_not_explanatory`。Patch effect 在 last 与 best-val 下方向一致且均为 0/4 wins；个别 arm 存在 selector sensitivity，但不能反转结论。

[Fact] P1-D256-F256-drop0.9 official-last control 精确复现此前 clean A6 ETTm1 metrics，MSE/MAE maximum absolute difference 为 0.0e+00。

## Protocol sensitivity

| comparison | selector | relative_mse_pct | relative_mae_pct |
| --- | --- | --- | --- |
| dropout_d02_vs_d09_p1 | last | 0.7891 | 0.8247 |
| dropout_d02_vs_d09_p1 | best_val | 0.3406 | 0.8476 |
| dropout_d02_vs_d09_p5 | last | -1.4234 | -0.0343 |
| dropout_d02_vs_d09_p5 | best_val | -1.2609 | -0.0256 |
| p5_f2048_vs_f256_d09 | last | -0.3033 | -0.2049 |
| p5_f2048_vs_f256_d09 | best_val | -0.3801 | -0.2221 |

负值表示 candidate 更优。降低 dropout 对 P5 有帮助，尤其是短 horizon，但修正后的 P5 仍然弱于对应 P1。

## Training 与 selector dynamics

| arm | best_epoch | best_val_mean_mse | last_val_mean_mse | last_vs_best_val_pct | last_train_loss | best_vs_last_test_mean_mse_pct |
| --- | --- | --- | --- | --- | --- | --- |
| p1_d256_f256_d09 | 9 | 0.5996 | 0.5999 | 0.0509 | 0.3732 | -0.1527 |
| p1_d384_f96_d09 | 7 | 0.6045 | 0.6058 | 0.2157 | 0.3780 | -0.0575 |
| p5_d52_f256_d09 | 8 | 0.6192 | 0.6199 | 0.1178 | 0.3786 | -0.1194 |
| p5_d52_f2048_d09 | 9 | 0.6175 | 0.6179 | 0.0583 | 0.3758 | -0.1966 |
| p1_d256_f256_d02 | 3 | 0.6039 | 0.6134 | 1.5714 | 0.3482 | -0.5917 |
| p5_d52_f2048_d02 | 9 | 0.6181 | 0.6185 | 0.0674 | 0.3596 | -0.0314 |

所有 runs 均正常优化且没有 divergence。Validation minima 出现在不同 epochs，但显式 best-val evaluation 不改变 architecture ranking。

## Encoder 风险裁决

1. `patch_num=1`：不是已证实的 defect。它是 full-window global compression bias；所有受测 P5 controls 更差。
2. Global width：P1-D384 没有收益，因此没有证据把 256-dimensional token 视为 bottleneck。
3. Cross-patch mixing：accepted P1 每个 channel 只有一个 full-window token，因此不存在待混合的 local patch axis；720-to-D projection 与 residual MLP 已产生 material cross-region interactions。只有引入 P5 independent tokens 后，缺少 mixing 才成为 design concern。
4. Dropout 0.9：dropout 位于 residual MLP correction branch 内，identity token path 始终保留，evaluation 时 dropout 关闭，因此它不等于丢弃 90% history representation。返回结果不支持降低 P1 dropout。
5. `official-last`：accepted P1 validation drift 仅 0.05%，best-val 使 test mean MSE 改变 -0.15%，无法解释 ETTm1 结果；但这不消除之前观察到的 cross-dataset selector risk。
6. Unused `proj_x`：仍是 code/parameter-accounting debt，但不影响 forward 或 metrics。以后只应通过 exact-equivalence cleanup 移除，不应作为研究实验。

[Conclusion] 当前 ETTm1 P1 Encoder 没有被证实存在理论无效性。C0 解决的是 implementation/control question；它本身没有完成 strong architecture-only paper claim 所需的独立 unified-vs-fixed fair-task confirmation。

## H720 disjoint-segment consistency

| comparison | selector | segment_wins | segment_count | min_relative_mse_pct | max_relative_mse_pct |
| --- | --- | --- | --- | --- | --- |
| patch_matched_d09 | last | 0 | 8 | 2.0316 | 4.9825 |
| patch_matched_d09 | best_val | 0 | 8 | 2.0392 | 5.0060 |
| patch_matched_d02 | last | 0 | 8 | 0.8549 | 2.7267 |
| patch_matched_d02 | best_val | 0 | 8 | 1.4413 | 3.8732 |

Matched P5 在 H720 的 disjoint segments 中一个也没有获胜，因此退化不是 cumulative-prefix averaging 或某个孤立 future region 导致的。

## Failure attribution

- `hypothesis_false`：不支持 ETTm1 P1 是 performance defect 这一窄假设。
- `intervention_point_wrong`：对 P5 no-mix 仍然可能，因为 frozen P1 具有 material cross-patch interactions。
- `readout_or_head_design_wrong`：仍可能；把 independent P5 tokens 直接 flatten 到 coefficient head，无法保持 P1 nonlinear global computation。
- `optimization_or_numeric_pathology`：不支持；所有 arms 训练稳定且 metrics 有限。
- `capacity_control_explains`：不足。P5 d_ff 从 256 增至 2048 后，mean MSE 仅变化 -0.30% (last) 与 -0.38% (best-val)，没有一致恢复。

## 下一步裁决

保留 `P=1,D=256,dropout=0.9` 作为 accepted ETTm1 A6 carrier setting。关闭 patch-defect route，不运行 seeds 2022/2023。C0 不需要单独 mixer control：它只能检查 P5 能否恢复 P1 interaction capacity，不能再检查 inherited P1 是否有缺陷。StageB 应回到 Step 2/3 或暂停，不继续叠加 Encoder mechanisms。

## Statistic definitions

- `relative_mse_pct=(candidate/baseline-1)*100`; negative favors candidate.
- `last_vs_best_val_pct=(last_validation/best_validation-1)*100`.
- `best_vs_last_test_mean_mse_pct` is the arithmetic mean of four per-horizon relative MSE changes.
- Segment results 使用 `target_horizon=720` artifact 中记录的 disjoint regions。
