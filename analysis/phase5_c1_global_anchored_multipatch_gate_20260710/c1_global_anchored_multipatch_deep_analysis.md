# Phase5 C1 Global-Anchored Multi-Patch 深度分析

## 结论

[Decision] `c1_carrier_normalization_gate_failed`。Shared scales和validation-selected scale均明显越过退化预算；不追加 seeds、dropout follow-up、local-mask control或readout repair。

[Protocol Mismatch] Runner统一传入`learning_rate=1e-4`，ETTh2 A6 source preset实际为`5e-4`；因此ETTh2 A6不是source-faithful exact reproduction。ETTm1/Weather learning rate一致且metrics exact reproduce。

该 mismatch不改变失败裁决：ETTh2三arms在同一LR下仍可作controlled comparison，且C1相对fixed TimeAlign直接失败。相对既有source-faithful A6 official-last的独立结果如下：

| arm | dataset | mean_mse_vs_source_a6_pct | wins_vs_source_a6 | settings | max_mse_vs_source_a6_pct |
| --- | --- | --- | --- | --- | --- |
| gamp_p16s8 | ETTh2 | 6.0250 | 0 | 4 | 8.2957 |
| gamp_p16s8 | ETTm1 | 1.0977 | 0 | 4 | 1.7087 |
| gamp_p16s8 | Weather | 6.7817 | 0 | 4 | 7.7720 |
| gamp_p16s8 | ALL | 4.6348 | 0 | 12 | 8.2957 |
| gamp_p48s24 | ETTh2 | 2.4435 | 0 | 4 | 5.2796 |
| gamp_p48s24 | ETTm1 | 2.0465 | 0 | 4 | 2.5873 |
| gamp_p48s24 | Weather | 11.0225 | 0 | 4 | 12.2005 |
| gamp_p48s24 | ALL | 5.1708 | 0 | 12 | 12.2005 |

A6 reproduction artifact的跨数据集maximum absolute metric difference为`1.0e-02`；差异只来自上述ETTh2 LR mismatch。

## Training dynamics

| arm | dataset | best_epoch | best_val_mean_mse | last_val_mean_mse | last_vs_best_val_pct | last_train_loss | best_vs_last_test_mean_mse_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a6_clean | ETTh2 | 2 | 0.4015 | 0.4107 | 2.2931 | 0.4086 | 4.0946 |
| a6_clean | ETTm1 | 9 | 0.5996 | 0.5999 | 0.0509 | 0.3732 | -0.1527 |
| a6_clean | Weather | 8 | 0.4885 | 0.4895 | 0.2083 | 0.3303 | -0.0713 |
| gamp_p16s8 | ETTh2 | 1 | 0.3898 | 0.4402 | 12.9444 | 0.3686 | -1.0838 |
| gamp_p16s8 | ETTm1 | 2 | 0.6157 | 0.6457 | 4.8712 | 0.3505 | 1.7005 |
| gamp_p16s8 | Weather | 6 | 0.4984 | 0.5016 | 0.6443 | 0.3162 | -1.2815 |
| gamp_p48s24 | ETTh2 | 1 | 0.3859 | 0.4255 | 10.2654 | 0.3610 | 3.5312 |
| gamp_p48s24 | ETTm1 | 4 | 0.6086 | 0.6252 | 2.7328 | 0.3479 | -0.2723 |
| gamp_p48s24 | Weather | 4 | 0.4979 | 0.5030 | 1.0338 | 0.3105 | -2.5112 |

GAMP在ETTh2/ETTm1的train loss持续下降，但validation minima很早出现并随后恶化；这是明确overfitting evidence。Weather的validation gap较小，但test仍显著退化，说明regularization不是唯一解释。

## Scale stability

| dataset | validation_selected | p48s24_vs_p16s8_val_pct | test_preferred_last | p48s24_vs_p16s8_test_last_pct | test_preferred_best_val | p48s24_vs_p16s8_test_best_val_pct |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | gamp_p48s24 | -0.9961 | gamp_p48s24 | -3.0138 | gamp_p16s8 | 1.0658 |
| ETTm1 | gamp_p48s24 | -1.1541 | gamp_p16s8 | 0.9074 | gamp_p48s24 | -0.9734 |
| Weather | gamp_p48s24 | -0.1056 | gamp_p16s8 | 3.9051 | gamp_p16s8 | 2.4560 |

Validation在三个datasets都选择P48-S24，但test preference随dataset/selector变化；Weather validation差仅约0.1%，test却稳定偏向P16-S8。Scale selection不够稳定，不能作为dataset-specific carrier依据。

## Capacity and state-width attribution

| dataset | arm | candidate_vs_a6_parameters_pct | a6_readout_state_width | candidate_readout_state_width | candidate_vs_a6_state_width_pct |
| --- | --- | --- | --- | --- | --- |
| ETTh2 | gamp_p16s8 | 69.7732 | 1536 | 256 | -83.3333 |
| ETTh2 | gamp_p48s24 | 68.5445 | 1536 | 256 | -83.3333 |
| ETTm1 | gamp_p16s8 | 41.5689 | 256 | 256 | 0.0000 |
| ETTm1 | gamp_p48s24 | 40.5443 | 256 | 256 | 0.0000 |
| Weather | gamp_p16s8 | -45.7680 | 6144 | 256 | -95.8333 |
| Weather | gamp_p48s24 | -46.1605 | 6144 | 256 | -95.8333 |

C1 total parameters在ETTh2/ETTm1高于A6却仍退化，因此total capacity不是通用解释。另一方面，global-only readout把ETTh2/Weather state width从1536/6144压到256，Weather同时减少约46% active parameters；这构成readout/state bottleneck。ETTm1 state width未减少仍退化，说明bottleneck也不是唯一原因。

## Failure attribution

- `hypothesis_false`：不能证明所有统一multipatch carriers都不可行；
- `intervention_point_wrong`：可能，C1用一个随机global token替代了各dataset已验证的A6 hidden contract；
- `readout_or_head_design_wrong`：强支持，global-only D256 readout丢失ETTh2/Weather原有P*D state width；
- `optimization_or_numeric_pathology`：无divergence/OOM，但存在早期validation overfit；
- `capacity_control_explains`：仅能部分解释Weather，不能解释ETTh2/ETTm1。

## Dropout decision

Best-val仍整体退化3.75%，Weather退化5.49%，远超near-miss范围。更强dropout可能缓解train-validation gap，但无法单独修复state/readout contract；按预注册协议不追加dropout sweep。

## Carrier decision

恢复并冻结`A6-LBF-r256 + exact valid HPM [B,C,29,48]`。Forecast path允许source-faithful dataset hyperparameters；所有后续模块通过HPM获得统一local-token interface。继续修改readout/scale/dropout会把control-only cleanup扩张为新的architecture search，当前停止。

## 下一研究方向

1. 先完成Contribution 1的matched multi-prefix-vs-single-prefix supervision control，固定同一720-step architecture、dropout与selector，只改变objective。
2. StageB回到Step 2/3。优先重新审计B7 horizon-agnostic supervision allocation，因为现有multi-prefix loss对early steps有14.39x exposure，而Encoder/local retrieval routes已被多轮controls阻断。
3. B7只能先做continuous-prefix、benchmark-horizon-free diagnostic；若不能证明tail degradation与exposure imbalance存在因果关系，则暂停第二贡献搜索，进入论文收束。
