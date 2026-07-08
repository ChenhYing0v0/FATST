# Phase5 StageB B10-TSI-D Failure Attribution

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `B10-TCO` |
| `diagnostic_id` | `B10-TSI-D` |
| `current_step` | StageB Step 3：failure attribution and diagnostic redesign |
| `problem` | 分离 target-set 信息价值、intervention point、readout/head 稳定性 |
| `scope` | Frozen A6 encoder/basis；offline ridge diagnostic；不训练 forecasting model |

## Diagnostic Design

本诊断不再只测试 `frozen coeff -> Linear_s(coeff)`。它固定 A6 encoder 与 learned basis，
比较三个 feature sources：

- `coeff_late`: B10-TSI-C 对应的 late coefficient intervention；
- `memory_pool`: encoder patch memory 的 mean / last / std pooling，代表更早的 memory-level intervention；
- `memory_plus_coeff`: memory-level feature 与 A6 coeff 组合，用于检查 coeff 是否补充必要信息。

每个 feature source 都拟合 full target-set-specific readout 和 shrinkage target-set readout，
并和 `shared_control`、`pooled_multihead_control`、`wrong_target_control` 比较。输出 target 使用 basis segment 的
rank-truncated output row-space coordinates，避免 B10-TSI-C 的 full coefficient inverse 与
singular-value back-projection 病态。

## Summary

| dataset | feature_source | feature_dim | rowspace_rank | train_rows | val_rows | test_rows | shared_alpha | pooled_multihead_alpha | target_set_alpha | stabilized_alpha | stabilized_beta | base_mse | pooled_multihead_control_mse | target_set_aware_mse | stabilized_target_set_mse | wrong_target_control_mse | target_vs_pooled_multihead_reduction_pct | stabilized_vs_pooled_multihead_reduction_pct | target_vs_wrong_target_reduction_pct | pathology_flag | stabilized_pathology_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | coeff_late | 256 | 64 | 6000 | 3000 | 3000 | 1000000.000000 | 1000000.000000 | 1000000.000000 | 1000000.000000 | 0.100000 | 0.949506 | 5.158379 | 11.762199 | 5.292930 | 11.211186 | -128.021217 | -2.608404 | -4.914846 | true | true |
| ETTh2 | memory_pool | 96 | 64 | 6000 | 3000 | 3000 | 1000000.000000 | 1000000.000000 | 1000000.000000 | 1000000.000000 | 0.100000 | 0.949506 | 5.228000 | 11.917159 | 6.478744 | 11.190517 | -127.948724 | -23.923947 | -6.493371 | true | true |
| ETTh2 | memory_plus_coeff | 352 | 64 | 6000 | 3000 | 3000 | 1000000.000000 | 1000000.000000 | 1000000.000000 | 1000000.000000 | 0.100000 | 0.949506 | 5.320434 | 12.195825 | 6.739293 | 11.453174 | -129.226121 | -26.668100 | -6.484234 | true | true |
| ETTm1 | coeff_late | 256 | 64 | 6000 | 3000 | 3000 | 10000.000000 | 10000.000000 | 10000.000000 | 1000.000000 | 0.500000 | 0.740899 | 0.741537 | 0.744753 | 0.744602 | 0.750803 | -0.433739 | -0.413422 | 0.805746 | false | false |
| ETTm1 | memory_pool | 768 | 64 | 6000 | 3000 | 3000 | 100000.000000 | 100000.000000 | 10000.000000 | 10000.000000 | 1.000000 | 0.740899 | 0.741468 | 0.746172 | 0.746172 | 0.758016 | -0.634378 | -0.634378 | 1.562539 | false | false |
| ETTm1 | memory_plus_coeff | 1024 | 64 | 6000 | 3000 | 3000 | 100000.000000 | 100000.000000 | 100000.000000 | 10000.000000 | 0.500000 | 0.740899 | 0.741341 | 0.741906 | 0.743888 | 0.748509 | -0.076207 | -0.343562 | 0.882233 | false | false |
| Weather | coeff_late | 256 | 64 | 6000 | 3000 | 3000 | 10.000000 | 10.000000 | 10.000000 | 10.000000 | 1.000000 | 7.841946 | 8.093478 | 10.852270 | 10.852270 | 12.501423 | -34.086609 | -34.086609 | 13.191726 | false | false |
| Weather | memory_pool | 384 | 64 | 6000 | 3000 | 3000 | 1000.000000 | 100.000000 | 100.000000 | 100.000000 | 1.000000 | 7.841946 | 9.805772 | 19.385153 | 19.385153 | 21.190773 | -97.691243 | -97.691243 | 8.520783 | true | true |
| Weather | memory_plus_coeff | 640 | 64 | 6000 | 3000 | 3000 | 1000.000000 | 100.000000 | 100.000000 | 100.000000 | 1.000000 | 7.841946 | 9.908958 | 20.451551 | 20.451551 | 22.382812 | -106.394564 | -106.394564 | 8.628320 | true | true |

## Segment Detail

| dataset | feature_source | segment | base_mse | pooled_multihead_control_mse | target_set_aware_mse | stabilized_target_set_mse | wrong_target_control_mse | target_vs_pooled_multihead_reduction_pct | stabilized_vs_pooled_multihead_reduction_pct | target_vs_wrong_target_reduction_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | coeff_late | early_0_96 | 0.411479 | 7.980821 | 0.642511 | 6.785094 | 1.464119 | 91.949307 | 14.982511 | 56.116182 |
| ETTh2 | coeff_late | mid_96_192 | 0.515405 | 7.680061 | 1.746653 | 5.934454 | 14.214919 | 77.257303 | 22.729070 | 87.712537 |
| ETTh2 | coeff_late | late_192_336 | 0.663897 | 5.922840 | 10.442688 | 6.291408 | 41.862802 | -76.312184 | -6.222828 | 75.054972 |
| ETTh2 | coeff_late | tail_336_720 | 1.299642 | 3.535675 | 17.540823 | 4.385080 | 1.402664 | -396.109546 | -24.023818 | -1150.536670 |
| ETTh2 | memory_pool | early_0_96 | 0.411479 | 7.911761 | 0.648087 | 8.303066 | 1.523678 | 91.808558 | -4.945854 | 57.465587 |
| ETTh2 | memory_pool | mid_96_192 | 0.515405 | 7.697296 | 1.778166 | 7.731569 | 14.132756 | 76.898822 | -0.445261 | 87.418123 |
| ETTh2 | memory_pool | late_192_336 | 0.663897 | 5.965027 | 10.460810 | 7.630728 | 41.749301 | -75.369024 | -27.924449 | 74.943749 |
| ETTh2 | memory_pool | tail_336_720 | 1.299642 | 3.663350 | 17.815306 | 5.277463 | 1.412123 | -386.311856 | -44.061108 | -1161.597417 |
| ETTh2 | memory_plus_coeff | early_0_96 | 0.411479 | 8.082427 | 0.642165 | 8.705993 | 1.471258 | 92.054805 | -7.715084 | 56.352689 |
| ETTh2 | memory_plus_coeff | mid_96_192 | 0.515405 | 7.871400 | 1.751669 | 8.133359 | 14.585256 | 77.746409 | -3.327991 | 87.990138 |
| ETTh2 | memory_plus_coeff | late_192_336 | 0.663897 | 6.066945 | 10.731645 | 7.943157 | 42.803798 | -76.887125 | -30.925149 | 74.928289 |
| ETTh2 | memory_plus_coeff | tail_336_720 | 1.299642 | 3.712253 | 18.244346 | 5.447652 | 1.409149 | -391.462917 | -46.747873 | -1194.706961 |
| ETTm1 | coeff_late | early_0_96 | 0.535708 | 0.543872 | 0.554207 | 0.556526 | 0.546738 | -1.900406 | -2.326686 | -1.366150 |
| ETTm1 | coeff_late | mid_96_192 | 0.644897 | 0.648767 | 0.650760 | 0.660884 | 0.653894 | -0.307252 | -1.867698 | 0.479212 |
| ETTm1 | coeff_late | late_192_336 | 0.776668 | 0.778229 | 0.784591 | 0.786204 | 0.808398 | -0.817488 | -1.024753 | 2.944985 |
| ETTm1 | coeff_late | tail_336_720 | 0.802784 | 0.800386 | 0.800949 | 0.796951 | 0.804448 | -0.070295 | 0.429200 | 0.434976 |
| ETTm1 | memory_pool | early_0_96 | 0.535708 | 0.540269 | 0.564512 | 0.564512 | 0.551623 | -4.487156 | -4.487156 | -2.336455 |
| ETTm1 | memory_pool | mid_96_192 | 0.644897 | 0.649743 | 0.664061 | 0.664061 | 0.661294 | -2.203722 | -2.203722 | -0.418454 |
| ETTm1 | memory_pool | late_192_336 | 0.776668 | 0.778029 | 0.788422 | 0.788422 | 0.832303 | -1.335759 | -1.335759 | 5.272275 |
| ETTm1 | memory_pool | tail_336_720 | 0.802784 | 0.800989 | 0.796271 | 0.796271 | 0.805938 | 0.589031 | 0.589031 | 1.199407 |
| ETTm1 | memory_plus_coeff | early_0_96 | 0.535708 | 0.541427 | 0.549328 | 0.556943 | 0.543114 | -1.459282 | -2.865740 | -1.144125 |
| ETTm1 | memory_plus_coeff | mid_96_192 | 0.644897 | 0.649954 | 0.653380 | 0.664509 | 0.648093 | -0.527003 | -2.239353 | -0.815665 |
| ETTm1 | memory_plus_coeff | late_192_336 | 0.776668 | 0.777654 | 0.781489 | 0.785738 | 0.804014 | -0.493237 | -1.039578 | 2.801522 |
| ETTm1 | memory_plus_coeff | tail_336_720 | 0.802784 | 0.800549 | 0.797338 | 0.794775 | 0.804148 | 0.401056 | 0.721220 | 0.846866 |
| Weather | coeff_late | early_0_96 | 2.095566 | 3.279585 | 3.332794 | 3.332794 | 16.914243 | -1.622435 | -1.622435 | 80.295933 |
| Weather | coeff_late | mid_96_192 | 4.391323 | 5.077230 | 16.387687 | 16.387687 | 10.019242 | -222.768256 | -222.768256 | -63.562150 |
| Weather | coeff_late | late_192_336 | 9.365578 | 10.096235 | 11.196946 | 11.196946 | 18.341889 | -10.902190 | -10.902190 | 38.954239 |
| Weather | coeff_late | tail_336_720 | 9.569835 | 9.299979 | 11.219031 | 11.219031 | 9.828589 | -20.635016 | -20.635016 | -14.146910 |
| Weather | memory_pool | early_0_96 | 2.095566 | 6.596050 | 6.316355 | 6.316355 | 26.204573 | 4.240341 | 4.240341 | 75.895984 |
| Weather | memory_pool | mid_96_192 | 4.391323 | 8.174896 | 24.691924 | 24.691924 | 19.689845 | -202.045714 | -202.045714 | -25.404361 |
| Weather | memory_pool | late_192_336 | 9.365578 | 11.934199 | 15.864158 | 15.864158 | 46.618290 | -32.930227 | -32.930227 | 65.970098 |
| Weather | memory_pool | tail_336_720 | 9.569835 | 10.217762 | 22.646033 | 22.646033 | 10.777236 | -121.633988 | -121.633988 | -110.128400 |
| Weather | memory_plus_coeff | early_0_96 | 2.095566 | 6.871748 | 6.553376 | 6.553376 | 30.667215 | 4.633051 | 4.633051 | 78.630677 |
| Weather | memory_plus_coeff | mid_96_192 | 4.391323 | 8.412999 | 29.087110 | 29.087110 | 20.337019 | -245.740081 | -245.740081 | -43.025433 |
| Weather | memory_plus_coeff | late_192_336 | 9.365578 | 12.154966 | 16.544436 | 16.544436 | 49.170098 | -36.112562 | -36.112562 | 66.352647 |
| Weather | memory_plus_coeff | tail_336_720 | 9.569835 | 10.199998 | 23.232374 | 23.232374 | 10.777927 | -127.768420 | -127.768420 | -115.555115 |

## Failure Attribution

[Fact] 本诊断把 B10-TSI-C 的单一 late readout 拆成三个 feature sources：
`coeff_late`、`memory_pool`、`memory_plus_coeff`。每个 source 使用相同的 rank-truncated
basis row-space target、相同 alpha validation、相同 no-target controls。

- `coeff_late` target vs pooled control mean reduction: `-54.1805%`; stabilized target vs pooled: `-12.3695%`; stabilized pathology datasets: `1`.
- `memory_pool` target vs pooled control mean reduction: `-75.4248%`; stabilized target vs pooled: `-40.7499%`; stabilized pathology datasets: `2`.
- `memory_plus_coeff` target vs pooled control mean reduction: `-78.5656%`; stabilized target vs pooled: `-44.4687%`; stabilized pathology datasets: `2`.

## Decision

[Decision] `B10-TSI-D` 仍出现 memory-level pathology，不能否定 target-set-aware 方向。

[Next] 必须先修正 diagnostic 的 feature/readout 稳定性，再讨论 B10 rollback。
