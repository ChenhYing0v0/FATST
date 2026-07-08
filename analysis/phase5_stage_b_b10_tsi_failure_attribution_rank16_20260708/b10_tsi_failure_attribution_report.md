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
| ETTh2 | coeff_late | 256 | 16 | 6000 | 3000 | 3000 | 1000000.000000 | 1000000.000000 | 1000000.000000 | 1000000.000000 | 0.100000 | 0.949506 | 5.152621 | 11.731373 | 5.286296 | 11.181167 | -127.677800 | -2.594309 | -4.920830 | true | true |
| ETTh2 | memory_pool | 96 | 16 | 6000 | 3000 | 3000 | 1000000.000000 | 1000000.000000 | 1000000.000000 | 1000000.000000 | 0.100000 | 0.949506 | 5.222238 | 11.886306 | 6.470586 | 11.160720 | -127.609445 | -23.904466 | -6.501242 | true | true |
| ETTh2 | memory_plus_coeff | 352 | 16 | 6000 | 3000 | 3000 | 1000000.000000 | 1000000.000000 | 1000000.000000 | 1000000.000000 | 0.100000 | 0.949506 | 5.314505 | 12.164103 | 6.730748 | 11.422438 | -128.884963 | -26.648634 | -6.493057 | true | true |
| ETTm1 | coeff_late | 256 | 16 | 6000 | 3000 | 3000 | 10000.000000 | 10000.000000 | 10000.000000 | 1000.000000 | 0.500000 | 0.740899 | 0.740999 | 0.742813 | 0.743345 | 0.747955 | -0.244791 | -0.316660 | 0.687500 | false | false |
| ETTm1 | memory_pool | 768 | 16 | 6000 | 3000 | 3000 | 100000.000000 | 100000.000000 | 10000.000000 | 10000.000000 | 1.000000 | 0.740899 | 0.741215 | 0.743569 | 0.743569 | 0.754058 | -0.317622 | -0.317622 | 1.391026 | false | false |
| ETTm1 | memory_plus_coeff | 1024 | 16 | 6000 | 3000 | 3000 | 100000.000000 | 100000.000000 | 10000.000000 | 10000.000000 | 0.500000 | 0.740899 | 0.740986 | 0.744765 | 0.742667 | 0.755656 | -0.509961 | -0.226914 | 1.441266 | false | false |
| Weather | coeff_late | 256 | 16 | 6000 | 3000 | 3000 | 10.000000 | 10.000000 | 10.000000 | 10.000000 | 1.000000 | 7.841946 | 7.865332 | 8.851700 | 8.851700 | 9.974586 | -12.540713 | -12.540713 | 11.257466 | false | false |
| Weather | memory_pool | 384 | 16 | 6000 | 3000 | 3000 | 100.000000 | 10.000000 | 10.000000 | 10.000000 | 1.000000 | 7.841946 | 9.035602 | 15.530508 | 15.530508 | 16.898100 | -71.881268 | -71.881268 | 8.093169 | false | false |
| Weather | memory_plus_coeff | 640 | 16 | 6000 | 3000 | 3000 | 100.000000 | 10.000000 | 10.000000 | 10.000000 | 1.000000 | 7.841946 | 9.156437 | 16.685419 | 16.685419 | 18.416166 | -82.226115 | -82.226115 | 9.397979 | true | true |

## Segment Detail

| dataset | feature_source | segment | base_mse | pooled_multihead_control_mse | target_set_aware_mse | stabilized_target_set_mse | wrong_target_control_mse | target_vs_pooled_multihead_reduction_pct | stabilized_vs_pooled_multihead_reduction_pct | target_vs_wrong_target_reduction_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | coeff_late | early_0_96 | 0.411479 | 7.970354 | 0.629176 | 6.774336 | 1.439902 | 92.106041 | 15.005830 | 56.304232 |
| ETTh2 | coeff_late | mid_96_192 | 0.515405 | 7.669892 | 1.721869 | 5.924326 | 14.153148 | 77.550280 | 22.758679 | 87.834020 |
| ETTh2 | coeff_late | late_192_336 | 0.663897 | 5.915848 | 10.398117 | 6.283344 | 41.779779 | -75.767140 | -6.212062 | 75.112083 |
| ETTh2 | coeff_late | tail_336_720 | 1.299642 | 3.532659 | 17.509270 | 4.380885 | 1.399009 | -395.639916 | -24.010956 | -1151.548340 |
| ETTh2 | memory_pool | early_0_96 | 0.411479 | 7.901471 | 0.634135 | 8.290305 | 1.499208 | 91.974472 | -4.921041 | 57.702016 |
| ETTh2 | memory_pool | mid_96_192 | 0.515405 | 7.687149 | 1.752896 | 7.718782 | 14.071776 | 77.197064 | -0.411500 | 87.543181 |
| ETTh2 | memory_pool | late_192_336 | 0.663897 | 5.958007 | 10.416483 | 7.620607 | 41.667572 | -74.831663 | -27.905303 | 75.000985 |
| ETTh2 | memory_pool | tail_336_720 | 1.299642 | 3.660288 | 17.783885 | 5.272349 | 1.408265 | -385.860284 | -44.041912 | -1162.822175 |
| ETTh2 | memory_plus_coeff | early_0_96 | 0.411479 | 8.071738 | 0.628806 | 8.692300 | 1.446601 | 92.209781 | -7.688084 | 56.532159 |
| ETTh2 | memory_plus_coeff | mid_96_192 | 0.515405 | 7.860991 | 1.726173 | 8.119915 | 14.521611 | 78.041275 | -3.293786 | 88.113072 |
| ETTh2 | memory_plus_coeff | late_192_336 | 0.663897 | 6.059776 | 10.685592 | 7.932683 | 42.718722 | -76.336404 | -30.907186 | 74.986162 |
| ETTh2 | memory_plus_coeff | tail_336_720 | 1.299642 | 3.709099 | 18.211852 | 5.442343 | 1.405497 | -391.004764 | -46.729524 | -1195.758654 |
| ETTm1 | coeff_late | early_0_96 | 0.535708 | 0.541497 | 0.544390 | 0.549688 | 0.544559 | -0.534266 | -1.512616 | 0.030984 |
| ETTm1 | coeff_late | mid_96_192 | 0.644897 | 0.647068 | 0.647707 | 0.657958 | 0.647356 | -0.098735 | -1.683019 | -0.054272 |
| ETTm1 | coeff_late | late_192_336 | 0.776668 | 0.777352 | 0.780035 | 0.782661 | 0.803349 | -0.345112 | -0.682942 | 2.902093 |
| ETTm1 | coeff_late | tail_336_720 | 0.802784 | 0.800725 | 0.802237 | 0.798363 | 0.803181 | -0.188835 | 0.294921 | 0.117598 |
| ETTm1 | memory_pool | early_0_96 | 0.535708 | 0.539076 | 0.552555 | 0.552555 | 0.549082 | -2.500324 | -2.500324 | -0.632511 |
| ETTm1 | memory_pool | mid_96_192 | 0.644897 | 0.648658 | 0.659193 | 0.659193 | 0.652570 | -1.624121 | -1.624121 | -1.014899 |
| ETTm1 | memory_pool | late_192_336 | 0.776668 | 0.777444 | 0.782287 | 0.782287 | 0.824710 | -0.623007 | -0.623007 | 5.143966 |
| ETTm1 | memory_pool | tail_336_720 | 0.802784 | 0.801302 | 0.797897 | 0.797897 | 0.804179 | 0.424995 | 0.424995 | 0.781240 |
| ETTm1 | memory_plus_coeff | early_0_96 | 0.535708 | 0.539808 | 0.553770 | 0.549902 | 0.550436 | -2.586480 | -1.869886 | -0.605674 |
| ETTm1 | memory_plus_coeff | mid_96_192 | 0.644897 | 0.648596 | 0.660353 | 0.661224 | 0.654909 | -1.812740 | -1.946971 | -0.831308 |
| ETTm1 | memory_plus_coeff | late_192_336 | 0.776668 | 0.776938 | 0.783504 | 0.782336 | 0.829718 | -0.845151 | -0.694801 | 5.569843 |
| ETTm1 | memory_plus_coeff | tail_336_720 | 0.802784 | 0.800896 | 0.799089 | 0.796344 | 0.804374 | 0.225631 | 0.568380 | 0.657025 |
| Weather | coeff_late | early_0_96 | 2.095566 | 2.753158 | 2.239773 | 2.239773 | 7.022211 | 18.647150 | 18.647150 | 68.104448 |
| Weather | coeff_late | mid_96_192 | 4.391323 | 4.885282 | 8.446877 | 8.446877 | 6.642594 | -72.904608 | -72.904608 | -27.162325 |
| Weather | coeff_late | late_192_336 | 9.365578 | 9.729205 | 10.035887 | 10.035887 | 15.426950 | -3.152174 | -3.152174 | 34.945748 |
| Weather | coeff_late | tail_336_720 | 9.569835 | 9.189435 | 10.161818 | 10.161818 | 9.501041 | -10.581535 | -10.581535 | -6.954784 |
| Weather | memory_pool | early_0_96 | 2.095566 | 5.095009 | 2.816643 | 2.816643 | 13.945741 | 44.717603 | 44.717603 | 79.802843 |
| Weather | memory_pool | mid_96_192 | 4.391323 | 7.301130 | 15.568717 | 15.568717 | 11.242677 | -113.237077 | -113.237077 | -38.478727 |
| Weather | memory_pool | late_192_336 | 9.365578 | 11.077409 | 11.727971 | 11.727971 | 41.756248 | -5.872867 | -5.872867 | 71.913256 |
| Weather | memory_pool | tail_336_720 | 9.569835 | 9.688691 | 20.125373 | 20.125373 | 9.728239 | -107.720250 | -107.720250 | -106.875811 |
| Weather | memory_plus_coeff | early_0_96 | 2.095566 | 5.551234 | 2.921854 | 2.921854 | 17.558569 | 47.365686 | 47.365686 | 83.359385 |
| Weather | memory_plus_coeff | mid_96_192 | 4.391323 | 7.641055 | 18.587993 | 18.587993 | 12.411503 | -143.264742 | -143.264742 | -49.764240 |
| Weather | memory_plus_coeff | late_192_336 | 9.365578 | 11.357525 | 12.708835 | 12.708835 | 46.240807 | -11.897927 | -11.897927 | 72.515975 |
| Weather | memory_plus_coeff | tail_336_720 | 9.569835 | 9.611175 | 21.141886 | 21.141886 | 9.697491 | -119.971920 | -119.971920 | -118.013973 |

## Failure Attribution

[Fact] 本诊断把 B10-TSI-C 的单一 late readout 拆成三个 feature sources：
`coeff_late`、`memory_pool`、`memory_plus_coeff`。每个 source 使用相同的 rank-truncated
basis row-space target、相同 alpha validation、相同 no-target controls。

- `coeff_late` target vs pooled control mean reduction: `-46.8211%`; stabilized target vs pooled: `-5.1506%`; stabilized pathology datasets: `1`.
- `memory_pool` target vs pooled control mean reduction: `-66.6028%`; stabilized target vs pooled: `-32.0345%`; stabilized pathology datasets: `1`.
- `memory_plus_coeff` target vs pooled control mean reduction: `-70.5403%`; stabilized target vs pooled: `-36.3672%`; stabilized pathology datasets: `2`.

## Decision

[Decision] `B10-TSI-D` 仍出现 memory-level pathology，不能否定 target-set-aware 方向。

[Next] 必须先修正 diagnostic 的 feature/readout 稳定性，再讨论 B10 rollback。
