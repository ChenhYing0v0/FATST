# Phase5 StageB B13-FUCO-A Future-Unit Granularity Diagnostic

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `B13-FUCO` |
| `diagnostic_id` | `B13-FUCO-A` |
| `current_step` | Step 2/3：large-unit granularity stability |
| `problem` | A6 global coefficient 是否在较大的 benchmark-independent future units 上仍承受稳定的方向压力 |
| `decision` | `partial_pass_large_unit_granularity_robust` |

## Scope

- main unit sizes: `120/144/180/240`;
- coarse control: `360`;
- datasets: `ETTh2/ETTm1/Weather`;
- clean A6 checkpoint, train split, checkpoint-local gradients;
- rank-32 A6 basis subspace geometry control;
- no model training and no residual fitting.

## Summary

| dataset | unit_size | unit_count | role | mean_pairwise_cosine | first_last_cosine | adjacent_cosine | far_cosine | shared_alignment_efficiency | bootstrap_mean_pairwise_p95 | bootstrap_first_last_p95 | basis_adjacent_overlap | basis_far_overlap | gradient_basis_pair_spearman | robust_support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 120 | 6 | main | 0.1218 | 0.0234 | 0.2371 | 0.0459 | 0.2781 | 0.1336 | 0.0293 | 0.1426 | 0.1348 | 0.6643 | yes |
| ETTh2 | 144 | 5 | main | 0.1226 | 0.0300 | 0.2076 | 0.0353 | 0.3039 | 0.1352 | 0.0403 | 0.1453 | 0.1316 | 0.7333 | yes |
| ETTh2 | 180 | 4 | main | 0.1344 | 0.0341 | 0.2057 | 0.0631 | 0.3540 | 0.1499 | 0.0477 | 0.1420 | 0.1316 | 0.7714 | yes |
| ETTh2 | 240 | 3 | main | 0.1502 | 0.0587 | 0.1959 | 0.0587 | 0.4313 | 0.1708 | 0.0778 | 0.1421 | 0.1406 | 0.5000 | yes |
| ETTh2 | 360 | 2 | coarse_control | 0.1684 | 0.1684 | 0.1684 | nan | 0.5637 | 0.2033 | 0.2017 | 0.1583 | nan | nan | no |
| ETTm1 | 120 | 6 | main | 0.1520 | 0.0604 | 0.1934 | 0.1112 | 0.2957 | 0.1612 | 0.0974 | 0.1507 | 0.1511 | 0.6500 | yes |
| ETTm1 | 144 | 5 | main | 0.1660 | 0.1129 | 0.2044 | 0.1122 | 0.3351 | 0.1734 | 0.1631 | 0.1474 | 0.1497 | 0.1879 | yes |
| ETTm1 | 180 | 4 | main | 0.2035 | 0.1369 | 0.2528 | 0.1542 | 0.4039 | 0.2150 | 0.1811 | 0.1606 | 0.1571 | 0.6000 | yes |
| ETTm1 | 240 | 3 | main | 0.2300 | 0.1872 | 0.2514 | 0.1872 | 0.4856 | 0.2482 | 0.2225 | 0.1669 | 0.1681 | -0.5000 | yes |
| ETTm1 | 360 | 2 | coarse_control | 0.3268 | 0.3268 | 0.3268 | nan | 0.6608 | 0.3664 | 0.3619 | 0.1860 | nan | nan | no |
| Weather | 120 | 6 | main | 0.0645 | 0.0255 | 0.1343 | 0.0183 | 0.2231 | 0.0725 | 0.0377 | 0.1397 | 0.1414 | -0.1464 | yes |
| Weather | 144 | 5 | main | 0.0686 | 0.0262 | 0.1344 | 0.0194 | 0.2582 | 0.0770 | 0.0350 | 0.1410 | 0.1432 | 0.0061 | yes |
| Weather | 180 | 4 | main | 0.0773 | 0.0278 | 0.1333 | 0.0214 | 0.3115 | 0.0864 | 0.0377 | 0.1391 | 0.1411 | -0.2000 | yes |
| Weather | 240 | 3 | main | 0.0839 | 0.0336 | 0.1090 | 0.0336 | 0.3901 | 0.0983 | 0.0437 | 0.1408 | 0.1483 | -0.5000 | yes |
| Weather | 360 | 2 | coarse_control | 0.1001 | 0.1001 | 0.1001 | nan | 0.5467 | 0.1193 | 0.1206 | 0.1514 | nan | nan | no |

## Gate Reading

- ETTh2: `4/4` main unit sizes pass the pre-registered robust-support gate.
- ETTm1: `4/4` main unit sizes pass the pre-registered robust-support gate.
- Weather: `4/4` main unit sizes pass the pre-registered robust-support gate.

[Decision] `partial_pass_large_unit_granularity_robust`.

[Strong Evidence] Shared-coefficient gradient pressure survives larger, benchmark-independent unit sizes on at least two datasets. The B9 signal is therefore not limited to canonical horizon boundaries or small units.

[Boundary] Diagnostic A only permits Diagnostic B. It does not prove that prefix-causal composition beats independent/no-transition capacity controls.

## Basis-Geometry Confound

[Fact] `1` main dataset/size settings have gradient-vs-basis pair Spearman `>=0.75`.

A high value means the gradient relation may be substantially inherited from A6 basis row-subspace geometry. Such settings support future-region heterogeneity but cannot by themselves prove that a compositional generator is necessary.

## Failure Attribution

- `hypothesis_false`: not decided unless the large-unit gate fails broadly;
- `basis_geometry_confounded`: explicitly measured by pairwise gradient/basis association;
- `granularity_specific`: applies if support concentrates on isolated unit sizes;
- `capacity_control_explains`: remains untested until Diagnostic B;
- `direction_level_rejection`: Diagnostic A cannot reject all future-unit architectures.

## Next

Only if the decision is `partial_pass_large_unit_granularity_robust`, design Diagnostic B with parameter-matched `shared / independent / no-transition / prefix-causal composed` unit states. No model candidate may enter Step 4-6 before that control gate.
