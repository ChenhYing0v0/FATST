# StageC SC0-R1 Unified Stopping Policy Offline Gate

## Metric Definitions

- `stop_epoch`: 从原SC0 validation trajectory按连续未改善epoch数模拟得到的停止点。
- `selected_epoch`: stop前最低`val_mean_mse`对应epoch，即restore-best checkpoint。
- `retains_full_trajectory_best`: stop前best是否等于完整20-epoch trajectory的best。
- `epochs_saved`: `20 - stop_epoch`；只表示计算节省，不参与profile选择。

## Candidate Summary

| patience | retained runs | early-stopped runs | total epochs saved | mean saved/run |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 7/9 | 9/9 | 136 | 15.11 |
| 5 | 9/9 | 9/9 | 105 | 11.67 |
| 7 | 9/9 | 9/9 | 87 | 9.67 |

## Decision

[Strong Evidence] 最小且保留9/9原SC0 best checkpoint的候选是`patience=5`。SC0-R1预注册`max_epochs=20`, `min_delta=0`, `restore_best=true`。

该离线结果只验证stopping rule不会在已有trajectory上截断已知best；它不证明新seed也会稳定，因此SC0-R1仍须运行三臂全部三个seeds。
