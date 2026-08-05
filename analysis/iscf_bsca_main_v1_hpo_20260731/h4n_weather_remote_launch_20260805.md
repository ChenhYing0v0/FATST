# H4N Weather-only Remote Launch

日期：2026-08-05

## 1. Exact freeze and preflight

- experiment commit：`ba17fc97e85cdd362611a5eaa58b088be85add54`；
- remote project：`/home/yingch/projects/FATST`；
- config hash：`cd20062d982fe6569758f79bbf57ca1c2a59d9fe8fe95570da4de1017e8c7164`；
- search-space hash：`0961d3197584c3f991b7d14e38279fe81895d268753fec3348e9adfafe1a226e`；
- output root：`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h4n`；
- remote worktree原有三份unrelated dirty CSV保持不变；
- local与remote checker均确认40/40 profiles、189个历史profiles audited、零重复、
  `test_jobs=0`；
- launch前GPU0/1/2均为18 MiB used、约24.1 GiB free、0% utilization；
- quota=`169G / 200G soft / 220G hard`，无需删除formal H4L/H4M artifacts。

## 2. Resource smoke

2026-08-05 10:44:54--10:46:45完成40/40 profiles的one-epoch/two-batch
resource smoke：

- checkpoints=40/40；
- `metrics_by_target_horizon.csv`=40/40；
- logs=40/40；
- test=0/40；
- 未发现Traceback、RuntimeError、OOM、NaN或Inf。

Smoke只证明execution/memory/data path健康，不构成validation或test performance evidence。

## 3. Full training launch

正式three-GPU dynamic queue于`2026-08-05T10:47:16+08:00`启动：

- supervisor PID=`1397808`；
- jobs=`40 Weather × seed2021`；
- budget=`120 epochs / patience 24`；
- training-time official test=`0/40`；
- initial jobs：
  - GPU0：`Weather__h4n_seq512_p16_lr2e5_d128_ff256_r128`；
  - GPU1：`Weather__h4n_seq512_p16_lr2e5_d96_ff192_r128`；
  - GPU2：`Weather__h4n_seq704_p22_lr2e5`。

启动后前三个jobs均进入epoch2。Observed GPU memory为GPU0/1/2约
`1687/1640/1602 MiB`，utilization约`77/73/64%`，保留约22 GiB安全余量。

预计45--75 GPU-hours、15--26 wall-hours。下一次检查使用长间隔，并报告
complete jobs、active trial、epoch位置及根据实际吞吐更新的ETA；40/40 training
artifacts与immutable checkpoint manifest通过前不访问official test。

Decision=`H4N_Weather_40_train_validation_jobs_active_test_zero`。
