# Phase4 Future-State Anchored HSS 代码说明

## 研究定位

Phase4-FSA-F1 不是新的 forecasting head，也不是继续调 `aux_weight/top_ratio`。
它验证一个更上层的问题：

> HSS 的 supervision pressure 是否需要先落在 future-structured `target_states`
> 上，才可能稳定提升 unified multi-horizon forecasting？

因此本阶段只复用仓库已有的 future-state alignment branch，不改变 inference-time
prediction path。实验比较：

- `F1-C0`: `single_720_prefix_risk`，future alignment off；
- `F1-C1`: `r3_prefix_risk`，future alignment off；
- `F1-A0`: `single_720_prefix_risk + future-state anchor`；
- `F1-A1`: `r3_prefix_risk + future-state anchor`；
- `F1-W0`: `full_time_mse + future-state anchor`，只作为 weak control。

## Forward 数据流

预测主路径仍是 target-set decoder。给定 history：

$$
x \in \mathbb{R}^{B \times L \times C},
$$

encoder 产生 history patch states：

$$
Z \in \mathbb{R}^{(BC) \times N \times d}.
$$

target segment queries 与 $Z$ cross-attend 后得到：

$$
U_T \in \mathbb{R}^{(BC) \times J \times d}.
$$

`U_T` 进入 condition/readout path，输出预测：

$$
\hat{Y}_{1:H}=f_\theta(x,H).
$$

这个路径不读取 ground-truth future。evaluation/test 只传入 history 与 requested horizon。

当 future-state anchor 开启时，training batch 额外把 ground-truth future
`future_y` 输入 teacher branch。teacher branch 将 normalized future sequence 按 segment
切分并编码为：

$$
S_T^Y \in \mathbb{R}^{(BC) \times J \times d_s}.
$$

student branch 从 inference-time `U_T` 投影得到：

$$
S_T^X=P_\theta(U_T).
$$

alignment loss 只约束 `S_T^X` 接近 stop-gradient teacher state `S_T^Y`。teacher branch
还用 reconstruction head 重建 normalized future segments，避免 teacher latent 成为任意
不可解释 anchor。

## Objective

主预测 loss 由 supervision strategy 决定：

- `single_720_prefix_risk`: h720-only training，`prefix_risk` step pressure；
- `r3_prefix_risk`: mixed-horizon exposure + `prefix_risk` step pressure；
- `full_time_mse`: h720-only full future MSE。

future-state anchor 额外加入：

$$
\mathcal{L}
=
\mathcal{L}_{pred}
+
\lambda_{local}\mathcal{L}_{local}
+
\lambda_{recon}\mathcal{L}_{recon}.
$$

F1 默认不启用 relation alignment：

$$
\lambda_{rel}=0.
$$

默认配置：

- `future_teacher_layers=1`;
- `future_align_weight=0.01`;
- `future_relation_weight=0.0`;
- `future_recon_weight=0.001`;
- `future_recon_normalization=target_energy`;
- `future_align_weighting=reconstruction_confidence`;
- `future_confidence_floor=0.05`;
- `learning_rate=5e-5`。

这里的权重刻意保守：F1 要测试 representation substrate 是否有价值，而不是让 auxiliary
loss 覆盖 forecasting objective。

## Runner

`scripts/remote/run_phase4_future_state_anchor_gate.sh` 执行 F1 matrix。默认：

- remote output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase4_future_state_anchor_gate`;
- datasets: `Weather ETTh2`;
- arms: `F1-C0 F1-C1 F1-A0 F1-A1 F1-W0`;
- GPUs: `1 2`;
- job order: `dataset_major`;
- seed: `2021`;
- epochs: `100`。

`dataset_major` 是 workload-aware 默认值。它按 dataset 外层、arm 内层排队，因此会先把
`Weather` 的多个 arms 分散到 GPU 1/2 上，再进入更快的 `ETTh2` jobs。这样避免
`Weather + ETTh2` 成对运行时 ETTh2 很快结束、对应 GPU 长时间空等。若需要复现旧的
逐 arm 配对顺序，可显式设置 `JOB_ORDER=arm_major`。

每个 arm 使用唯一 `run_name`，避免相同 strategy 的 on/off 配置覆盖：

| Arm | `run_name` |
| --- | --- |
| `F1-C0` | `PatchEncoderFSAF1SinglePrefixBase` |
| `F1-C1` | `PatchEncoderFSAF1R3Base` |
| `F1-A0` | `PatchEncoderFSAF1SinglePrefixFutureAnchor` |
| `F1-A1` | `PatchEncoderFSAF1R3FutureAnchor` |
| `F1-W0` | `PatchEncoderFSAF1FullTimeFutureAnchor` |

`scripts/remote/check_phase4_future_state_anchor_progress.sh` 汇总 log tail 和已完成的
`metrics_by_target_horizon.csv`。

## Analysis

`scripts/analyze_phase4_future_state_anchor_gate.py` 从 raw artifacts 生成：

- `phase4_fsa_f1_main_metrics.csv`;
- `phase4_fsa_f1_main_deltas.csv`;
- `phase4_fsa_f1_main_summary.csv`;
- `phase4_fsa_f1_dataset_summary.csv`;
- `phase4_fsa_f1_segment_deltas.csv`;
- `phase4_fsa_f1_segment_region_summary.csv`;
- `phase4_fsa_f1_training_summary.csv`;
- `phase4_fsa_f1_future_alignment.csv`;
- `phase4_fsa_f1_future_alignment_summary.csv`;
- `phase4_fsa_f1_checkpoint_diagnostics.csv`;
- `phase4_fsa_f1_config_summary.csv`;
- `phase4_future_state_anchor_gate_report.md`。

核心比较包括：

- `F1-A0` vs `F1-C0`: future anchor 是否改善 clean h720-only prefix-risk base；
- `F1-A1` vs `F1-C1`: future anchor 是否改善 R.3 compound strong reference；
- `F1-A0` vs `F1-C1`: clean anchored HSS 是否接近 R.3；
- `F1-W0` vs `F1-C0/C1`: full-time anchored base 是否仍弱。

segment analysis 聚合 `early_1_96`、`middle_97_336`、`late_337_720`，并特别检查
Weather h720 `337-720`。

## Gate

F1 通过不要求所有 arm 同时赢。它判断 substrate 是否值得进入 FSA-F2：

1. `F1-A1` 相对 R.3 mean MSE 不劣于 `+0.3%`，并改善 Weather h720 或 Weather long/late；
2. 或 `F1-A0` 相对 `single_720_prefix_risk` 至少 `5/8` main MSE wins；
3. `prediction_leakage_max_abs <= 1e-7`；
4. reconstruction confidence 不应 collapse 到 floor；
5. 若只有 h720/long oracle checkpoint 有收益，则先研究 validation metric，而不是继续改模型。

## Code-Theory Consistency

[Intended theory] HSS 失败可能不是 supervision signal 不存在，而是当前 `target_states`
没有 future-aware geometry。future-state anchor 应当让 HSS pressure 有更稳定的更新落点。

[Code realization] F1 不改预测主路径，只在 training loss 中加入 teacher/student alignment
和 teacher reconstruction。`future_y` 不参与 prediction tensor 计算，test-time leakage 由
`future_alignment_stats.csv` 和 `future_leakage_audit.json` 检查。

[Proxy] learned future teacher 仍是 proxy，不是物理真实 future state。若 alignment 指标改善
但 MSE/MAE 不改善，说明该 anchor 与 forecasting objective 不一致。

[Falsification] 若 `F1-A0/A1` 均输给各自 base，且 leakage/confidence diagnostics 正常，
则当前 future-state anchor 不能作为 HSS substrate；下一步应回 Step 2/3，重新定义
representation problem 或停止 Phase4 local HSS stacking。
