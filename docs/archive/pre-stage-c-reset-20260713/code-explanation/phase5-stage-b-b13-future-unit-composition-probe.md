# Phase5 StageB B13-FUCO-B Future-Unit Composition Probe

对应文件：

- `scripts/analyze_phase5_stage_b_b13_future_unit_composition_probe.py`

## Purpose

该 analyzer 执行 B13-FUCO Step 2/3 Diagnostic B。Diagnostic A 已证明 larger future units 对 A6 shared
coefficient 的 gradient pressure 跨 granularity 和 dataset 稳定；Diagnostic B 进一步测试：

> prefix-causal latent unit composition 是否比 exact parameter-matched no-transition unit generation 更有预测价值？

脚本固定 clean A6 encoder，只训练轻量 diagnostic probes。`--memory-source coeff` 复现 B1 的
post-coefficient intervention；`--memory-source hidden` 执行 B2 的 pre-coefficient intervention repair。
结果不是 end-to-end forecasting performance，也不能直接成为论文主结果。

## Frozen Artifact Path

默认 checkpoint：

```text
analysis/phase5_stage_b_b8_ocd_coefficient_oracle_20260707/raw/
  TimeAlignOfficialUnified720_A6LBF_r256_main_official-last/
    {dataset}/mixed_h96_h192_h336_h720/seed2021/checkpoint.pt
```

datasets：`ETTh2`, `ETTm1`, `Weather`。

## Probe Data Construction

对 train/val/test split 执行 deterministic A6 eval forward：

```text
batch_x: [B,720,C]
  -> Normalize + PatchEmbed + A6 encoder
hidden: [B,C,R]
  -> memory_source=hidden: memory [B,C,R]
  -> learned_basis_coeff
  -> memory_source=coeff: memory [B,C,256]
```

target 使用当前 history sample 的 A6 normalization statistics：

```text
target_norm = (target - history_mean) / history_std  # [B,720,C]
```

然后按 sample/channel flatten：

```text
memory_rows: [N*C,memory_dim]
target_rows: [N*C,720]
```

默认最多提取 `8192/2048/2048` train/val/test rows；B2 remote runner 显式使用
`4096/1024/1024`，控制高维 hidden memory 的 extraction 与 probe training 成本。`memory_rows` 再按
train feature mean/std standardize，val/test 只应用 train statistics。

## Probe Architecture

共同 modules：

```text
InputProject: memory_dim -> D
CoordinateMLP: 1 -> D -> D
Transition: GRUCell(D,D)
SharedDecoder: D -> U
```

默认 `D=64`，unit sizes 为 `180/240`。unit coordinate 是 unit center 除以 `720` 的连续标量，不使用
benchmark horizon id 或 learned stage table。

### `parallel_no_transition`

```text
base = tanh(InputProject(memory))
coord_m = CoordinateMLP(center_m / 720)
state_m = GRUCell(base + coord_m, base)
segment_m = SharedDecoder(state_m)
```

所有 unit states 都独立读取相同 base state；该 arm 同时是 independent-unit/no-transition control。

### `prefix_causal_composed`

```text
base = tanh(InputProject(memory))
state_-1 = base
state_m = GRUCell(base + coord_m, state_{m-1})
segment_m = SharedDecoder(state_m)
```

两个 arms 的 module instances、parameter shapes 和 trainable parameter count 完全一致；唯一差异是
GRUCell hidden state 来自 `base` 还是 previous latent unit。

预测值不会反馈给下一个 unit，因此不是 autoregressive value rollout。

## Prefix Consistency

`FutureUnitProbe.forward(memory, max_units=k)` 只执行前 `k` 个 units。由于 composed arm 的第 `m` 个
state 只读取 earlier states，理论上：

```text
probe(memory, max_units=k)
==
probe(memory, max_units=M)[:, :k*U]
```

每个 run 在 test rows 上报告 `prefix_max_abs`。

## Optimization Protocol

- seeds: `2021/2022/2023`；
- epochs: `20`；
- optimizer: AdamW；
- learning rate: `1e-3`；
- weight decay: `1e-4`；
- batch size: `256`；
- loss: normalized full-trajectory MSE，等价于 equal-size unit losses 的平均；
- checkpoint: best validation state，仅用于 diagnostic optimization fairness。

best-val 不属于当前 paper protocol，也不能被写成最终方法收益。这里使用它是为了避免 probe optimization
差异掩盖 composition/no-transition 的机制比较。

## Output Artifacts

默认目录：

```text
analysis/phase5_stage_b_b13_future_unit_composition_20260710/
```

| File | Meaning |
| --- | --- |
| `b13_future_unit_probe_runs.csv` | 每个 dataset/unit size/arm/seed 的 train/val/test metrics、parameter count 与 prefix error |
| `b13_future_unit_probe_comparisons.csv` | same-seed composed vs parallel paired comparison 和 per-unit relative MSE |
| `b13_future_unit_probe_summary.csv` | dataset/unit-size multi-seed aggregate 与 composition-support gate |
| `b13_future_unit_composition_report.md` | reader-facing Diagnostic B decision |

Analyzer 支持 `--report-only`：当 probe runs 已完成、只需修正 gate/report logic 时，直接从现有 runs 与
summary CSV 重建 report，避免重复训练。

## Remote B2 Runner

`scripts/remote/run_phase5_stage_b_b13_hidden_memory_probe.sh` 固定以下 repair boundary：

- `--memory-source hidden`；
- unit sizes `180/240`；
- seeds `2021/2022/2023`；
- state dimension `64`、epochs `20`；
- train/val/test row caps `4096/1024/1024`；
- `CUDA_VISIBLE_DEVICES` 只暴露预检后选择的单个 GPU。

runner 记录 git commit、GPU inventory、有效 row caps、checkpoint/output roots。结果由
`scripts/sync_phase5_stage_b_b13_hidden_memory_probe_results.sh` 从 repo-external remote output root 同步到
local `analysis/`。这两个 wrappers 只负责可复现的启动与 artifact 回收，不改变 analyzer gate。

## Gate

单个 dataset/unit-size setting 为 composition support，当：

```text
composed mean relative test MSE <= -0.5%
and
composed wins >= 2/3 seeds
```

整体通过需要：

- `6` settings 中至少 `4` 个 composition support；
- 每个 dataset 至少一个 size 的 mean degradation 不超过 `+0.25%`；
- 没有 non-finite loss；
- 不超过四分之一 runs 同时出现 `test/val ratio >3` 且绝对 MSE gap `>1`。

`target_norm` 使用 history-window std，future distribution shift 可能使绝对 normalized MSE 大于 `10`；
因此绝对阈值不能单独作为 numeric pathology。有效性由 finite check 和 val/test mismatch 联合判断。

通过标签：

```text
partial_pass_prefix_causal_composition
```

失败标签：

```text
no_transition_control_explains
```

病态标签：

```text
diagnostic_invalid_for_direction_rejection
```

## Code-Theory Consistency

Intended theory：

- future units 不只是不同 coordinate-conditioned independent heads；
- earlier latent future state 应给 later units 提供 history memory 之外的 compositional context；
- requested horizon 可以通过停止 unit composition 实现，而不是 clipping full-horizon latent state。

Code realization：

- exact same GRUCell/decoder/coordinate path；
- composed arm 只改变 recurrent hidden source；
- no predicted value feedback；
- prefix-limited forward path；
- paired seeds 与 exact parameter-count check。

Proxy limitations：

- B1 的 frozen A6 coefficient 可能已经丢失 composition 所需信息；B2 的 hidden repair 只检查这个
  intervention-point confound；
- GRUCell/shared decoder 可能不是最终正确 operator；
- best-val diagnostic 与 paper official-last protocol 不同；
- row subsampling 不是完整 dataset training；
- probe gain 不是 end-to-end forecasting gain。

Failure attribution：

- composed 输给 no-transition，可以阻断当前 latent-transition claim；
- 但不能单独证明所有 future-unit generation 都无效，因为 intervention point 与 probe design 仍可能错误；
- numeric/val-test pathology 出现时，只能修复 probe，不能做方向级拒绝。

## Returned B2 Verification

Hidden-memory B2 在 remote commit `013dd35` 上完成 `36` runs：

- `18/18` paired parameter deltas 为 `0`；
- max prefix error 为 `0`；
- all reported losses finite；
- `0/36` severe val-test mismatch；
- setting support 为 `4/6`，但 ETTh2-U180/U240 均平均退化约 `+5%`。

Analyzer 根据 `memory_source` 将 coefficient 与 hidden reports 分别标记为 `B13-FUCO-B1` 与
`B13-FUCO-B2`。由于 hidden width 为 ETTh2 `1536`、ETTm1 `256`、Weather `6144`，report 按
`dataset/unit_size` 显示 parameter set；只要求同一 paired comparison exact match，不要求跨 dataset 相等。

Per-unit audit 进一步发现，多个正向 settings 的最大收益出现在 unit 0，而 unit 0 没有 previous-unit
information；ETTm1-U240 最后一个 unit 还平均退化 `+7.50%`。因此当前 code 确实检验并否定了
GRU-based progressive composition claim，而不是仅仅因为 aggregate gate 少一个 dataset 通过。
