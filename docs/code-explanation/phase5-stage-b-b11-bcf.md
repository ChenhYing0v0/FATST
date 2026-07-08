# Phase5 StageB B11-BCF Code Explanation

本文档记录 `B11-BCF` 的最小本地实现。该实现对应
`docs/experiments/phase5-stage-b-emergent-subspace-aggregation.md` 中的 Step 7 local gate。

## Scope

入口文件：

- `baselines/timealign_official/models/TimeAlign.py`;
- `baselines/timealign_official/train_repo.py`;
- `scripts/check_phase5_stage_b_b11_bcf_local.py`;
- `scripts/remote/run_phase5_stage_b_b11_bcf_small_gate.sh`;
- `scripts/sync_phase5_stage_b_b11_bcf_small_gate_results.sh`;
- `scripts/analyze_phase5_stage_b_b11_bcf_small_gate.py`。

新增 readout modes：

| Mode | Role |
| --- | --- |
| `basis-conditioned-coefficient-field` | B11-BCF method candidate |
| `basis-conditioned-coefficient-field-no-basis` | no-basis capacity/control arm |
| `basis-conditioned-coefficient-field-shuffled-basis` | shuffled basis-order control |
| `basis-conditioned-coefficient-field-constant-slot` | constant row-mixture control |

所有 B11 modes 都属于 prefix-native learned-basis readout，因此 `train_repo.py` 会将 effective
`w_recon=0.0`、`w_align=0.0`，不启用 TimeAlign future reconstruction/alignment branch。

## Forward Tensor Path

A6-LBF-r256 的基本路径仍然保留：

1. history encoder 输出 `hidden: [B,C,R]`；
2. `learned_basis_coeff(hidden)` 得到 `coeff_base: [B,C,K]`；
3. `learned_temporal_basis[:H]: [H,K]` 与 `coeff_base` 投影得到 `output: [B,H,C]`。

B11-BCF 在第 2 与第 3 步之间加入 continuous coefficient field：

1. `basis_field_window_starts` 根据 `window_len=96`、`stride=48` 在 `pred_len=720` 上构造
   `M=14` 个 overlapping windows；
2. 对每个 window，取 `learned_temporal_basis[start:end]` 的均值，经过
   `basis_field_desc_norm` 和 `basis_field_desc_proj` 得到 `window_desc: [M,D]`；
3. 对每个 future row，`learned_temporal_basis[:H]` 经过同一 descriptor projector 得到
   `row_desc: [H,D]`；
4. `alpha = softmax(normalize(row_desc) @ normalize(window_desc)^T / tau)`，形状为 `[H,M]`；
5. `basis_field_state_proj(hidden)` 得到 `state: [B,C,D]`；
6. `field_state = state[:, :, None, :] + window_desc[None, None, :, :]`，形状为 `[B,C,M,D]`；
7. `basis_field_delta(gelu(field_state))` 得到 `delta: [B,C,M,K]`；
8. `coeff_slots = coeff_base[:, :, None, :] + sigmoid(gate) * tanh(delta)`，形状为 `[B,C,M,K]`；
9. 输出直接用 einsum 计算：

$$
\hat y_{b,t,c} = \sum_m \alpha_{t,m}\, basis_t^\top coeff\_slots_{b,c,m} + bias_t.
$$

这里没有 hard `stage_id`、没有 `horizon_id`，也没有独立 dense trajectory head。B11 的新增信息来自
`learned_temporal_basis` 自身的 continuous window descriptors。

## Controls

### no-basis

`basis-conditioned-coefficient-field-no-basis` 不读取 `learned_temporal_basis` 来生成 descriptors。
它使用同数量的 learned `basis_field_no_basis_rows: [720,D]` 和
`basis_field_no_basis_slots: [M,D]`。若该 control 与 B11-BCF 持平，则收益更可能来自多 slot head
capacity，而不是 basis geometry。

### shuffled-basis

`basis-conditioned-coefficient-field-shuffled-basis` 用 reversed basis order 构造 descriptors，但输出仍
投影到原始 `learned_temporal_basis[:H]`。它保留 basis row 的数值分布，但破坏 continuous future
order。

### constant-slot

`basis-conditioned-coefficient-field-constant-slot` 保留 basis descriptors 和 slot generation，但把
`alpha[t,m]` 替换为所有 rows 共享的平均 mixture。它测试 row-wise continuous coefficient field 是否真正必要。

## Function-Preserving Initialization

`basis_field_delta` 的 weight/bias 初始化为 0。因此初始时：

$$
coeff\_slots_{b,c,m} = coeff\_base_{b,c}.
$$

同时每个 row 的 `alpha[t,:]` 经 softmax 后和为 1，所以 B11 初始输出应与 A6 输出一致。代码中这是
function-preserving initialization path；论文叙事不应写成 residual correction。

## Configuration

`train_repo.py` 新增参数：

| Argument | Default | Meaning |
| --- | ---: | --- |
| `--basis-field-window-len` | `96` | basis descriptor window length |
| `--basis-field-stride` | `48` | basis descriptor stride |
| `--basis-field-rank` | `32` | descriptor/state hidden dimension |
| `--basis-field-tau` | `1.0` | row-window softmax temperature |
| `--basis-field-gate-init` | `-5.0` | initial scalar gate logit |

`model_diagnostics.json` 会记录 window count、rank、tau、gate sigmoid 和相关权重范数。

## Local Verification

已运行：

```bash
conda run -n r2026-fsa python -m py_compile \
  baselines/timealign_official/models/TimeAlign.py \
  baselines/timealign_official/train_repo.py \
  scripts/check_phase5_stage_b_b11_bcf_local.py
```

已运行：

```bash
conda run -n r2026-fsa python scripts/check_phase5_stage_b_b11_bcf_local.py
```

结果：

| Check | Result |
| --- | ---: |
| A6 fallback H96 max abs | `3.695488e-06` |
| B11 H96 vs H720 prefix max abs | `0.000000e+00` |
| B11/control backward | passed for all four B11 modes |

真实 ETTh2 one-batch CPU smoke 已运行：

```bash
conda run -n r2026-fsa python baselines/timealign_official/train_repo.py \
  --dataset-root /Users/river/PaperResearch/Project/datasets \
  --dataset ETTh2 \
  --mode unified \
  --seq-len 720 \
  --label-len 48 \
  --pred-len 720 \
  --target-horizons 96,192,336,720 \
  --readout-mode basis-conditioned-coefficient-field \
  --basis-rank 256 \
  --pred-loss-mode multi-prefix \
  --batch-size 2 \
  --epochs 1 \
  --patience 1 \
  --max-train-batches 1 \
  --max-eval-batches 1 \
  --num-workers 0 \
  --device cpu \
  --run-name smoke_b11_bcf_etth2 \
  --output-dir artifacts/smoke_phase5_stage_b_b11_bcf_local/b11_bcf_etth2
```

Artifact:

- `artifacts/smoke_phase5_stage_b_b11_bcf_local/b11_bcf_etth2/`.

Smoke result:

- one train batch / one val batch / one test batch completed；
- effective official args show `w_align=0.0`, `w_recon=0.0`；
- `basis_field_window_count=14`, `basis_field_gate_sigmoid=0.006693`.

## Remote Small Gate Scripts

`scripts/remote/run_phase5_stage_b_b11_bcf_small_gate.sh` 默认启动 required small gate：

- datasets: `Weather ETTm1 ETTh2`；
- arms: `a6_clean b11_bcf b11_no_basis b11_constant_slot`；
- GPUs: `0 1 2`；
- output root: `/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b11_bcf_small_gate`。

脚本沿用 dataset-major scheduling，让慢数据集优先铺到可用 GPU，避免所有 Weather 长任务堆在同一张卡。
`b11_shuffled_basis` 是 optional arm，可通过 `ARMS` 环境变量追加。

`scripts/sync_phase5_stage_b_b11_bcf_small_gate_results.sh` 会同步 remote metrics，并调用
`scripts/analyze_phase5_stage_b_b11_bcf_small_gate.py` 生成：

- `b11_bcf_small_gate_comparisons.csv`;
- `b11_bcf_small_gate_summary.csv`;
- `b11_bcf_small_gate_model_diagnostics.csv`;
- `b11_bcf_small_gate_report.md`。

Analyzer 的 gate rule：

1. `b11_bcf` 必须不劣于 `a6_clean`；
2. `b11_bcf` 必须优于 `b11_no_basis`；
3. `b11_bcf` 必须优于 `b11_constant_slot`；
4. 若 optional `b11_shuffled_basis` 存在，则额外报告 order-control 对比。

## Code-Theory Consistency

[Intended theory] B11-BCF should let A6 learned basis geometry organize sample-wise coefficient states through a
continuous, soft coefficient field.

[Code realization] The code derives row/window descriptors from `learned_temporal_basis`, uses them to produce
`alpha[t,m]` and slot-specific `coeff_slots`, then projects through the same original basis rows.

[Proxy boundary] The current minimal implementation uses hidden-level state projection rather than modifying encoder
memory attention. It tests whether basis-conditioned coefficient field is useful before adding a heavier memory-level
aggregation path.

[Falsification] If `no-basis` or `constant-slot` matches B11-BCF in remote small gate, the mechanism should be
classified as capacity/head effect rather than basis-conditioned architecture contribution.
