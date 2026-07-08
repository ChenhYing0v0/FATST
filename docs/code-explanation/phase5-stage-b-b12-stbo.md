# Phase5 StageB B12-STBO Implementation Explanation

对应文件：

- `baselines/timealign_official/models/TimeAlign.py`
- `baselines/timealign_official/train_repo.py`
- `scripts/check_phase5_stage_b_b12_stbo_local.py`
- `scripts/remote/run_phase5_stage_b_b12_stbo_small_gate.sh`
- `scripts/sync_phase5_stage_b_b12_stbo_small_gate_results.sh`
- `scripts/analyze_phase5_stage_b_b12_stbo_small_gate.py`

## Purpose

`B12-STBO` 指 `Subspace-Tiled Basis Operator`。这次实现不是基于 A6 输出的 residual correction，也不是
`coeff + delta`。它直接新增一类 primary prediction head：

```text
full-720 step basis projection
  -> tile-local basis projection
```

该实现用于 small trainable architecture gate，测试 native STBO 是否能从训练中主动形成 stage/tile-local
prediction operator。它不能因为 offline diagnostic 弱而被直接否定，也不能在未通过 controls 前写成 paper-core
method。

## Readout Modes

新增四个 `readout_mode`：

| Mode | Role |
| --- | --- |
| `subspace-tiled-basis-operator-shared` | 所有 tiles 共享一套 learned local basis |
| `subspace-tiled-basis-operator-bank` | 多个 learned local basis banks，tile 通过 learned soft mixture 选择 |
| `subspace-tiled-basis-operator-dct` | fixed local DCT basis control |
| `subspace-tiled-basis-operator-independent` | 每个 tile 独立 learned local basis；capacity / upper-bound control |

这四个模式都属于 prefix-native readout。`train_repo.py` 会自动关闭 TimeAlign future recon/align branch，
即 `w_recon=0.0`、`w_align=0.0`。

## Tensor Flow

设：

- `hidden: [B,C,R]`，来自 TimeAlign history encoder；
- `pred_len=720`;
- `tile_len=L=48`;
- `tile_count=M=15`;
- `stbo_rank=Rb=16`。

### 1. Generate Tile Coefficients

STBO 不使用 A6 的：

```text
learned_basis_coeff(hidden) -> coeff: [B,C,K]
learned_temporal_basis: [720,K]
```

而是直接生成 tile-wise coefficients：

```text
stbo_coeff(hidden) -> [B,C,M*Rb]
reshape -> tile_coeff: [B,C,M,Rb]
```

这意味着 sample-wise information 不是一个服务整段 720 future 的 global coeff，而是每个 tile 有自己的
local coefficient。

### 2. Build Local Basis Tiles

不同 mode 生成：

#### shared

```text
stbo_shared_basis: [L,Rb]
basis_tiles = repeat(stbo_shared_basis, M) -> [M,L,Rb]
```

#### bank

```text
stbo_basis_bank: [Q,L,Rb]
stbo_tile_bank_logits: [M,Q]
pi = softmax(stbo_tile_bank_logits, dim=-1)
basis_tiles[m] = sum_q pi[m,q] * stbo_basis_bank[q]
```

#### dct

```text
stbo_dct_basis: [L,Rb]
basis_tiles = repeat(stbo_dct_basis, M) -> [M,L,Rb]
```

`stbo_dct_basis` 是 buffer，不训练，用于测试 learned STBO 是否只是 generic local smoothness。

#### independent

```text
stbo_tile_basis: [M,L,Rb]
```

这是 upper-bound/capacity control，不应直接作为首选论文方法。

### 3. Tile Projection

对需要的 prefix horizon `H`：

```text
needed_tiles = ceil(H / L)
tile_output = einsum("mlr,bcmr->bcml", basis_tiles[:needed_tiles], tile_coeff[:needed_tiles])
reshape -> [B,C,needed_tiles*L]
trim -> [B,C,H]
permute -> [B,H,C]
```

再加：

```text
stbo_temporal_bias[:H]
```

输出之后走原有 `Normalize(..., "denorm")`。

## Prefix Consistency

STBO 的 prefix consistency 来自相同的 tile coefficients 和 basis prefix：

```text
model(x, target_prefix=96)
==
model(x, target_prefix=720)[:, :96, :]
```

本地检查脚本会对四个 STBO modes 做该验证。

## Training Interface

新增 CLI 参数：

| Argument | Default | Meaning |
| --- | ---: | --- |
| `--stbo-tile-len` | `48` | local tile length |
| `--stbo-rank` | `16` | local basis rank |
| `--stbo-bank-count` | `4` | basis bank count for bank mode |
| `--stbo-basis-init-std` | `stbo_rank ** -0.5` | learned local basis init std |

`--stbo-tile-len` 必须整除 `pred_len`，`--stbo-rank <= stbo_tile_len`。

## Local Verification

`scripts/check_phase5_stage_b_b12_stbo_local.py` 执行三类检查：

1. synthetic prefix consistency：`H=96` 直接输出与 `H=720` prefix 一致；
2. synthetic backward：四个 STBO modes 都能反向传播到 `stbo_coeff`；learned modes 还能传播到 local basis；
3. ETTh2 one-batch CPU smoke：使用真实 data loader 做一批 forward/backward。

## Code-Theory Consistency

Intended theory:

- B12 应测试 native STBO architecture 能否主动学出 stage/tile-local operator；
- 它不是从 A6 full-basis 解中寻找已经存在的模式；
- controls 必须区分 learned STBO、fixed local DCT、independent tile capacity。

Code realization:

- STBO readouts 不实例化 A6 `learned_temporal_basis` 主路径；
- `hidden -> tile_coeff` 是主预测路径；
- local basis tiles 直接参与 output projection；
- DCT 和 independent modes 是 mandatory controls。

Proxy limitation:

- 本地 smoke 只证明 shape、prefix consistency 和 gradient path；
- 不证明性能；
- 不证明 learned bank 的 mixture 有语义。

Falsification:

- 若 learned shared/bank 不优于 fixed local DCT，则不能 claim learned subspace operator；
- 若 independent tile 显著更好而 shared/bank 不行，则方法可能退化为 segmented Direct head；
- 若 STBO 相对 A6 没有稳定收益，则不能作为 StageB paper-core method。

## Remote Small Gate Scripts

### Runner

`scripts/remote/run_phase5_stage_b_b12_stbo_small_gate.sh` 固定 small gate 的 required arms：

| Arm | `readout_mode` | Role |
| --- | --- | --- |
| `a6_clean` | `learned-basis-forecast-operator` | clean A6 anchor |
| `stbo_shared` | `subspace-tiled-basis-operator-shared` | shared local-basis method candidate |
| `stbo_bank4` | `subspace-tiled-basis-operator-bank` | learned basis-bank method candidate |
| `stbo_dct` | `subspace-tiled-basis-operator-dct` | fixed local basis control |
| `stbo_independent` | `subspace-tiled-basis-operator-independent` | independent-tile capacity control |

默认远程矩阵：

- datasets: `Weather ETTm1 ETTh2`;
- horizons: `96,192,336,720`;
- seed: `2021`;
- epochs/patience: `10/3`;
- checkpoint policy: `official-last`;
- STBO config: `tile_len=48`, `rank=16`, `bank_count=4`;
- loss: `multi-prefix`;
- scheduling: dataset-major order on `GPU_IDS`, so slower datasets are launched first across available GPUs.

Runner 的输出目录结构与已有 Phase5 gates 一致：

```text
${OUTPUT_ROOT}/official-last/
  TimeAlignOfficialUnified720_${arm}_official-last/
    ${dataset}/mixed_h96_h192_h336_h720/seed2021/
```

若目标 run 已存在 `metrics_by_target_horizon.csv` 和 `checkpoint.pt`，runner 会跳过该 run，便于 remote resume。

### Sync Wrapper

`scripts/sync_phase5_stage_b_b12_stbo_small_gate_results.sh` 从远程
`${REMOTE_OUTPUT_ROOT}/official-last/` 同步 artifact 到：

```text
analysis/phase5_stage_b_b12_stbo_small_gate_${date}/raw/official-last/
```

默认排除：

- `checkpoint.pt`;
- `predictions_test.npz`。

同步后自动调用 analyzer；如果只需拉取原始结果，可设置 `SKIP_ANALYSIS=1`。

### Analyzer

`scripts/analyze_phase5_stage_b_b12_stbo_small_gate.py` 输出：

- `b12_stbo_small_gate_comparisons.csv`;
- `b12_stbo_small_gate_summary.csv`;
- `b12_stbo_small_gate_model_diagnostics.csv`;
- `b12_stbo_small_gate_report.md`。

核心 comparisons：

- `stbo_shared` / `stbo_bank4` / `stbo_dct` / `stbo_independent` vs `a6_clean`;
- learned STBO vs `stbo_dct`;
- learned STBO vs `stbo_independent`;
- `stbo_bank4` vs `stbo_shared`。

Report gate 不允许只凭超过 A6 宣称方法成立。必须同时读：

1. learned shared/bank 是否超过 fixed DCT；
2. learned shared/bank 是否不是只被 independent-tile capacity 压倒；
3. 相对 A6 是否没有系统性退化。

Failure attribution 规则：

- DCT 持平或更好：`generic_local_basis_control_explains`；
- 只有 independent-tile 明显更好：`independent_tile_capacity_explains`；
- learned STBO 不稳或退化：仅拒绝当前 tested implementation，不拒绝全部 native multi-horizon operator 方向。
