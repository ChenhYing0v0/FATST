# Phase5 TimeAlign Carrier Gate 代码说明

## 研究定位

Phase5 不是直接提出 HSS 方法，而是先验证 TimeAlign 是否能成为新的 HSS carrier。
本阶段回答两个问题：

1. fixed-horizon TimeAlign-style carrier 在当前 repo 数据、split、metric 下是否可用；
2. unified-720 TimeAlign 是否相对 fixed-horizon TimeAlign 出现可解释的 multi-horizon gap。

只有当 fixed carrier 可用且 unified gap 存在，才进入 TimeAlign-based HSS 设计。

## Forward 数据流

输入 history：

$$
x \in \mathbb{R}^{B \times L \times C}.
$$

`TimeAlignCarrier` 先做 per-sample channel-wise normalization，然后把 history 分为
`patch_num` 个 patch：

$$
X_p \in \mathbb{R}^{B \times C \times N \times d}.
$$

history branch 经过 `e_layers` 个 residual MLP blocks，得到预测 state：

$$
H_x \in \mathbb{R}^{B \times C \times N \times d}.
$$

readout head 将 `H_x` flatten 到 patch dimension，并输出：

$$
\hat{Y} \in \mathbb{R}^{B \times T \times C}.
$$

当 training batch 提供 ground-truth future：

$$
y \in \mathbb{R}^{B \times T \times C},
$$

future branch 以同构 patch embedding 编码 `y`，经过 autoencoder blocks 得到：

$$
H_y \in \mathbb{R}^{B \times C \times N \times d}.
$$

student projection 把 `H_x` 映射到 alignment space，并与 stop-gradient `H_y` 对齐。
future branch 同时通过 reconstruction head 输出 `\hat{Y}_{recon}`。

## Loss

训练 loss 为：

$$
\mathcal{L}
=
\mathcal{L}_{pred}^{L1}
+ w_{recon}\mathcal{L}_{recon}^{L1}
+ w_{align}\mathcal{L}_{align}.
$$

`GlocalAlignment` 包含：

- local alignment：patch-level normalized feature product；
- global alignment：patch relation Gram matrix difference；
- dynamic balancing：按 component detached loss 做比例平衡，避免单个 component 主导。

## 与 Phase4 Future Anchor 的差异

Phase4-FSA 把 future teacher 接到 `target_states` 上，且只使用很小的 auxiliary weight：

- `future_align_weight=0.01`;
- `future_recon_weight=0.001`;
- `future_relation_weight=0`;
- prediction carrier 仍是 target-set decoder。

Phase5 TimeAlign carrier 则把 predict/reconstruct/alignment 作为同一 carrier 的训练范式：

- future reconstruction branch 与 prediction branch patch 结构同构；
- reconstruction loss 使用 `w_recon=1.0`；
- alignment 同时包含 local/global relation；
- inference path 仍只依赖 history，不读取 future。

因此 Phase5 不是继续修补 Phase4 anchor，而是在 Step 2/3 重新验证一个更接近原论文机制的
carrier。

## Runner

`scripts/remote/run_phase5_timealign_carrier_gate.sh` 默认运行：

- datasets: `Weather ETTh2`;
- fixed-horizon runs: `h96/h192/h336/h720`;
- unified run: `pred_len=720`，`target_horizons=96,192,336,720`;
- output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_carrier_gate`;
- GPUs: `1 2`;
- epochs: `10`;
- patience: `3`.

Weather 使用更大的 model setting：

- `d_model=128`;
- `d_ff=256`;
- `w_align=0.1`;
- `local_margin=0.5`;
- `layer_norm=off`.

ETTh2 使用轻量 setting：

- `d_model=32`;
- `d_ff=32`;
- `w_align=0.1`;
- `local_margin=0.0`;
- `layer_norm=on`.

## Analysis

`scripts/analyze_phase5_timealign_carrier_gate.py` 生成：

- `phase5_timealign_fixed_metrics.csv`;
- `phase5_timealign_unified_metrics.csv`;
- `phase5_timealign_unified_gap.csv`;
- `phase5_timealign_unified_gap_summary.csv`;
- `phase5_timealign_training_summary.csv`;
- `phase5_timealign_carrier_gate_report.md`.

核心比较是：

$$
\Delta_{unified}
=
\frac{MSE_{unified720@h} - MSE_{fixed@h}}{MSE_{fixed@h}}.
$$

## Code-Theory Consistency

[Intended theory] TimeAlign 的 fixed-horizon distribution alignment 可能是一个比当前
target-set future anchor 更强的 carrier；HSS 的研究问题应落在 unified multi-horizon 下
多个 future distributions 的 supervision conflict。

[Code realization] 代码实现 fixed-horizon 与 unified-720 两种训练协议，并使用相同
TimeAlign carrier。HSS 还没有加入，因此 gap 可以归因于 carrier/protocol，而不是 schedule。

[Proxy] 该实现是 source-informed local baseline，不追求与官方代码逐行等价。它保留核心机制：
同构 future reconstruction branch、stop-gradient future state、local/global alignment、
reconstruction loss。

[Falsification] 若 fixed-horizon carrier 本身弱，不能进入 TimeAlign-HSS；若 unified-720 不
产生明确 gap，则 HSS 缺少问题支点，不应强行写成 schedule 贡献。
