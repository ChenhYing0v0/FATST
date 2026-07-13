# StageC PMFO-RCT Step 7A Code Explanation

## 1. Scope And Status

本版本在frozen `A6-LBF-natural-baseline` Encoder后增加四个StageC readout：

- `pmfo-rct`：shared parent-to-child state transition + conservative synthesis；
- `pmfo-rct-no-transition`：history直接产生各scale coefficient group；
- `pmfo-rct-no-conservation`：保留相同tree state path，但允许detail改变parent projection；
- `dense-mlp-matched`：与PMFO decoder近似等参数的nonlinear dense control。

本轮不改Encoder、不改RevIN、不实现MIPR、不训练forecast model。`step7a_pass`只代表实现与algebra
满足预注册contract，不是effectiveness证据。

## 2. Forward Tensor Flow

### 2.1 Frozen history carrier

`TimeAlign.Model.forward`保留A6路径：

```text
x: [B,720,C]
-> RevIN norm
-> patch_emb_x + 2-layer token MLP Encoder
-> memory M: [B,C,P,D]
-> flatten(P,D)
-> hidden z: [B,C,768]
```

Weather/ETTh2使用`P=12,D=64`，ETTm1使用`P=24,D=32`，所以三个profile的readout input均为
`z: [B,C,768]`。`TimeAlign.py`在readout dispatch处把`z`交给`pmfo_readout`，所得
`[B,H,C]`继续通过原A6 RevIN denormalization。

### 2.2 `pmfo-rct` state path

`PMFORCTReadout.tree_coefficients`执行：

```text
z: [B,C,768]
-> seed + GELU
-> u0: [B,C,8,32]
-> coarse_head
-> a0: [B,C,8]
```

四层radix为`(3,3,2,5)`。level $l$的active parent states为`[B,C,N_l,32]`：

```text
split_l(parent state)
-> child states: [B,C,N_l,r_l,32]
concat(parent, flatten(children))
-> detail_head_l
-> d_l: [B,C,N_l,r_l-1]
```

child states reshape为`[B,C,N_l*r_l,32]`后进入下一层。`split_l`与`detail_head_l`在同层所有
future nodes间共享参数；主候选没有一次性`Linear(768,720)`，因此不是把dense output换到wavelet坐标。

### 2.3 Fixed conservative synthesis

`ConservativeTreeSynthesis.refine`为每个radix注册固定
$u_r=\mathbf 1/\sqrt r$与Helmert contrast $Q_r$，并计算：

$$
a^{child}=a^{parent}u_r+Q_rd.
$$

代码中的`parent.unsqueeze(-1) * scaling`对应第一项，
`einsum("...d,rd->...r", detail, contrast)`对应第二项。由于$Q_r^Tu_r=0$，detail perturbation
不能改变$u_r^Ta^{child}$。四层synthesis后得到unit leaves `[B,C,H]`，再转为`[B,H,C]`。

### 2.4 Horizon path

`target_prefix`只被转换为整数`horizon`，用于：

1. `ceil(H / block_size)`计算各层active prefix node数；
2. 对node state、coefficient group和最终leaves做prefix slice。

所有`Linear`只接收state tensor。decoder没有`LayerNorm`、`BatchNorm`、attention、router或任何沿active
node axis计算的normalization，因此H不会作为learned semantic feature改变prefix内节点。

## 3. Controls

### 3.1 `pmfo-rct-no-transition`

五个direct heads从同一`z: [B,C,768]`产生`(8,16,48,72,576)` coefficient groups，再使用与主候选
完全相同的conservative synthesis。它故意提供更大的direct capacity；如果它解释PMFO收益，则recursive
refinement不能成为paper claim。

### 3.2 `pmfo-rct-no-conservation`

它复用seed、split和parent/child context，但`detail_head_l`输出$r_l$维unconstrained update，而不是
$r_l-1$维orthogonal detail。children仍以parent scaling为anchor，但update可以具有constant component，
因此parent projection不再守恒。

### 3.3 `dense-mlp-matched`

路径为`768 -> GELU(144) -> 720 -> [:H]`。其active decoder参数为`215,136`，主PMFO为
`212,010`，差异约`1.47%`，满足预注册的近似参数匹配；它用于排除“新增nonlinearity/decoder capacity”解释。

## 4. Local Gate And Artifact Semantics

`scripts/check_stage_c_pmfo_rct_step7a.py`构造三个frozen natural profiles，检查五个arms（含A6）与
六个horizons。输出：

- `shape_prefix_checks.csv`：`dataset/profile/variant/horizon`定义case；`output_shape`来自真实model forward；
  `full_prefix_max_abs`为prefix forward与full-H720 crop的最大绝对差；`pass`表示是否不超过`1e-6`；
- `parameter_flop_audit.csv`：`total_parameters`是完整state-dict参数，包含兼容性保留但不走当前forward的
  legacy `proj_x`；`active_forward_parameters_h720`由H720 backward是否产生gradient判定；
  `decoder_parameters`与`active_decoder_parameters_h720`只统计当前readout；
  `linear_dominant_flops_*_per_series`按Linear乘加及显式basis/synthesis乘加估计，不含GELU、RevIN、reshape；
- `step7a_gate.json`：汇总shape/prefix、refinement recovery、conservation、locality与horizon-path gate；
- `step7a_manifest.json`与`environment.json`：记录seed、profiles、variants、tree contract、Python/Torch与
  execution device；
- `step7a_local_gate_report.md`：面向research decision的简表和边界说明。

2026-07-13结果：90/90 shape-prefix cases通过；full-prefix最大误差`4.172e-7`，refinement recovery
`2.384e-7`，detail perturbation conservation误差`2.682e-7`，support外变化`0`。

## 5. Code-Theory Consistency Evaluation

| Intended theory | Code realization | Remaining proxy / limit |
| --- | --- | --- |
| shared future refinement | 同层`split_l/detail_head_l`跨node共享 | mixed-radix tree固定为T=720 |
| refinement conservation | fixed $u_r,Q_r$与orthogonal detail synthesis | float32仅在`1e-6`容差内，而非symbolic proof |
| exact prefix projectivity | node-wise modules + deterministic active-prefix slice | `seed`仍先计算8个coarse states，H1不是最小理论FLOPs |
| local support | 每个detail只进入其parent subtree | locality只做synthetic perturbation，尚无trained attribution |
| horizon不是semantic feature | learned-module hook只观测tensor input，无node-axis norm | Python control flow仍读取H以决定计算域，这是设计允许项 |
| recursive mechanism有独立价值 | no-transition/no-conservation/dense controls已实现 | 必须由Step 7B训练结果判定，Step 7A不能证明 |

可证伪条件：任一trained checkpoint破坏prefix/refinement invariant；收益被dense/no-transition control解释；
或三数据集screening出现稳定退化。前者回滚Step 6，control解释回滚Step 4，numeric pathology只能否定本实现。
