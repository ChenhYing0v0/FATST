# Introduction Evidence Visualization Pilot Step7A 代码说明

## 1. 状态与作用

| Field | Content |
| --- | --- |
| `date` | `2026-07-30` |
| `protocol` | `SC-UVHF-INTRO-EVIDENCE-FULL-SEARCH-v1` |
| `role` | exploratory visualization only |
| `matrix` | 5 datasets × seed2021 × (4 DLinear horizons + 5 neutral sharing extents) |
| `test_accessed` | false |
| `formal_problem_gate` | false |

本次实现只为Introduction寻找清晰的illustrative examples。它不修改ISCF-BSCA，
不构成新的paper method，也不替代正式跨dataset/seed evidence。当前允许选择
validation search space中的最强样本，但必须公开selection rule与search space。

## 2. Prefix-disagreement artifact path

### 2.1 DLinear validation export

`baselines/dlinear/train.py`保留原有DLinear architecture，只扩展artifact接口：

```text
x: [B,L,C]
  -> DLinear
pred: [B,H,C]
```

训练完成并恢复best-validation checkpoint后，新增保存：

```text
predictions_val.npz
  pred:         [N_val,H,C]
  true:         [N_val,H,C]
  history:      [N_val,L,C]
  origin_index: [N_val]
  train_mean:   [1,C]
  train_std:    [1,C]
metrics_val.json
```

`--skip-test`使visualization pilot完全不构造test dataset，也不产生
`predictions_test.npz`。默认不传该参数时，历史paper-facing behavior保持不变：
仍会保存test metrics与predictions。

### 2.2 Prefix analyzer

`scripts/analyze_intro_prefix_disagreement.py`读取
$H\in\{96,192,336,720\}$的四份validation artifacts。

它首先取四个runs都具备的最小origin count，并验证：

1. history tensor完全对齐；
2. H96/H192/H336 target分别等于H720 target的相同prefix；
3. prediction shape与declared horizon一致；
4. padding未进入comparison。

对每个$H_i<H_j$计算：

```text
abs(pred_Hi - pred_Hj[:, :Hi])
  -> mean over origin, future step, channel
  -> NCHPD
```

由于dataset output已经按train scaler标准化，标准化空间的mean absolute
disagreement等价于inverse-transform后再除以train standard deviation。

当前full search对每个`origin × channel`联合单元计算六个horizon pairs的
mean-over-overlap disagreement，再选择全局最大联合单元。联合选择避免“分别选
origin与channel后，组合单元本身并不强”的问题。`origin_channel_candidates.csv`
保存所有候选及其aggregate score；`summary.json`公开：

- `selection_mode=maximum`；
- 搜索的origin-channel单元数；
- selected joint score与percentile；
- 独立的origin/channel marginal percentile。

该选择是明确的maximum validation example，不是representative sample或prevalence
estimate。score先在每个overlap内对future steps取均值，避免由一个孤立spike决定。

输出包括：

- `prefix_disagreement_overlay.svg/png`；
- `prefix_disagreement_heatmap.svg/png`；
- `pair_metrics.csv`；
- `summary.json`。

overlay下半panel固定以H720 forecast为零参考，绘制H96/H192/H336/H720在共同
96-step prefix上的prediction difference；这只改变显示坐标，不改变selection
statistic。

## 3. Neutral single-sharing-extent model

### 3.1 Forward tensor flow

实现位于`baselines/intro_evidence_neutral/model.py`。

输入：

$$
\mathbf X\in\mathbb R^{B\times L\times C},
\qquad L=96.
$$

首先按channel编码raw history：

```text
X [B,L,C]
  -> transpose
X_c [B,C,L]
  -> Linear(96,128) -> GELU -> Linear(128,64)
R [B,C,64]
```

future-step descriptors：

```text
step_embedding: [T,32], T=720
```

candidate state generator利用线性层对concat的可分解性，避免显式构造
`[B,C,T,96]` concat：

```text
history_to_hidden(R)       [B,C,128]
step_to_hidden(phi)        [T,128]
  -> broadcast add + GELU
joint_hidden               [B,C,T,128]
  -> hidden_to_state
U                          [B,C,T,64]
```

这与对`[R,phi_tau]`施加一个joint affine layer再接GELU等价，但显著减少concat
memory。

对唯一sharing extent $s$，完整blocks被一次性reshape：

```text
U_prefix [B,C,G*s,64]
  -> reshape [B,C,G,s,64]
  -> mean(s) + LayerNorm
Z_g [B,C,G,64]
  -> expand(s) + reshape
Z_prefix [B,C,G*s,64]
```

若$T$不能被$s$整除，只对最后一个tail单独执行一次mean、LayerNorm与broadcast。
因此$s=1$从720次Python pooling/LayerNorm调用降为一次batched call；所有scale
最多两次调用。

最后每个future step仍有独立synthesis vector：

```text
synthesis [T,64]
Z * synthesis -> sum(state_dim) + output_bias[T]
forecast [B,C,T] -> transpose -> [B,T,C]
```

因此$s=720$只共享history-conditioned latent state，不会强迫720个outputs相同；
step-specific synthesis仍保留不同future-step coefficients。

### 3.2 Matched parameter and compute contract

五个variants只改变Python integer `sharing_extent`，它不创建parameter。所有
variants都计算完整candidate tensor
$\mathbf U\in\mathbb R^{B\times C\times720\times64}$，然后进行parameter-free
pooling。

本地contract：

| Scale | Parameters | Prediction shape | Candidate shape | Pooled shape |
| ---: | ---: | --- | --- | --- |
| 1 | 111,312 | `[2,720,3]` | `[2,3,720,64]` | `[2,3,720,64]` |
| 8 | 111,312 | `[2,720,3]` | `[2,3,720,64]` | `[2,3,720,64]` |
| 32 | 111,312 | `[2,720,3]` | `[2,3,720,64]` | `[2,3,720,64]` |
| 128 | 111,312 | `[2,720,3]` | `[2,3,720,64]` | `[2,3,720,64]` |
| 720 | 111,312 | `[2,720,3]` | `[2,3,720,64]` | `[2,3,720,64]` |

五个variants的14组parameters均得到finite nonzero gradients。将同一state dict
加载到$s=1$和$s=720$后，maximum prediction gap为`0.08002925`，说明intervention
没有退化为functionally identical label。

向量化实现与原loop reference在五个scales上的local double-precision contract：

| Quantity | Maximum gap |
| --- | ---: |
| pooled output | `0.0` |
| candidate-state gradient | `0.0` |
| LayerNorm weight gradient | `1.85e-13` |
| LayerNorm bias gradient | `2.27e-13` |

这证明改动只消除operator-launch/Python-loop overhead，没有改变被审计的函数与
反向传播语义。

## 4. Neutral training and artifacts

`baselines/intro_evidence_neutral/train.py`：

- 使用channel-wise standardized raw history；
- loss为uniform full-domain MSE；
- validation MSE选择checkpoint；
- 只构造train/validation datasets；
- 从不构造或访问test split；
- 保存best checkpoint、validation predictions、history、targets、train scaler、
  training log、effective config与environment。

每个run目录：

```text
NeutralSharingExtent/<dataset>/s<scale>/seed<seed>/
  checkpoint.pt
  predictions_val.npz
  metrics_val.json
  training_log.csv
  effective_config.json
  environment.json
```

## 5. Sharing-demand analyzer

`scripts/analyze_intro_sharing_demand.py`先检查：

- 五个scales parameter count完全相同；
- prediction、target、history shapes相同；
- target/history alignment gap不超过$10^{-6}$；
- 所有artifacts来自validation。

随后计算：

```text
error^2 [N,720,C]
  -> mean over N,C
step_risk [5,720]
  -> split into 12 bins of 60 steps
region_risk [5,12]
```

以全域validation MSE最低的single scale为`best fixed`，画：

1. sharing extent × future region relative-risk heatmap；
2. 第一组达到frozen margin的crossing pair与best-fixed scale的31-step smoothed
   risk curves；
3. 同一组display scales相对best-fixed scale的12-region risk contrast。

display scales由统计量自动确定，而不是手工按dataset选择：有margin-qualified
crossing时使用`first crossing pair + best fixed`；否则使用
`s1 + best fixed + s720`。第三panel直接展示问题层面的region-wise sign与magnitude，
不再用全域region-oracle headroom替代crossover证据；oracle数值仍保留在
`summary.json`中。

`crossover_visualization_candidate`只要求至少两个descriptive region winners且
存在margin-qualified crossing pair，用于区分“问题存在性图是否可读”；
`visualization_signal`继续保留更严格的oracle-headroom条件，两者不能混用。
第二panel会公开displayed crossing pair与best-fixed scale；若没有qualified pair，
则明确标注。第一个panel的逐region最小值轨迹称为`Descriptive argmin`。

输出：

- `sharing_demand_visualization.svg/png`；
- `step_risk.csv`；
- `region_risk.csv`；
- `summary.json`。

### 5.1 Sample-level heterogeneity selector

新增`scripts/analyze_intro_sharing_sample_candidates.py`。它不在所有origins上先
平均，而是保留sample axis：

```text
error^2 [N,720,C]
  -> mean over C
sample_step_risk [N,5,720]
  -> reshape and mean over each 60-step region
sample_region_risk [N,5,12]
```

对每个origin定义：

- `supported_winner_count`：赢得至少两个region的scale数量；
- `distinct_winner_count`：至少赢得一个region的scale数量；
- `winner_entropy`：12-region winner histogram的$\log 5$归一化entropy；
- `qualified_crossing_pair_count`：双向relative ordering均超过0.5%的scale pair数；
- `mean_winner_margin`：每个region第一名相对第二名的normalized gap均值；
- `sample_oracle_headroom`：best fixed scale相对region oracle的descriptive gap。

选择按以上顺序lexicographic maximization，最后才以origin index作deterministic
tie-break。先使用`supported_winner_count`，避免单个region的偶然胜者主导选择。
每个region已同时聚合60个future steps和所有channels。

输出包括全部sample候选、selected sample的region/step risk、三panel候选图与
`summary.json`。图和summary都标注`maximum heterogeneity candidate`。

## 6. Runner

原pilot runner保留用于单dataset复现。当前full search使用
`scripts/remote/run_intro_evidence_full_search.sh`：

```text
datasets: ETTh1, ETTh2, ETTm1, ETTm2, Weather
complete target: 25 neutral + 20 DLinear runs
reused complete artifacts: 14
new runs: 31
```

31个missing jobs进入一个global three-GPU queue。预计更慢的$s=1$、
long-horizon与较慢datasets排在前面，但不固定到某张GPU；任意GPU完成后立刻领取
下一个job。因此GPU 0不再天然成为critical path。runner支持：

- `DRY_RUN=1`：运行完整local contract并检查所有analyzer CLI；
- `STATUS_ONLY=1`：报告全五dataset完成数；
- restart-safe skip existing；
- 任一job失败时保留其他已完成artifacts并阻止analysis promotion；
- 完成45-run artifact contract后，为每个dataset运行aggregate与sample-level
  analyzers，再生成跨dataset ranking。

默认output root：

`/home/yingch/exp_outputs/r-2026-fatst/intro_evidence_visualization_pilot_v1`

## 7. Code-theory consistency

### Intended theory

不同future regions可能偏好不同的cross-step sharing extent；prefix-specific
models可能对相同future steps给出不同predictions。

### Code realization

- DLinear四个horizons为独立训练的functions；
- neutral family每个run只含一个sharing extent；
- parameter、candidate-state generator、step synthesis与training protocol匹配；
- sharing extent只作用于history-conditioned state pooling；
- requested horizon不进入neutral model。

### 仍然只是proxy的部分

- DLinear local adapter不是最终native multi-family paper evidence；
- single Weather/seed只适合illustration；
- region oracle来自同一validation split，不是out-of-sample adaptive benefit；
- neutral family只能回答该exact intervention point。

### Falsification

- same-origin target/history alignment失败；
- parameter count或candidate path随scale变化；
- scale endpoints functionally indistinguishable；
-某一scale出现numeric/optimization pathology；
- selected candidate只由单step spike或单region噪声支持；
- maximum candidate仍不能形成可读的prefix difference或multi-scope ridge。

以上情况只能阻断当前visualization或exact diagnostic，不能自动否定fixed
ISCF-BSCA architecture。
