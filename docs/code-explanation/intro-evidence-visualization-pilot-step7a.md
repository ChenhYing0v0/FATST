# Introduction Evidence Visualization Pilot Step7A 代码说明

## 1. 状态与作用

| Field | Content |
| --- | --- |
| `date` | `2026-07-29` |
| `protocol` | `SC-UVHF-INTRO-EVIDENCE-VIZ-v1` |
| `role` | exploratory visualization only |
| `initial_matrix` | Weather × seed2021 × (4 DLinear horizons + 5 neutral sharing extents) |
| `test_accessed` | false |
| `formal_problem_gate` | false |

本次实现只为Introduction寻找清晰但不过度极端的illustrative examples。它不修改
ISCF-BSCA，不构成新的paper method，也不替代正式跨dataset/seed evidence。

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

案例选择不是最大值：

1. 对每个origin汇总六个horizon pairs的mean disagreement；
2. 选择最接近85% quantile的origin；
3. 对channel使用相同85% quantile规则；
4. 在`sample_selection`相关summary fields中公开真实percentile。

输出包括：

- `prefix_disagreement_overlay.svg/png`；
- `prefix_disagreement_heatmap.svg/png`；
- `pair_metrics.csv`；
- `summary.json`。

远程pilot结果返回后的视觉审计发现，85% quantile案例的raw forecast curves真实但
彼此接近。因此overlay下半panel不再重复绘制四条绝对预测，而是固定以H720 forecast
为零参考，绘制H96/H192/H336/H720在共同96-step prefix上的prediction difference。
该修改不改变origin、channel、metric或quantile，只改变同一证据的显示坐标，避免
继续搜索更极端样本。

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

对唯一sharing extent $s$：

```text
U[:,:,start:end,:]
  -> mean over future-step axis
  -> LayerNorm
Z_g [B,C,64]
  -> broadcast to every step in the same block
Z   [B,C,T,64]
```

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

## 6. Runner

`scripts/remote/run_intro_evidence_visualization_pilot.sh`默认：

```text
Weather, seed2021
DLinear: H96/H192/H336/H720       4 runs
Neutral: s1/s8/s32/s128/s720      5 runs
total                              9 runs
```

三张GPU按neutral workload优先分配。runner支持：

- `DRY_RUN=1`：五个scale的CPU synthetic gradient/shape smoke；
- `RESOURCE_SMOKE=1`：一个CUDA synthetic smoke；
- `STATUS_ONLY=1`：报告neutral与DLinear完成数；
- `RUN_MODE=all|sharing-only|prefix-only`：只执行需要的evidence family；
- restart-safe skip existing；
- 完成对应family后自动生成相应visualization。

后续visualization dataset search使用
`DATASET=ETTm1 RUN_MODE=sharing-only`，只新增五个neutral scales，不重复已经能够
形成figure candidate的DLinear prefix runs。config同时约束ETTm1为当前唯一授权
fallback；ETTh2仍不可由runner自动启动。

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
- best scale不随region变化且无risk crossover；
- prefix disagreement只存在于极端top-1%样本，85% quantile不可见。

以上情况只能阻断当前visualization或exact diagnostic，不能自动否定fixed
ISCF-BSCA architecture。
