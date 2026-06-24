# Phase4-S：State/Difficulty-Conditioned Supervision Scheduling

`current_step`: Step 7 local implementation complete；下一步是 commit/push 后执行 Step 8 small remote gate。

## 11-step 记录

| Field | Content |
| --- | --- |
| `current_step` | Step 7 complete |
| `problem` | 静态 horizon-free supervision units 有局部收益但整体输 R.3，是否需要 train-side condition 来分配 unit pressure |
| `existence_evidence` | Phase4-R remote gate；Phase4-S post-hoc segment diagnostic；QDF/TransDF/ElasTST/SRP++ seed evidence |
| `idea` | 训练时仍使用 horizon-free units，但 unit sampling / auxiliary pressure 由 train-side difficulty 或 state proxy 条件化 |
| `theory_check` | future-step dependency、task conflict、step heterogeneity 和 curriculum/dynamic weighting 共同支持“pressure 不应全局静态” |
| `design` | 实现 `conditioned_future_unit_scheduling`：full future dense anchor + train-side conditioned sparse unit auxiliary |
| `gate` | proxy 必须不泄漏 evaluation horizon；local smoke 后再做 small remote gate；若 trace 退化为固定 late weighting，回退 Step 4 |
| `artifacts` | `analysis/phase4_horizon_decoupled_gate_20260624/phase4_s_conditioning_diagnostic_report.md`; `artifacts/runs/smoke_phase4_s_conditioned/SmokePhase4SCFUS` |
| `decision` | S1 local smoke pass；允许进入 small remote gate，不允许直接 full matrix |

## 为什么提出 Phase4-S

[Fact] Phase4-R 的 `D2-D6` 静态 horizon-decoupled replacement 全部不通过 paper-core
gate。最好的 `D2_random_future_mask` 仍为 `+3.51%` mean relative MSE，只有 `1/12`
MSE wins。

[Fact] 失败不是 prefix consistency 或 unified evaluation interface 损坏导致。所有 strategy
的 prefix mismatch 都在 numerical-zero 量级。

[Strong Evidence] 但 horizon-free units 不是完全无效。segment-level diagnostic 显示：

- `D2_random_future_mask` 有 `4/30` segment wins，全部集中在 R.3 high-residual bucket；
- `D3_interval_supervision` 有 `5/30` segment wins，也全部集中在 R.3 high-residual bucket；
- `D3_interval_supervision` 在 late region 有 `2/3` wins，mean relative MSE 为 `-0.46%`；
- early region 上所有候选均为 `0/12` wins，且显著退化。

[Inference] 这说明 static mask/interval 的问题不是 unit 类型完全错误，而是 pressure 分配错误。
全局固定策略会在 easy/early regions 施加不必要 pressure，同时没有把 pressure 定向给
high-residual 或 late regions。

[Decision] 因此下一步不应继续调 `mask_ratio`、`interval_length` 或 component rank，也不应把
目标改成 repair R.3。R.3 只作为 baseline / carrier evidence，不是 paper-core 叙事对象。真正该问的是：

> 什么 train-side condition 能定义一个独立的 horizon-free supervision schedule，并在 full
> evaluation horizons 上超过 R.3？

## 设计修正：撤回 R.3 Repair 方向

[Decision] 撤回此前把 `S2_r3_plus_sparse_unit_aux` 作为第一优先级的判断。

原因：

1. [Fact] R.3 的强点来自 `horizon_mixed` sampling + `prefix_risk` step weighting 的组合。
   它已经把 early/shared prefix 变成主要优化压力。
2. [Risk] 若再叠加 high-residual / late-region auxiliary，就会与 R.3 的 prefix-protection
   目标发生梯度压力冲突。
3. [Decision] 即便 repair R.3 有轻微性能收益，也不能自然形成本文 core narrative，因为本文目标是
   `Horizon Supervision Scheduling for Unified Multi-Horizon Forecasting`，不是给 R.3 打补丁。
4. [Decision] R.3 后续只保留三种角色：primary baseline、carrier sanity check、诊断参照。
   不再作为被修补对象。

[Boundary] 如果未来保留一个 `R3 + auxiliary` 实验，它只能是 ablation/control：
回答“conditioned auxiliary 是否能在不破坏 R.3 的情况下提供上界参考”。它不能作为主线。

## 前人工作基础

### Forecasting 直接基础

[Strong Evidence] `QDF`：standard MSE 等价于 identity weighting，忽略 future-step
autocorrelation 和 heterogeneous task weights。它支持 objective pressure 需要被设计，但不要求
按 benchmark horizon 组织。

[Strong Evidence] `TransDF`：temporal MSE 忽略 label sequence autocorrelation；长 forecast
horizon 会造成过多 step-wise tasks 和 task conflict。它支持把 supervision 从 raw time steps
转向 transformed components 或更少的关键 units。

[Strong Evidence] `ElasTST`：varied-horizon forecasting 强调 horizon-invariant / prefix-stable
inference。它支持 evaluation horizon 是测试接口，而不是训练 schedule 的必要定义。

[Strong Evidence] `SRP++`：multi-step forecasting 存在 future-step / segment heterogeneity。
它支持不同 future regions 可能需要不同 treatment，但它本身是 representation route，不是
Phase4-S 的 training strategy 直接模板。

### 通用 training dynamics 类比基础

[Moderate Evidence] 以下工作不是 TSF 直接证据，但支持“训练 pressure 可随 difficulty / task
state 动态调整”的大原则。进入论文引用前必须导入 Zotero 并完成 fulltext note。

| Work | 可借鉴点 | 与 Phase4-S 的关系 |
| --- | --- | --- |
| [Curriculum Learning](https://dl.acm.org/doi/10.1145/1553374.1553380) | 训练顺序可影响 optimization path 与 local minima | 支持 schedule path 不是无关实现细节 |
| [Self-Paced Learning for Latent Variable Models](https://papers.nips.cc/paper/3923-self-paced-learning-for-latent-variable-models) | 根据样本 easiness 逐步纳入训练 | 支持 difficulty-conditioned sample/unit selection |
| [GradNorm](https://arxiv.org/abs/1711.02257) | 动态平衡 multi-task gradient magnitudes | 支持多 future-unit pressure 不应固定 |
| [Uncertainty loss weighting](https://arxiv.org/abs/1705.07115) | 用 task uncertainty 学习 loss weights | 支持 heterogeneous objectives 的 learned weighting |

[Boundary] 这些通用工作不能直接证明 Phase4-S 会提升 forecasting；它们只证明“conditioned
training pressure”是有理论谱系的优化思想。Phase4-S 的 forecasting 合法性仍必须由 QDF /
TransDF / ElasTST / SRP++ 和本仓库 artifacts 共同支撑。

## 可证伪假设

[Hypothesis] Horizon-free supervision units 可以改善 unified multi-horizon forecasting，但
只有当额外 pressure 被定向到 high-difficulty future units 时才有效；全局静态 pressure 会损害
easy/early regions。

形式化地，设最长训练标签为 $Y_{1:T}$，evaluation set 为：

$$
\mathcal{H}_{eval}=\{96,192,336,720\}.
$$

evaluation 只使用：

$$
\operatorname{Eval}(f_\theta;\mathcal{H}_{eval}).
$$

训练定义 horizon-free units：

$$
\mathcal{U}=\{u_1,\dots,u_K\}.
$$

Phase4-R 的失败策略等价于全局静态分布：

$$
p_t(u \mid x, y, \theta)=p(u).
$$

Phase4-S 改为：

$$
p_t(u \mid x, y, \theta)=p(u \mid c_t(x,y,\theta)),
$$

其中 $c_t$ 是只由 train split 和训练过程构造的 condition，例如 label novelty、running loss
bucket 或 residual proxy。该 condition 不能使用 test/evaluation result，也不能直接把
`96,192,336,720` 写回 training schedule。

## 候选设计

### S1：Conditioned Future-Unit Scheduling

[Decision] 第一优先级。

核心：训练仍使用 `pred_len=720` 的 full future sequence，但 supervision pressure 由
train-side condition 决定，而不是由 evaluation horizon 或 R.3 loss 决定。

定义 horizon-free units：

$$
\mathcal{U}=\{u_1,\dots,u_K\},
$$

其中 $u$ 可以是 interval、block mask、frequency band 或 component group。训练 loss 写成：

$$
\mathcal{L}_{CFUS}
=
\frac{1}{|\mathcal{T}|}\sum_{t\in\mathcal{T}} e_t^2
+
\lambda
\sum_{u\in\mathcal{U}} a(u\mid c_t)\mathcal{L}_u,
$$

其中第一项是 full-time dense anchor，第二项是 conditioned sparse pressure。关键约束：

- dense anchor 保护全 future coverage，避免只优化 late/hard units；
- $a(u\mid c_t)$ 由 train-side condition 决定；
- 不采样 `96,192,336,720` 作为 training horizon；
- 不复用 R.3 的 prefix-risk 权重作为主 loss。

可用 condition：

| Condition | Source | 用途 |
| --- | --- | --- |
| label novelty | train label 与 recent history / local mean 的偏离 | 找到非平稳或突变 future units |
| local variation | train label 的局部差分能量 | 找到高频或难拟合区间 |
| online loss bucket | training trace 中 unit loss 的 running EMA | 动态补强当前难学 units |
| train residual proxy | 仅 train split 上的 out-of-fold 或 previous-epoch residual | 作为后续扩展，不做第一版 |

### S2：Difficulty-Conditioned Interval

核心：

$$
p(u=\text{interval}_{a:b}) \propto \phi(c_t(a:b)),
$$

其中 $c_t$ 可以来自 train-label novelty、局部 variation、running loss bucket 或 train residual
proxy。interval 不按 evaluation horizon 定义。

风险：

- 若 proxy 实际上只是 future position index，会退化为手工 late weighting；
- 若 proxy 依赖 validation/test residual，会造成 leakage。

### S3：R.3 + Auxiliary Control

[Decision] 只作为 control，不作为主线。

用途：

- 验证 conditioned auxiliary 是否天然与 R.3 prefix-risk 冲突；
- 给出上界参照；
- 若它赢，也只能说明 R.3 可被修补，不能直接成为 HSS 主贡献。

### S4：Error-Process Reweighting

核心：先估计 train split 上的 residual/error process，再对 future units 施加附加 pressure。

当前不作为第一实现，因为：

- residual proxy 容易和 evaluation horizon 绑定；
- 需要更严格的 split 管理；
- Phase2 的 QDF/covariance route 已提示直接 transfer objective matrix 风险较高。

## Phase4-S 进入实现前的 gate

实现前必须满足：

1. condition 明确来自 train split 或 online training trace；
2. condition 不直接使用 evaluation horizon identity；
3. 有一份 local diagnostic 说明该 condition 与 high-residual / late-region weakness 有关系；
4. 第一版主线不得以 R.3 loss 为 base；R.3 只能作为 baseline/control；
5. local smoke 必须输出 `supervision_trace.csv`，记录 condition bucket、unit type、active steps、
   auxiliary weight 和 loss；
6. small remote gate 先跑 `ETTh2` + `Weather`，不直接 full matrix；
7. gate 必须包含 h96/early-region no-collapse，避免用 late-region 收益掩盖 early-prefix 退化。

## S1 本地实现计划

[Decision] 第一版实现 `conditioned_future_unit_scheduling`，不实现 `R.3 + auxiliary`。

CLI 参数：

| Argument | Default | Meaning |
| --- | --- | --- |
| `--supervision-strategy conditioned_future_unit_scheduling` | - | 启用 S1 |
| `--supervision-condition` | `label_novelty` | condition score 类型 |
| `--supervision-condition-top-ratio` | `0.25` | 每个 batch 选择 top-scored future blocks 的比例 |
| `--supervision-aux-weight` | `0.1` | sparse unit auxiliary loss 权重 |
| `--supervision-block-size` | `48` | future unit block size |

第一版 condition：

$$
c(u_{a:b})=
\frac{1}{B(b-a)C}
\sum_{i=1}^{B}\sum_{t=a}^{b}\sum_{j=1}^{C}
(y_{i,t,j}-x_{i,L,j})^2.
$$

其中 $x_{i,L,j}$ 是最后一个 history step。该 score 只使用当前 train batch 的 history 和
future label，不使用 validation/test，不使用 evaluation horizon identity。

训练 loss：

$$
\mathcal{L}
=
\operatorname{MSE}(\hat{Y}_{1:720},Y_{1:720})
+
\lambda
\operatorname{MSE}(\hat{Y}_{u^\star},Y_{u^\star}),
$$

其中 $u^\star$ 是 top condition blocks 的并集。

smoke 必须检查：

1. `effective_config.json` 中 `training_evaluation_decoupled=true`；
2. `train_horizons_effective=[720]`；
3. `supervision_trace.csv` 中 `unit_type=conditioned_sparse`；
4. trace 记录 `condition_type`、`condition_top_blocks`、`condition_mean_score`、`auxiliary_weight`；
5. validation/test 仍覆盖 `96,192,336,720`；
6. prefix mismatch 保持 numerical-zero 量级。

## S1 本地 smoke 结果

[Fact] 已完成 local smoke：

```bash
conda run -n r2026-fsa python baselines/patch_encoder_target_set_decoder/train.py \
  --dataset ETTh2 \
  --target-horizons 96,192,336,720 \
  --supervision-strategy conditioned_future_unit_scheduling \
  --supervision-pred-len 720 \
  --supervision-condition label_novelty \
  --supervision-condition-top-ratio 0.25 \
  --supervision-aux-weight 0.1 \
  --epochs 1 \
  --steps-per-epoch 2 \
  --max-eval-batches 1 \
  --batch-size 16 \
  --run-name SmokePhase4SCFUS \
  --output-root artifacts/runs/smoke_phase4_s_conditioned \
  --device cpu
```

smoke artifact：

- `artifacts/runs/smoke_phase4_s_conditioned/SmokePhase4SCFUS/ETTh2/mixed_h96_h192_h336_h720/seed2021`

[Verification] smoke 通过：

- `training_evaluation_decoupled=true`;
- `train_horizons_effective=[720]`;
- `evaluation_target_horizons=[96,192,336,720]`;
- `step_loss_weighting=uniform`;
- `supervision_trace.csv` 中 `unit_type=conditioned_sparse`;
- trace 记录 `condition_type=label_novelty`、`condition_top_blocks=4`、`auxiliary_weight=0.1`;
- prefix mismatch 为 numerical-zero 量级。

## Step 8 small remote gate

[Decision] 允许进入 small remote gate，但不允许直接 full matrix。

入口脚本：

- `scripts/remote/run_phase4_s_cfus_gate.sh`

默认矩阵：

| Dimension | Values |
| --- | --- |
| datasets | `ETTh2`, `Weather` |
| strategies | `conditioned_future_unit_scheduling`, `full_time_mse`, `r3_prefix_risk` |
| evaluation horizons | `96,192,336,720` |
| condition | `label_novelty` |
| top ratio | `0.25` |
| aux weight | `0.1` |

small gate 判定：

1. CFUS 相对 `full_time_mse` 必须有明确收益；
2. CFUS 相对 R.3 的 gap 必须显著小于 Phase4-R static strategies；
3. `h96` / early-region 不得系统性 collapse；
4. `Weather` 不得出现类似 `D3_interval_supervision` 的明显退化；
5. trace 必须证明 selected units 由 condition 驱动，而不是固定 late weighting。

## 回退规则

- 若 train-side proxy 无法定义，回退到 Step 1：补 literature 和 data diagnostic。
- 若 proxy 定义成立但 local smoke 显示 trace 退化为固定 late weighting，回退到 Step 4。
- 若 CFUS 相对 R.3 无接近空间，并且只优于 full-time MSE，则不能作为 paper-core，回退到 Step 2-3。
- 若 small remote gate 只在 `ETTh2` 有效而 `Weather` 明显退化，回退到 Step 6，重新设计
  condition，而不是调大 sweep。
- 若只有 `R.3 + auxiliary control` 有效，而独立 CFUS 无效，则说明当前方向退化为 R.3 repair；
  Phase4-S 不能作为主线继续。
