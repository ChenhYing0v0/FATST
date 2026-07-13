# Phase4-S：State/Difficulty-Conditioned Supervision Scheduling

`current_step`: Step 9-11 complete；S1 和 S2 small remote gate 均已完成。当前结论是
S1 `conditioned_future_unit_scheduling` 与 S2 `predictability_downweight` 都不通过
paper-core gate；下一步回退到 Step 5/6，重做 predictability proxy 与 shielding 机制。

## 11-step 记录

| Field | Content |
| --- | --- |
| `current_step` | Step 9-11 complete |
| `problem` | 静态 horizon-free supervision units 有局部收益但整体输 R.3；train-side condition 能否区分 learnable-hard 与 noisy-hard，并调度 supervision pressure |
| `existence_evidence` | Phase4-R remote gate；Phase4-S post-hoc segment diagnostic；S1 CFUS gate；S2 predictability gate；QDF/TransDF/ElasTST/SRP++ seed evidence |
| `idea` | 训练时仍使用 horizon-free units，但 unit pressure 由 train-side difficulty / predictability / state proxy 条件化 |
| `theory_check` | future-step dependency、task conflict、step heterogeneity 和 SRP++ interference 共同支持“pressure 不应全局静态”，但 noisy-hard 不能简单加压 |
| `design` | S1 `conditioned_future_unit_scheduling`; S2 `predictability_downweight` |
| `gate` | 保留 full-time gain；缩小 R.3 gap；Weather 不 collapse；trace 证明 condition/split 生效；diagnostic 能解释收益来源 |
| `artifacts` | `analysis/phase4_s_cfus_gate_20260624`; `analysis/phase4_predictability_diagnostic_20260624`; `analysis/phase4_s_predictability_gate_20260625` |
| `decision` | S1 与 S2 均 fail as paper-core；不进入 full matrix；回退 Step 5/6，重新设计 stronger predictability proxy 或 isolated shielding path |

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

[Fact] small remote gate 已完成。

入口脚本：

- `scripts/remote/run_phase4_s_cfus_gate.sh`

remote output root：

- `/home/yingch/exp_outputs/r-2026-fatst/phase4_s_cfus_gate`

local analysis root：

- `analysis/phase4_s_cfus_gate_20260624`

分析脚本：

- `scripts/analyze_phase4_s_cfus_gate.py`

决策报告：

- `analysis/phase4_s_cfus_gate_20260624/phase4_s_cfus_gate_decision_report.md`

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

## Step 9-11：结果分析与决策

[Fact] `conditioned_future_unit_scheduling` 相比 `full_time_mse` 有清楚收益：

| Baseline | Settings | MSE wins | MAE wins | Mean relative MSE |
| --- | ---: | ---: | ---: | ---: |
| `D0_full_time_mse` | 8 | 6 | 8 | `-2.74%` |

[Fact] 相比 primary baseline `R.3_prefix_risk`，S1 没有通过：

| Baseline | Settings | MSE wins | MAE wins | Mean relative MSE |
| --- | ---: | ---: | ---: | ---: |
| `D1_r3_prefix_risk` | 8 | 3 | 3 | `+2.22%` |

[Fact] dataset split 暴露了机制不稳定：

| Dataset | Baseline | Settings | MSE wins | Mean relative MSE |
| --- | --- | ---: | ---: | ---: |
| `ETTh2` | `D1_r3_prefix_risk` | 4 | 3 | `-0.35%` |
| `Weather` | `D1_r3_prefix_risk` | 4 | 0 | `+4.78%` |

[Strong Evidence] S1 的 train/eval 解耦实现是干净的：

- `train_horizons_effective=720`;
- `step_loss_weighting=uniform`;
- `unit_type=conditioned_sparse`;
- `condition_type=label_novelty`;
- prefix mismatch 仍是 numerical-zero 量级。

[Strong Evidence] S1 不是无效机制。它在 ETTh2 上相对 full-time 和 R.3 都有接近或收益，
说明 train-side conditioned sparse pressure 可以改变优化结果。

[Counter-Evidence] S1 不能作为 paper-core。Weather 相比 R.3 四个 horizons 全输，
且 h96/early-region 没有证明被保护。当前策略更像对 full-time dense anchor 的局部补强，
不是稳定的 `Horizon Supervision Scheduling`。

[Diagnostic Gap] 当前 `supervision_trace.csv` 只记录 `condition_top_blocks=4` 和
`condition_mean_score`，没有记录 selected block indices / block ranges / per-block scores。
因此不能排除 `label_novelty` 实际退化为固定 late weighting proxy。

[Decision] S1 不进入 full matrix；不继续 sweep `top_ratio` 或 `aux_weight`。当前回退到
Step 6：重做 condition 可观测性与 CFUS-v2 schedule。

下一步最小任务：

1. 在 trace 中记录 selected block indices、block ranges 和 per-block condition scores。
2. 做 offline train-label condition diagnostic，确认 `label_novelty` 是否长期偏向 late blocks。
3. 如果偏向 late blocks，改用 `novelty within future-region groups` 或
   `balanced condition buckets`，让 condition 同时保护 early/easy regions。
4. 只有 CFUS-v2 的 local trace 证明不是固定 late weighting 后，再启动新的 small gate。

## Step 4-6 回退：Predictability-Conditioned Scheduling

[Question] 当前 S1 的隐含假设是 high-novelty / hard blocks 应该被加压。但 hard block
至少有两类：

| Type | 含义 | 合理 training action |
| --- | --- | --- |
| `learnable-hard` | 难，但包含可学习结构，例如 smooth shift、趋势变化或周期错位 | 加压或重点学习 |
| `noisy-hard` / `low-predictability` | 难，因为局部扰动强、简单 predictor 也无法解释 | 降权、隔离或只让 auxiliary path 学习 |

[Fact] SRP++ 不直接提出“avoid hard steps”，但它支持一个关键前提：不同 future
steps/segments 的学习压力会互相干扰，step-specific adaptation 通过局部 adapter 和 frozen
foundation 避免 long-range adaptation 破坏 short-term prediction。对本项目的启发是：
HSS 不一定只做 hard-block emphasis，也可以做 hard-block shielding。

[Decision] 新方向命名为：

> Predictability-Conditioned Supervision Scheduling

核心问题：

> train-side condition 应该区分 learnable-hard 与 noisy-hard。前者可以加压，后者应避免污染
> shared representation。

### Offline Diagnostic

[Fact] 已新增 train-only diagnostic：

- script: `scripts/analyze_phase4_predictability_diagnostic.py`;
- analysis root: `analysis/phase4_predictability_diagnostic_20260624`;
- report:
  `analysis/phase4_predictability_diagnostic_20260624/phase4_predictability_diagnostic_report.md`;
- datasets: `ETTh2`, `Weather`;
- split: train only;
- pred_len: `720`;
- block_size: `48`;
- selected blocks: 按当前 S1 `label_novelty` top ratio `0.25` 复现。

诊断指标：

| Metric | Meaning |
| --- | --- |
| `novelty_mse` | 当前 CFUS 的 label novelty，即 future block 相对最后一个 history step 的 MSE |
| `seasonal24_mse` | 24-step seasonal naive reference 的 MSE |
| `best_naive_mse` | `min(novelty_mse, seasonal24_mse)`，越高表示简单可预测性越弱 |
| `local_variation` | future block 内部一阶差分能量 |
| `smoothness_ratio` | `local_variation / novelty_mse` |

[Fact] 诊断结果：

| Dataset | Selected best-naive ratio | Selected local-variation ratio | Late-block selected share | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `ETTh2` | `2.51x` | `1.20x` | `0.485` | selected blocks 更像 smooth shift / learnable-hard |
| `Weather` | `2.83x` | `8.45x` | `0.436` | selected blocks 明显是 high-variation / noisy-hard |

[Strong Evidence] 这解释了 S1 small gate 的 dataset split：ETTh2 上 hard-block emphasis
可以接近或超过 R.3；Weather 上同样的 label-novelty emphasis 会把大量 high-variation
low-predictability blocks 加权，污染 shared 720-step dense learning。

[Decision] 当前 `label_novelty` 不是稳定的 difficulty proxy。下一版不应继续简单给
high-novelty blocks 加压，也不应仅做 balanced top-k。真正的 CFUS-v2 应转向：

> learnable-hard emphasis + noisy-hard downweight/isolation。

### CFUS-v2 候选

最小设计：

$$
\mathcal{L}
=
\sum_{u\in\mathcal{U}} w_{\text{pred}}(u)\mathcal{L}_u
+
\lambda
\sum_{u\in\mathcal{U}_{learnable-hard}} a(u)\mathcal{L}_u.
$$

其中：

- $w_{\text{pred}}(u)$ 是 predictability-aware dense weight，低可预测 block 不再主导梯度；
- $\mathcal{U}_{learnable-hard}$ 是 high novelty 但低 local variation / smooth shift 的 blocks；
- noisy-hard blocks 保留 floor weight，避免完全丢弃，但不再获得额外 auxiliary pressure。

更激进但暂缓的 SRP-inspired 版本：

$$
\mathcal{L}
=
\mathcal{L}_{shared,predictable}
+
\lambda\mathcal{L}_{isolated,low\text{-}predictability}.
$$

该版本需要 lightweight adapter 或 detached auxiliary path，当前只作为 Step 6 后续候选，不直接实现。

### 下一步 Gate

1. 先实现 `predictability_downweight`，不改 architecture；
2. trace 必须记录 per-block predictability score、learnable-hard/noisy-hard bucket、实际 loss weight；
3. local smoke 必须证明 train/eval 仍解耦，evaluation horizons 仍只用于测试；
4. small gate 仍只跑 `ETTh2 + Weather`；
5. pass 条件：保留 CFUS 相对 `full_time_mse` 的收益，同时消除 Weather vs R.3 的全面退化。

### S2 本地实现与 Smoke

`current_step`: Step 9-11 complete。

[Decision] 已实现最小 `predictability_downweight`，不改 architecture，只改 training loss：

- branch: `--supervision-strategy predictability_downweight`;
- code explanation:
  `docs/code-explanation/phase4-s-predictability-scheduling.md`;
- remote runner:
  `scripts/remote/run_phase4_s_predictability_gate.sh`。

训练 loss：

$$
\mathcal{L}
=
\mathcal{L}_{time,predictability-weighted}
+
\lambda\mathcal{L}_{learnable-hard}.
$$

其中：

- high novelty blocks 是 hard candidates；
- high variation blocks 是 noisy-hard proxy；
- `noisy_blocks = top_novelty \cap top_variation`，在 dense loss 中使用 floor weight；
- `learnable_blocks = top_novelty - top_variation`，获得 auxiliary emphasis；
- step weights 均值归一化，避免整体 loss scale 改变过大。

[Verification] local CPU smoke 已通过：

```bash
conda run -n r2026-fsa python baselines/patch_encoder_target_set_decoder/train.py \
  --dataset ETTh2 \
  --target-horizons 96,192,336,720 \
  --supervision-strategy predictability_downweight \
  --supervision-pred-len 720 \
  --supervision-condition-top-ratio 0.25 \
  --supervision-aux-weight 0.1 \
  --supervision-predictability-floor-weight 0.5 \
  --epochs 1 \
  --steps-per-epoch 2 \
  --max-eval-batches 1 \
  --batch-size 16 \
  --run-name SmokePhase4SPredictabilityDownweight \
  --output-root artifacts/runs/smoke_phase4_s_predictability \
  --device cpu
```

smoke artifact：

- `artifacts/runs/smoke_phase4_s_predictability/SmokePhase4SPredictabilityDownweight/ETTh2/mixed_h96_h192_h336_h720/seed2021`

[Fact] smoke trace 显示：

- `training_evaluation_decoupled=true`;
- `train_horizons_effective=[720]`;
- `unit_type=predictability_downweight`;
- `condition_type=novelty_x_variation`;
- trace 记录 `predictability_learnable_blocks`、`predictability_noisy_blocks`、
  `predictability_mean_weight`、`predictability_floor_weight`;
- prefix mismatch 保持 numerical-zero 量级。

[Fact] S2 small remote gate 已完成：

- remote output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase4_s_predictability_gate`;
- local analysis root:
  `analysis/phase4_s_predictability_gate_20260625`;
- analysis script:
  `scripts/analyze_phase4_s_predictability_gate.py`;
- decision report:
  `analysis/phase4_s_predictability_gate_20260625/phase4_s_predictability_gate_decision_report.md`。

主要结果：

| Comparison | Settings | MSE wins | MAE wins | Mean relative MSE |
| --- | ---: | ---: | ---: | ---: |
| S2 vs `D0_full_time_mse` | 8 | 4 | 5 | `-2.61%` |
| S2 vs `D1_r3_prefix_risk` | 8 | 3 | 3 | `+2.35%` |
| S2 vs S1-CFUS | 8 | 2 | 3 | `+0.13%` |

dataset split：

| Dataset | Baseline | Settings | MSE wins | Mean relative MSE |
| --- | --- | ---: | ---: | ---: |
| `ETTh2` | `D1_r3_prefix_risk` | 4 | 3 | `-0.34%` |
| `Weather` | `D1_r3_prefix_risk` | 4 | 0 | `+5.05%` |

[Strong Evidence] trace 证明 S2 的 noisy/learnable split 生效：

- ETTh2: mean learnable blocks `3.10`，mean noisy blocks `0.91`;
- Weather: mean learnable blocks `2.08`，mean noisy blocks `2.11`。

[Counter-Evidence] 指标没有支持 S2：Weather 相对 R.3 仍然 `0/4` wins；相对
`full_time_mse` 也退化 `+0.23%` mean relative MSE；相对 S1-CFUS 整体 `+0.13%`
mean relative MSE。

[Decision] S2 不通过 paper-core gate，不进入 full matrix，不继续 sweep 当前
`floor_weight=0.5`。当前失败的是简单 `top_novelty ∩ top_variation` proxy 和 shared dense
downweight formulation，不是 predictability-conditioned scheduling 问题本身。

[Decision] 回退到 Step 5/6：

1. 重新评估 predictability proxy：仅用 local variation 过粗；
2. 下一步应做 train-only baseline residual / seasonal residual stability / running residual
   stability diagnostic；
3. 若继续 noisy-hard shielding，应考虑 detached 或 isolated auxiliary path，而不是只在
   shared dense loss 中降权；
4. S1-CFUS 保留为 evidence：hard-block emphasis 对 ETTh2 有效，但需要 dataset/state-aware
   gate 判断何时启用。

## 回退规则

- 若 train-side proxy 无法定义，回退到 Step 1：补 literature 和 data diagnostic。
- 若 proxy 定义成立但 local smoke 显示 trace 退化为固定 late weighting，回退到 Step 4。
- 若 CFUS 相对 R.3 无接近空间，并且只优于 full-time MSE，则不能作为 paper-core，回退到 Step 2-3。
- 若 small remote gate 只在 `ETTh2` 有效而 `Weather` 明显退化，回退到 Step 6，重新设计
  condition，而不是调大 sweep。
- 若只有 `R.3 + auxiliary control` 有效，而独立 CFUS 无效，则说明当前方向退化为 R.3 repair；
  Phase4-S 不能作为主线继续。
