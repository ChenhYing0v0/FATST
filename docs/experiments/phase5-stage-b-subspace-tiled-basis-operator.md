# Phase5 StageB B12 Subspace-Tiled Basis Operator

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `B12-STBO` |
| `current_step` | Step 2/3：problem-existence and feasibility diagnostic completed |
| `problem` | A6-LBF-r256 当前使用 `basis[720,K]` 的 full-trajectory step basis，再按 horizon 做 prefix slicing；这可能不是 multi-horizon 原生的 basis-coeff operator |
| `existence_evidence` | B10/B11 显示 A6 `learned_temporal_basis` 已形成 stage/subspace geometry，且真实 `coeff` 在不同 subspaces 上的投影方向不同；B11-BCF 失败说明 late coefficient perturbation 不足以利用该结构 |
| `idea` | 将 full-720 step basis 重构为 non-overlapping future tiles 内的 shared/local subspace basis，使预测任务从单个 720-step basis projection 变为若干 tile-local basis projections |
| `theory_check` | 若 A6 basis 和 train labels 都能被 shared/local tile basis 或少量 basis banks 高效近似，则 B12 有可能成为原生 multi-horizon operator；若只有 independent per-tile basis 有效，则该方向可能退化为分段 Direct head |
| `design` | Offline diagnostic：basis factorization audit、train-label tile-basis audit、coeff projection audit；不训练新模型 |
| `narrative_gate` | pending; 只有 Step 2/3 诊断支持 shared/bank tile basis 后，才能进入 Step 4-6 method design |
| `effectiveness_gate` | not applicable before implementation |
| `artifacts` | `analysis/phase5_stage_b_b12_stbo_diagnostic_20260708/b12_stbo_report.md`; `docs/code-explanation/phase5-stage-b-b12-stbo-diagnostic.md`; `scripts/analyze_phase5_stage_b_b12_stbo_diagnostic.py` |
| `decision` | `diagnostic_not_enough_for_b12`; do not implement B12-STBO as currently defined |

## Motivation

A6-LBF-r256 的 accepted mechanism 是：

```text
hidden [B,C,R]
  -> coeff [B,C,K]
learned_temporal_basis [720,K]
  -> prediction [B,720,C]
```

该设计把 sample-wise 信息放在 `coeff` 中，把 future position 信息放在 `basis[t]` 中。它支持 unified
training，也支持 prefix evaluation；但严格来说，它仍是先生成一条 full-720 trajectory，再返回 requested
horizon prefix。也就是说，`H=96` 仍依赖一套为 `720` 长度定义的 step basis 参数。

B11 已证明 A6 的 `basis` 不是完全 step-independent：sliding-window basis subspaces 沿未来时间轴连续变化。
但 B11-BCF 只是在 `coeff` 后面加了一个很弱的 field perturbation，最终被 `no_basis` 与
`constant_slot` controls 解释。B12 因此不再修改 `coeff` 的 late perturbation，而是重新审视
`basis-coeff operator` 本身。

## Core Hypothesis

[Hypothesis] 如果 A6 的 basis 已经形成可复用的 future subspaces，那么更自然的 operator 不是
`basis[720,K]`，而是：

```text
future tiles m = 0..M-1
local offset tau = 0..L-1
shared/bank local basis U[tau, r]
tile-specific coefficients z[b,c,m,r]
prediction[b, m*L + tau, c] = U_m[tau] @ z[b,c,m]
```

其中 `L` 是 tile length。优先诊断 `L=48`，因为 benchmark horizons `96/192/336/720` 都能整除
48，短 horizon 只需要启动前 `H/L` 个 tiles，不必构造完整 720-step readout。

## Candidate Operator Families

### B12-A Shared Local Basis

所有 tiles 共享一套 local basis：

```text
U: [L, Rb]
z_m = f_m(hidden): [B,C,Rb]
y_m = U @ z_m
```

这条路线参数效率最高，multi-horizon 叙事最干净。它要求不同 future tiles 在 local-offset dimension 上共享
稳定坐标系。若诊断显示 shared local basis 接近 independent tile PCA / A6 tile basis 的覆盖率，它应成为
首选方法候选。

### B12-B Subspace Basis Bank

维护少量 local basis banks：

```text
U_q: [L, Rb], q = 1..Q
U_m = sum_q pi[m,q] * U_q
y_m = U_m @ z_m
```

这条路线保留 future-stage/subspace 差异，但不退化为 720 个独立 step rows。它比 hard stage token 更弱，
因为 `pi[m,q]` 可以是 learned smooth tile prior 或由 basis geometry 产生的 soft assignment。

### B12-C Independent Tile Basis

每个 tile 独立一套 basis：

```text
U_m: [L, Rb]
y_m = U_m @ z_m
```

这适合作为 diagnostic/control：如果只有 independent tile basis 明显有效，而 shared/bank controls 不行，
则 B12 的主贡献风险很高，因为它更像分段 Direct head，不能充分支撑 unified operator 叙事。

## Why This Is Not Residual

B12 不写成：

```text
y = A6(x) + correction(x)
```

也不写成：

```text
coeff = coeff_base + delta
```

它直接替换 primary prediction operator：

```text
full step basis projection
  -> subspace-tiled local basis projection
```

因此它的论文边界是 operator factorization / native multi-horizon readout，而不是 residual repair。

## Step 2/3 Diagnostic Plan

诊断必须先回答三个问题。

### 1. A6 Basis Factorization Audit

读取 clean A6 checkpoint 的 `learned_temporal_basis: [720,K]`， reshape 为：

```text
basis_tiles: [M,L,K], M=720/L
```

比较以下 reconstruction energy：

| Arm | Meaning |
| --- | --- |
| `shared_local_basis` | 所有 tiles 共享同一 `U: [L,r]`，只允许 tile-specific right coefficients |
| `basis_bank_q` | tiles 分配到 `Q` 个 local basis banks，每个 bank 有自己的 `U_q: [L,r]` |
| `independent_tile_basis` | 每个 tile 独立 SVD basis，作为 upper bound |
| `local_dct` | fixed DCT local basis control，测试是否只是 generic smoothness |

若 `shared_local_basis` 或少量 `basis_bank_q` 接近 `independent_tile_basis`，说明 A6 的 learned step basis
可被更结构化的 tile operator 解释。

### 2. Train-Label Tile-Basis Audit

只使用 train split labels，构造 normalized future matrix：

```text
future_labels: [N*C,720]
label_tiles_m: [N*C,L]
```

比较 shared local PCA、basis bank、independent tile PCA 与 local DCT 的 energy coverage。该步骤避免只解释
checkpoint 参数，而不解释真实 label distribution。

### 3. Coeff Projection Audit

读取 clean A6 forward 的真实 `coeff: [B,C,K]`，将其投影到每个 tile 的 A6 basis row-space。观察：

- adjacent tile projection cosine 是否高于 far tile；
- output energy 是否集中到少数 tiles；
- projection entropy 是否显示 `coeff` 同时服务多个 tile subspaces。

该步骤判断 B12 是否需要 stage/tile-local coefficient generation，而不是只改 basis 参数化。

## Gate

B12 进入 Step 4-6 method design 的必要条件：

1. `shared_local_basis` 或 `basis_bank_q` 在至少两个数据集上接近 `independent_tile_basis`；
2. 上述结果不能完全被 `local_dct` 解释；
3. train labels 的 tile-local shared/bank structure 也成立；
4. coeff projection 显示不同 tiles 的 coefficient usage 具有可分化结构；
5. 诊断报告必须明确 failure attribution：若失败，是 `hypothesis_false`、`generic_basis_control_explains`、
   `independent_tile_only`，还是 `coeff_path_not_supported`。

若仅 `independent_tile_basis` 有效，则不能直接进入 model implementation；只能重新设计一个不会退化为分段
Direct head 的共享约束。

## Step 2/3 Diagnostic Result

诊断已完成，输出目录：

- `analysis/phase5_stage_b_b12_stbo_diagnostic_20260708/`;
- `analysis/phase5_stage_b_b12_stbo_diagnostic_20260708/b12_stbo_report.md`。

默认设置：

- `tile_len=48`;
- `gate_rank=16`;
- A6 checkpoints 来自 `analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707/raw/official-last/TimeAlignOfficialUnified720_a6_clean_official-last/`;
- train-label windows 每数据集最多采样 `4096`;
- coeff rows 每数据集最多采样 `20000`。

### A6 Basis Factorization

| Dataset | Shared | Bank4 | Independent | DCT | Bank4-DCT | Bank4 Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | `0.481` | `0.505` | `0.572` | `0.444` | `0.061` | `0.067` |
| ETTm1 | `0.436` | `0.466` | `0.534` | `0.411` | `0.054` | `0.068` |
| Weather | `0.400` | `0.438` | `0.521` | `0.357` | `0.081` | `0.083` |

[Interpretation] A6 basis side 有弱到中等正证据：`bank4` 均优于 local DCT，但与 independent tile upper
bound 的 gap 仍为 `0.067-0.083`，未达到 Step 4-6 method gate。

### Train-Label Tile Factorization

| Dataset | Shared | Bank4 | Independent | DCT | Bank4-DCT |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | `0.975` | `0.975` | `0.975` | `0.973` | `0.002` |
| ETTm1 | `0.986` | `0.986` | `0.986` | `0.986` | `0.000` |
| Weather | `0.927` | `0.936` | `0.952` | `0.925` | `0.011` |

[Interpretation] label side 的 tile-local structure 很强，但几乎完全被 local DCT control 解释。这说明
“stage 内共享 local basis”本身并不新；必须证明 learned/bank basis 不是 generic smoothness，当前证据不足。

### Coeff Projection

| Dataset | Adjacent Cos | Far Cos | Projection Entropy | Output Entropy |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | `0.284` | `0.124` | `0.951` | `0.960` |
| ETTm1 | `0.030` | `0.164` | `0.954` | `0.952` |
| Weather | `0.035` | `0.047` | `0.881` | `0.895` |

[Interpretation] coeff 的 tile-subspace adjacent/far 结构只在 ETTh2 成立。ETTm1 和 Weather 不支持当前
B12 需要的跨数据集 tile-local coeff path 证据。

## Failure Attribution

- `hypothesis_false`: not proven。B11 sliding-window evidence 和 B12 A6-basis bank-vs-DCT gap 仍说明
  basis side 存在一定结构。
- `generic_basis_control_explains`: yes。Train-label tile structure 在三个数据集上都几乎被 local DCT 解释。
- `independent_tile_only`: not the main failure。Independent tile 更强，但 shared/bank 没有灾难性落后；问题是
  gap 不够小，无法进入 method gate。
- `coeff_path_not_supported`: yes。只有 ETTh2 有明确 adjacent > far coeff projection pattern。
- `direction_level_rejection`: no。该诊断阻断当前 B12-STBO，不否定所有 basis-operator redesign。

## Decision

`B12-STBO` 当前结论为 `diagnostic_not_enough_for_b12`。不得实现当前 shared/bank local basis operator。
若继续该方向，必须先重新定义能压过 local DCT control、且能在 ETTm1/Weather 上支持 coeff-path
分化的更强 basis-operator problem；否则 StageB 应回到 Step 2/3 architecture search。
