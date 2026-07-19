# Stage C D20 CST Step 6 Tooling

## 1. 功能边界

本阶段没有修改production model。新增内容只负责冻结并验证`SC-D20-CST`的Step6 diagnostic contract：

- `configs/stage_c_d20_cst_step6.json`：machine-readable设计与authorization；
- `scripts/check_stage_c_d20_cst_step6.py`：static algebra/shape/parameter/init checker；
- `step6_static_gate.json`：checker输出。

checker不加载dataset、不训练模型、不访问validation/test，也不授权remote。

## 2. Config modules

### `carrier`

冻结A6的输入长度720、预测长度720、basis rank 256、five-dataset profile hash，以及真实forward shape：

```text
normalized_history [B,720,C]
memory             [B,C,P,D]
hidden             [B,C,R]
coeff              [B,C,256]
prediction         [B,720,C]
```

### `summary_contract`

定义两个fixed projection buffer：

- `fixed_real_fourier_low32`：32组cos/sin，共64列；
- `fixed_gaussian_qr_s20260719`：Gaussian QR得到的64列random orthogonal control。

两者都接收`normalized_history`，计算`[B,C,720] @ [720,64] -> [B,C,64]`。

### `paired_initialization`

新增head写成`Linear(R+64,256)`，其中history hidden对应的权重与A6相同，summary对应的64列权重zero-init。
这保证三个arms初始function相同，同时summary columns在第一个batch可获得gradient。

### `matrix`与`primary_gates`

冻结15个from-scratch runs、60个official-test cells和两项对称comparison：

- `transfer_spec_vs_a6`；
- `specificity_spec_vs_random`。

## 3. Checker computation flow

### Projection construction

`real_fourier_projection()`逐frequency构造orthonormal cos/sin columns；`random_orthogonal_projection()`使用
fixed CPU generator、float64 reduced QR与sign canonicalization。这样不同机器上的projection identity可审计。

### Projection statistics

- `spectrum_orthogonality_max_abs`：$\|Q_{spec}^TQ_{spec}-I\|_\infty$；
- `random_orthogonality_max_abs`：$\|Q_{random}^TQ_{random}-I\|_\infty$；
- `spectrum_dc_leakage_max_abs`：每列元素和的最大绝对值，检测DC泄漏；
- `cross_subspace_singular_value_max`：$Q_{spec}^TQ_{random}$的最大奇异值，仅描述两个subspace重合程度，
  不作为pass/fail gate。

### Parameter accounting

`decoder_parameter_count()`按每个dataset的$R$计算：

$$
(R+q)K+K+TK+T.
$$

其中$q=0$为A6，$q=64$为SPEC/RANDOM，$K=256,T=720$。新增参数恒为$qK=16,384$；SPEC与RANDOM
必须完全一致。该统计只用于attribution，不参与candidate selection。

### Initialization and projectivity audit

`synthetic_initialization_audit()`执行三组检查：

1. $W_s=0$时，A6、SPEC与RANDOM输出必须相同；
2. 注入nonzero $W_s$后，从full 720 output crop与直接使用`basis[:H]`计算必须相同；
3. 在$W_s=0$处反向传播，summary-weight gradient必须非零，并且nonzero intervention必须改变prediction。

这分离了`function-preserving initialization`、`projectivity`与`path trainability`，避免只验证zero-init恒等式。

## 4. Output fields

`step6_static_gate.json`中的字段含义：

- `checks_passed/checks_total`：通过项数/总项数；
- `overall_pass`：所有static checks的逻辑与；
- `projection_diagnostics`：上述projection statistics；
- `parameter_rows`：逐dataset的$R$、A6/SPEC/RANDOM decoder params与增量；
- `initialization_diagnostics`：初始输出差、active prefix gap、deformation NRMSE与gradient norm；
- `checks`：每个gate的名称、布尔结果与原始details。

## 5. Verification

```bash
conda run -n r2026-fsa python -m py_compile \
  scripts/check_stage_c_d20_cst_step6.py
conda run -n r2026-fsa python \
  scripts/check_stage_c_d20_cst_step6.py \
  --config configs/stage_c_d20_cst_step6.json \
  --output analysis/stage_c_post_ccsf_step24_reset_20260719/d20_step6/step6_static_gate.json
```

当前结果为`14/14 pass`。它只授权Step7A local implementation，不是accuracy或paper-narrative evidence。
