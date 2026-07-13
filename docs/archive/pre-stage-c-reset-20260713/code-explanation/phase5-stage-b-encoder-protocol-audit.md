# Phase5 StageB Encoder And Protocol Audit Code Explanation

## Purpose And Boundary

`scripts/analyze_phase5_stage_b_encoder_protocol_audit.py`只读取现有 clean A6 artifacts，回答三个
diagnostic问题：

1. dataset-specific `patch_num/d_model/d_ff/dropout` 实际形成什么 history state与参数规模；
2. clean A6 的 last epoch相对 validation-best出现多大 validation drift；
3. frozen checkpoint中 token-wise residual MLP是否仍对预测有实际贡献。

它不训练新模型、不改变 checkpoint、不测试 `patch_num>1` 的 causal effect，也不能把 frozen bypass结果
解释为 retraining后的 architecture ablation。

## Inputs

| Input | Meaning |
| --- | --- |
| B9 small-gate `a6_clean` seed-2021 checkpoints | 与当前 clean A6相同的 `learned-basis-forecast-operator` carrier，用于 frozen forward audit |
| corresponding `effective_config.json` | effective official-720 dataset presets与 adapter settings |
| clean A6 rerun `training_log.csv` | best/last validation trajectory |
| local ETTh2/ETTm1/Weather test splits | 前 64 batches的 deterministic frozen ablation |

默认 checkpoint root：

```text
analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707/raw/official-last/
TimeAlignOfficialUnified720_a6_clean_official-last/
```

默认 training root：

```text
analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706/
```

## Dataflow

### Preset And Parameter Audit

脚本从每个 `effective_config.json` 读取：

- `patch_num`；
- `patch_len = seq_len // patch_num`；
- `d_model`、`d_ff`、`dropout`；
- `readout_dim = patch_num * d_model`。

`dropout_variance_factor` 定义为 inverted dropout mask对给定 activation的相对附加方差因子：

$$
\frac{p}{1-p}.
$$

`expected_kept_d_ff_units = d_ff(1-p)` 只表示一次 dropout mask下期望保留的 FFN units，不等于
effective rank或模型容量。

`active_forward_parameters` 只统计 clean A6实际调用的：

```text
patch_emb_x
encoder
norm_x
learned_basis_coeff
learned_temporal_basis
learned_temporal_bias
```

`unused_proj_x_parameters` 单列统计实例化但 A6 forward不调用的 official dense head。两者必须分开，避免
用 state-dict总参数量描述实际 A6 operator capacity。

### Checkpoint Drift

对每个 clean rerun training log：

```text
best_epoch = argmin_epoch(val_mean_mse)
last_minus_best_val_mse_pct
    = (last_val_mean_mse / best_val_mean_mse - 1) * 100
```

该统计只测 validation trajectory，不能推断 best-val test一定优于或劣于 last checkpoint。

### Frozen Residual-Branch Audit

full path：

```text
token -> token + residual_mlp(token) -> optional LayerNorm
```

两个 frozen controls：

- `no_mlp_keep_norm`：把每层 residual MLP替换为 zero update，保留已训练 LayerNorm；
- `embed_only`：同时绕过 residual MLP与 LayerNorm，仅保留 PatchEmbed到 A6 readout。

`branch_to_input_norm` 为 eval mode下每层：

$$
\mathbb{E}\left[\frac{\|f_\ell(x)\|_2}{\|x\|_2+10^{-12}}\right].
$$

`branch_input_cosine` 为 residual update与输入 token的 mean cosine similarity。

ablation MSE使用 test split前 64 batches：

$$
\Delta_{\mathrm{MSE}}
=\left(\frac{\mathrm{MSE}_{variant}}{\mathrm{MSE}_{full}}-1\right)100\%.
$$

## Outputs

| Output | Definition |
| --- | --- |
| `encoder_preset_audit.csv` | preset、dropout noise proxy、active/unused parameters |
| `checkpoint_drift.csv` | clean A6 best/last validation drift |
| `encoder_branch_statistics.csv` | eval-mode residual branch norm/cosine |
| `encoder_branch_ablation.csv` | full/no-MLP/embed-only frozen MSE/MAE与相对变化 |

## Code-Theory Consistency Evaluation

[Intended Theory] 若 ETTm1 `dropout=0.9` 已使 residual MLP完全失效，则 frozen bypass应接近 full
checkpoint；此时 A6更接近单一 global linear projection。

[Observed] `no_mlp_keep_norm` 在 ETTm1 H96/H192/H336/H720 上分别使 MSE增加
`12.96%/9.32%/7.29%/5.58%`。因此 residual MLP不是空路径。ETTm1 branch norm虽低于
ETTh2/Weather，但其预测贡献 material。

[Boundary] 该结果不证明 `dropout=0.9` 最优，也不证明 `patch_num=1` 合理。frozen bypass让 downstream
weights接收到未训练分布，effect size只能作为 branch-use evidence，不能替代 retrained causal ablation。

[Falsifier] 真正验证 ETTm1 tokenization需要
`docs/experiments/phase5-stage-b-ettm1-carrier-protocol-audit.md` 中的 retrained、state/capacity/dropout/
checkpoint-controlled matrix。

## Reproduction

```bash
conda run --no-capture-output -n r2026-fsa \
  python scripts/analyze_phase5_stage_b_encoder_protocol_audit.py
```
