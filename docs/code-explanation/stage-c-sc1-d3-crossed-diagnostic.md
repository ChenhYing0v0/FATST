# SC1-D3 Crossed Diagnostic Code Explanation

## 1. Functional Boundary

SC1-D3没有修改model code。它复用D2的frozen-memory extraction、sample-ID split、feature normalization、
training loop与validation evaluator，只新增缺失的`random basis × random group` head和2×2 analyzer。

## 2. Worker Tensor Flow

入口：`scripts/run_stage_c_sc1_d3_crossed_diagnostic.py`。

1. `load_model_and_loaders`加载frozen A6 checkpoint，得到Encoder memory
   `memory [B,C,P,D]`；
2. `collect_rows`变换为`features [B*C,P*D]`与normalized full-future
   `target [B*C,720]`；
3. `split_fit_holdout`按sample ID切分，避免同一样本的不同channel跨fit/holdout；
4. `random_orthogonal_basis(720, seed)`产生`basis [720,720]`；
5. `random_groups`产生11个coefficient index tensors，sizes固定为
   `[1,1,2,4,8,16,32,64,128,256,208]`；
6. `GroupedNonlinearHead`的每个block把`features [N,PD]`映射为`coeff_group [N,n_l]`，
   scatter回`coefficients [N,720]`；
7. training在orthogonal coefficient space计算evaluation-weighted MSE；validation通过
   `prediction [N,720] @ basis [720,720]`还原time-domain future；
8. 每dataset输出9 rows：3 checkpoints × 3 paired structure seeds。

### Worker artifacts

- `d3_probe_metrics.csv`：每fit的fit/holdout/validation MSE、MAE、params与seed；
- `d3_training_history.csv`：每epoch的fit和inner-holdout evaluation-space MSE；
- `d3_metadata.json`：checkpoint/profile/tensor width/split rows、contract hashes、orthogonality、
  environment与test/freeze flags。

## 3. Analyzer Artifact Flow

入口：`scripts/analyze_stage_c_sc1_d3_crossed_diagnostic.py`。

Analyzer从D2读取`TT/TR/RT`，从D3读取`RR`。每个factorial block输出：

- `tt_mse/tr_mse/rt_mse/rr_mse`及对应MAE：四个原始cell error；
- `mse_basis_true_group_log_effect = log(RT/TT)`；
- `mse_basis_random_group_log_effect = log(RR/TR)`；
- `mse_basis_main_log_effect`：上面两项的平均；
- `mse_interaction_log_effect`：上面两项之差；
- `*_reduction = 1-exp(-log_effect)`：正值表示true factor降低error。

聚合顺序固定为：

```text
45 factorial blocks
  -> average 3 structure-seed log effects within dataset/checkpoint
15 checkpoint primary units
  -> summarize 3 checkpoints within each dataset
5 dataset summaries
  -> preregistered macro + consistency gates
```

输出：

- `d3_factorial_blocks.csv`：45个block；
- `d3_checkpoint_effects.csv`：15个primary units；
- `d3_dataset_summary.csv`：5个dataset summaries；
- `d3_summary.json`：machine-readable gates与decision；
- `d3_diagnostic_report.md`：human-readable report。

## 4. Remote And Sync Flow

`scripts/remote/run_stage_c_sc1_d3_crossed_diagnostic.sh`在launch前验证D2 artifacts并执行`nvidia-smi`。
workload-aware顺序为：GPU0 `Weather -> ETTh1`，GPU1 `ETTm1 -> ETTh2`，GPU2 `ETTm2`。
remote output固定在`/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d3_crossed`。

`scripts/sync_stage_c_sc1_d3_crossed_results.sh`把remote raw artifacts同步到
`analysis/stage_c_sc1_d3_crossed_20260714/raw/`，再用local analyzer独立重算所有gates。

## 5. Code-Theory Consistency

| Intended theory | Code realization | Remaining proxy | Falsification |
| --- | --- | --- | --- |
| 补齐2×2缺失cell | same random basis + same random groups进入同一head | 仅diagonal seed pairing | random-group conditional不通过 |
| 隔离basis main effect | log-error conditional contrasts取平均 | probe family内的estimand | main effect或consistency gate失败 |
| 防pseudo-replication | structure seeds先聚合为15 checkpoint units | checkpoint seeds仍共享dataset | dataset consistency失败 |
| 与D2公平匹配 | 直接复用D2 extraction/train/eval functions | upstream D2代码变更风险 | config/invariant/hash gate失败 |

D3即便通过也不验证paper novelty或end-to-end effectiveness；这两项分别属于后续Step 4-6与Step 9-10。
