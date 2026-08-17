# Figure 5 mechanism diagnostics：design and prelaunch gate

## 1. 当前结论与执行边界

Section 5.6 是当前正文唯一尚未闭合的实验模块。本轮只复用 exact `ISCF-BSCA-v1` Full 与 Core-Ablation `Fixed Scope (s=144)` 的冻结 validation artifacts，不训练、不访问 official test，也不修改 checkpoint。

现有 `pcsd_validation_diagnostics.npz` 已包含真实 learned `Scope Probability`、五个 scope forecasts、完整 validation-row regional errors，以及 256 个 sequential probe rows。此前的 architecture/allocation schematic 不含 learned values，不能作为本轮证据。

## 2. Figure contract

- Core conclusion：冻结 unified model 在数值上满足 CHPC，并形成可测量的 region-dependent scope usage 与 scope-wise regional error differences；qualitative trajectory 只展示一个披露选择规则的有利案例。
- Archetype：`asymmetric mixed-modality quantitative figure`。
- Backend：Python/matplotlib。
- Final size：180 mm × 160 mm。
- Panel a：五数据集的 numerical CHPC verification。
- Panel b：selected validation row 的 future-step × scope probability map。
- Panel c：五数据集等权 macro aggregate scope utilization。
- Panel d：各 scope 在各 future region 相对 region-best scope 的 excess MSE。
- Panel e：Full 与 `Fixed Scope (s=144)` 的 performance-selected trajectory，并标出 H96/H192/H336/H720 nested boundaries。

## 3. Frozen matrix

- datasets：`Weather, ETTm1, ETTh1, ETTh2, ETTm2`；
- seed：2021；
- scopes：`1, 48, 144, 360, 720`；
- Full checkpoint objects：5；
- Fixed Scope checkpoint objects：5；
- aggregate statistics：使用每个 artifact 的全部 validation series rows；
- qualitative pool：每数据集前 256 个 sequential channel-series rows，共 1280 rows；
- new training：0；
- formal test：0。

## 4. Frozen statistics

1. CHPC：读取 `trained_invariants.json::prefix_rows`，比较每个请求 horizon 与同一次 H720 trajectory 的相同 prefix，阈值为 `2e-5`。
2. Scope Probability map：直接使用 selected row 的 `probe_direct_policy [720,5]`，不平滑、不重新归一化。
3. Aggregate utilization：先在每个 dataset 内对全部 validation rows 求平均，再对五个 datasets 等权 macro average。
4. Regional error：对 `arm_row_bin_mse/mae` 使用同样的 dataset-first macro reduction；Figure 5 展示每个 scope 相对该 region 最优 scope 的 excess MSE percentage。
5. Qualitative selection：在完整 1280-row pool 中，按 Full 相对 Fixed Scope 的 H720 MSE reduction percentage 选择最大者；MAE reduction percentage 只作完全相同 MSE score 时的 tie-breaker。

Qualitative selection 是刻意的 performance selection，caption 必须披露 comparator、validation split 与 selection rule；不得称为 representative、typical 或 prevalence evidence。

## 5. Claim boundary

本轮可以支持 exact prefix consistency、descriptive region-dependent scope usage 与 scope-wise error heterogeneity。按当时冻结边界，它不能补救2026-08-14版`w/o Target-Adaptive Allocation` matched ablation的失败。2026-08-17作者复跑汇总已独立修正该aggregate ablation方向，但本Figure 5仍不得用于声称reliable region-best routing、causal specialization或universal specialization。`realized allocation value` 不进入当前 contract。

## 6. Prelaunch audit

- 远端 5 个 Full 与 5 个 Fixed Scope run directories 均存在；
- 每个 run 均存在 `checkpoint.pt`、`effective_config.json`、`initialization_contract.json`、`pcsd_validation_diagnostics.npz` 与 `trained_invariants.json`；
- ETTm1 spot-check 显示 Full/Fixed 均具有相同的 75,607 validation series rows、8 regions、5 scopes 与256 probe rows；
- Full artifact 含 `probe_direct_policy [256,720,5]`；
- Full 与 Fixed artifacts 均为 validation-only，`uses_test_split=false`；
- 三张 RTX 3090 在 2026-08-16 17:13 CST 均为 18 MiB idle，但本轮 artifact reuse 不需要 GPU execution。

Decision：`prelaunch_pass_reuse_existing_validation_artifacts_no_training_no_formal_test`。
