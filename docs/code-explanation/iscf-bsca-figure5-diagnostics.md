# ISCF-BSCA Figure 5 diagnostics

## 功能模块

`scripts/build_iscf_bsca_figure5_diagnostics.py`只读取冻结的validation artifacts，不加载或修改checkpoint，也不访问test loader。输入由`configs/iscf_bsca_figure5_diagnostic_protocol.json`约束。

### Artifact validation

每个role/dataset目录必须包含`effective_config.json`、`initialization_contract.json`、`pcsd_validation_diagnostics.npz`与`trained_invariants.json`。脚本核对dataset、validation split、checkpoint SHA256、invariant pass、required arrays、finite values、scope顺序与future-region顺序。

Full与`Fixed Scope (s=144)`的`probe_targets [256,720]`必须exact equal，避免qualitative comparison发生origin/channel错位。Full的`probe_direct_policy [256,720,5]`还必须满足scope轴sum-to-one tolerance。

### Aggregate statistics

- `policy_row_bin_usage [N,8,5]`：先沿validation series-row轴求mean，得到dataset-level future-region × scope utilization，再对五datasets等权macro average。
- `arm_row_bin_mse/mae [N,8,5]`：使用相同dataset-first reduction；每个region的scope excess MSE定义为相对该region最低macro MSE的percentage increase。
- `trained_invariants.json::prefix_rows`：提取H96/H192/H336/H720相对H720 prefix的maximum absolute CHPD。

这里的`N`是dataset对应的全部sequential validation channel-series rows，不是抽样数量。256-row上限只作用于需要保留完整720-step tensor的qualitative pool。

### Qualitative selection

脚本在5 datasets × 256 rows=`1,280`完整pool中计算Full与Fixed Scope的H720 MSE/MAE。Primary score为Full相对Fixed Scope的MSE reduction percentage，MAE仅作tie-breaker；dataset order与row index提供完全相同时的deterministic tie resolution。被选row的target、两条fused forecasts、五条scope forecasts和Scope Probability全部写入`qualitative_source_data.csv`。

### Figure export

Figure 5使用Python/matplotlib构成五个panels：CHPC、selected-row probability map、aggregate utilization、scope-wise regional excess MSE与selected trajectory。SVG/PDF保留editable text，TIFF为600 dpi LZW，canvas为180 mm × 160 mm。

## 统计列定义

- `max_abs_chpd`：requested-H output与H720 output相同prefix的maximum absolute difference。
- `mean_probability`：指定dataset、future region与scope上的validation-row平均allocation probability。
- `mean_mse/mean_mae`：指定scope独立forecast在region内先按step求误差、再按全部validation rows求mean。
- `utilization_mse_agreement`：highest-utilization scope是否等于lowest-MSE scope。
- `mse_gain_percent/mae_gain_percent`：`100 × (control error - Full error) / control error`。

## Code-theory consistency

预期理论是统一trajectory提供CHPC，多个scope forecasts可能具有不同future-region error profiles，而allocation应在scope轴上融合它们。代码直接读取训练时真实scope arms与probabilities，因此能够验证前两项行为和allocation health。

仍然只是proxy的部分是“allocation是否利用了scope competence”：`utilization_mse_agreement`比较aggregate probability与label-derived best scope，只是descriptive alignment，不是因果utility。它不能替代matched `w/o Target-Adaptive Allocation` ablation，也不能把selected trajectory解释为allocation独立贡献。若probabilities接近均匀或agreement较低，应收窄claim，而不是修改统计或选择子集。
