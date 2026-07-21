# Stage C SIFF-v2 FCC-v1 工具说明

## 1. 目的与边界

本次更新不修改model forward path。它冻结并执行immutable
`SC1-SIFF-v2-EQ-ATTR-v1`的three-seed Final Claim Confirmation（FCC）。用户指定
`A6_FULL`替代`A6_MEASURE`后，FCC只包含三臂：

1. `siff_equal`：待确认的完整SIFF-v2 method package；
2. `a6_full`：用户指定的完整method-package baseline；
3. `siff_independent_equal`：same-objective、capacity-matched的ordered-field
   attribution control。

`SIFF_EQUAL vs A6_FULL`同时改变readout architecture与training objective，只能回答完整
method package是否稳定提升。ordered-field机制归因只由
`SIFF_EQUAL vs SIFF_INDEPENDENT_EQUAL`承担。`A6_MEASURE`不进入config arms、launch matrix、
comparison或machine gate；其历史negative evidence继续保留在研究记录中。

## 2. Frozen config

`configs/stage_c_siff_v2_fcc_v1.json`是唯一machine-readable contract：

- new matrix为`3 arms × 5 datasets × seeds2022/2023 = 30 runs`；
- 复用相同三臂的seed2021共15个historical runs，形成45 runs与180个official-test cells；
- checkpoint只由validation `{96,192,336,720}` mean MSE选择；
- new runs全部from-scratch joint encoder-decoder training；
- two primary comparisons共享MSE `+0.3%`、MAE正向、3/5 datasets、3/4 horizons与
  2/3 seeds gates；
- remote training已授权，formal test只允许在30/30 training完整后执行一次。

independent control的dataset-specific rank为
ETTh1/ETTh2/ETTm1/ETTm2/Weather=`109/116/116/106/116`，用途是近似匹配SIFF active
parameter budget，而不是构造新method。

## 3. Remote runner

`scripts/remote/run_stage_c_siff_v2_fcc_v1.sh`包含四种互斥执行角色：

- `DRY_RUN=1`：展开30个job并调用analyzer synthetic smoke；
- `STATUS_ONLY=1`：统计training/test artifacts完整度并读取各job log最后一行；
- `RESOURCE_SMOKE=1`：并行执行Weather-SIFF seed2022和ETTm2-independent seed2023的
  two-batch smoke，检查training artifacts、非有限值、traceback与OOM；
- `FORMAL_TEST_ONLY=1`：只有30/30 training artifacts完整后才读取test split；每个checkpoint在
  evaluation前后计算SHA256，hash变化立即失败。

普通training mode只在validation上完成最终evaluation，并按config launch order由GPU workers间隔取job。
runner记录commit、config/profile hashes、GPU状态、checkpoint selector、test-informed状态与
`a6_measure_in_fcc=false`。formal test完成后自动调用three-seed analyzer，避免只报告new seeds。

## 4. Three-seed analyzer

`scripts/analyze_stage_c_siff_v2_fcc.py`从seed2021 historical root和seed2022/2023 new root读取：

- `test_audit_metrics_by_target_horizon.csv`：每个standard horizon的MSE/MAE；
- `test_audit_invariants.json`：test authorization、split、checkpoint hash与prefix invariant；
- `effective_config.json`：dataset、seed、readout、objective、selector和training protocol；
- `initialization_contract.json`：encoder initialization hash；
- `checkpoint.pt`：checkpoint uniqueness与test nonmutation provenance；
- `pcsd_test_audit_diagnostics.npz`：arm、fusion、policy与scale-component internal health。

### 4.1 输出统计量

`test_metrics_standard_horizons.csv`中的一行是一个
`seed × dataset × arm × horizon` official-test cell。

`comparison_cells.csv`定义：

$$
\mathrm{gain\_percent}=100\left(1-\frac{\mathrm{candidate}}{\mathrm{reference}}\right),
$$

正值表示candidate error更低。`candidate_value`与`reference_value`是相应MSE或MAE原值。

`comparison_summary.csv`定义：

- `macro_gain_percent`：所有`seed × dataset × horizon` cell gain的等权平均；
- `cell_wins`：cell gain严格大于0的数量；
- `dataset_wins`：先在每个dataset内跨seed/horizon平均，再统计正向dataset数；
- `horizon_wins`：先在每个horizon内跨seed/dataset平均，再统计正向horizon数；
- `seed_wins`：先在每个seed内跨dataset/horizon平均，再统计正向seed数；
- `seed20xx_gain_percent`：对应seed的macro gain。

`run_audit.csv`定义每个run的source、artifact completeness、protocol pass、checkpoint SHA256、
encoder initialization hash、prefix gap与run path。45个checkpoint必须unique，同dataset同seed的三臂
encoder initialization hash必须一致。

### 4.2 Internal health

`mechanism_health.csv`中的每行对应一个SIFF `dataset × seed`：

- `all_finite`：所需diagnostic arrays全部finite；
- `oracle_gain_percent`：逐bin选择best arm相对fused forecast的潜在MSE gain，仅诊断headroom；
- `pairwise_arm_nrmse`：arm predictions两两RMSE，以fused forecast RMS归一化；
- `policy_normalized_entropy`：policy usage entropy除以最大entropy；
- `nonconstant_component_rms_ratio`：非constant scale component RMS相对fused forecast RMS。

这些量只验证内部路径是否活跃，不能覆盖negative official-test effectiveness。

### 4.3 Decision map与failure attribution

- 两项comparison与health全通过：`passed_core_candidate_pending_modern_baselines`；
- 只通过A6_FULL comparison：
  `performance_pass_attribution_blocked_stop_fcc_promotion`，归因为
  `capacity_control_explains`；
- 未通过A6_FULL comparison：
  `siff_v2_final_claim_not_confirmed_stop_paper_core_rescue`，归因为
  `hypothesis_false_or_seed_instability`；
- effectiveness通过但health失败：`design_fault_suspected_no_promotion`。

## 5. Prelaunch checker与生成物

`scripts/check_stage_c_siff_v2_fcc_prelaunch.py`执行25项检查：exact arms/seeds/datasets、30-job
完整性、matrix size、comparison IDs、未放宽margin、用户comparator change、remote/test authorization、
15个historical references的protocol/hash/init pairing、runner dry-run与analyzer smoke。它生成：

- `prelaunch_gate.json`：逐项boolean gate；
- `jobs.csv`：30个new jobs及其profile/rank/readout/objective；
- `historical_reference_audit.csv`：15个seed2021 references的审计快照。

## 6. Code-theory consistency

[Intended Theory] three-seed evidence应分别回答完整SIFF package的稳定effectiveness和ordered field在
same-objective matched control下的必要性。

[Code Realization] config把两个问题拆成两个comparison；runner保证from-scratch matched execution、validation
checkpoint selection与single formal-test boundary；analyzer在完整45-run matrix上执行相同gates并验证
initialization/checkpoint provenance。

[Proxy] independent arm的rank matching只近似active parameter capacity，不保证function class完全等价；internal
health只说明计算路径非退化，不说明它产生有效forecast gain。

[Falsifiers] A6_FULL comparison未通过则否定当前SIFF package的three-seed performance确认；independent
comparison未通过则ordered-field claim仍被capacity/control解释；protocol、hash、initialization或health异常只允许
标记design/protocol fault，不能把不完整audit晋升为paper-core evidence。
