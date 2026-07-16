# Active Scripts

StageC D14-A1 active output-coupling diagnostic：

- `check_stage_c_d14a1_step7a.py`: 验证80组parameter/partition/affine cases、20组forward/gradient cases与
  A6 encoder initialization pairing；
- `evaluate_stage_c_d14a1_checkpoint.py`: 从best-val checkpoint生成validation row×bin losses、persistence与
  aligned prediction probes；
- `analyze_stage_c_d14a1.py`: 分开应用neutral primary gate与A6 sensitivity gate；A6-negative不能方向拒绝；
- `analyze_stage_c_d14a1_multiseed.py`: 聚合2021/2022/2023，在dataset内要求2/3 seed stable crossing，并应用
  strict oracle、sample-over-bin oracle与dual-carrier confirmation gate；
- `remote/run_stage_c_d14a1_dual_carrier_grouped_mlp.sh`: neutral-first串行runner，只有neutral pass才允许A6；
- `remote/run_stage_c_d14a1_confirmation.sh`: 在seed2021 dual-carrier review后串行运行2022/2023 neutral+A6，
  最后生成multi-seed gate；不自动启动D14-B；
- `sync_stage_c_d14a1_dual_carrier_grouped_mlp_results.sh`: 同步必要configs、metrics、row diagnostics与gate。
- `sync_stage_c_pcsd_cf_step7b_results.sh`: 排除checkpoints后同步PCSD-CF Step7B完整validation diagnostics，
  本地独立重算formal gate与horizon/arm-credit deep dive。
- `analyze_stage_c_pcsd_cf_step7b_deep_dive.py`: 不改变formal gate，展开逐horizon/bin、same-run oracle、
  policy variation及DIRECT arm相对独立fixed E2E training的credit-starvation归因。
- `check_stage_c_sc2_pcc_step5.py`: 用float64 autograd验证PCC gradient identities、dense-prefix measure、
  full-domain projectivity与crossed synthetic router recoverability；只授权Step6 design。
- `remote/run_stage_c_pcsd_cf_test_audit.sh`: 不重训；对PCSD-CF-v1既有60个best-validation checkpoints执行
  official test dense-H1..720 evaluation，前后核对checkpoint hash并生成独立test aggregate。
- `check_stage_c_pcsd_cf_test_audit.py`: 冻结candidate/matrix/test授权/no-retraining/hash规则，并验证prefix
  cumulative MSE/MAE与直接计算等价；21/21才允许remote test audit。
- `analyze_stage_c_pcsd_cf_test_audit.py`: 对齐冻结validation与official test aggregate，输出逐reference/dataset
  gain reversal、60-run invariant accounting与`test_fail_with/without_arm_headroom`决策。

StageC D14-A0 archived evidence entrypoints：

- `run_stage_c_d14a_output_coupling_granularity.py`: 在neutral PCA64 carrier上拟合parameter-matched
  blockwise reduced-rank output operators，保存三fold calibration/validation metrics与row-level bin losses；
- `analyze_stage_c_d14a_output_coupling_granularity.py`: 计算carrier skill、stable crossing、canonical oracle
  headroom与contiguity controls，并应用预注册D14-A0 gates；
- `remote/run_stage_c_d14a_output_coupling_granularity.sh`: 在3090服务器按dataset-major顺序运行五数据集；
- `sync_stage_c_d14a_output_coupling_granularity_results.sh`: 仅同步D14-A0必要metrics、metadata、logs与压缩loss artifacts。

StageC active entrypoints：

- `evaluate_stage_c_natural_baseline_test.py`: 从冻结 checkpoint 计算 dense-horizon test metrics；
- `analyze_stage_c_natural_baseline_test.py`: 汇总逐 seed结果并审计 completeness/test leakage；
- `sync_stage_c_natural_baseline_test_results.sh`: 同步并分析远端 baseline reference；
- `remote/run_stage_c_natural_baseline_test.sh`: 3-dataset baseline evaluator；
- `remote/check_529lab_3090_gpus.sh`: 远端 GPU preflight；
- `check_project_structure.py`: 最小仓库结构检查。
- `run_stage_c_d1_offline_diagnostic.py`: D1-A/B/C evaluation-space、frozen-counterfactual、no-forecast-training worker；
- `analyze_stage_c_d1_offline_diagnostic.py`: D1 cross-dataset gate与报告；
- `remote/run_stage_c_d1_offline_diagnostic.sh`: 3-dataset parallel runner；
- `sync_stage_c_d1_offline_diagnostic_results.sh`: 同步并重新审计D1 artifacts。
- `check_stage_c_pmfo_rct_step7a.py`: PMFO-RCT四variants的shape、prefix、refinement、conservation、
  locality、horizon-path与parameter/FLOP local gate。
- `check_stage_c_step7b_local.py`: dense cumulative metrics与full-crop evaluation等价性检查；
- `check_stage_c_step7b_checkpoint_invariants.py`: trained checkpoint prefix/refinement/locality审计；
- `analyze_stage_c_step7b_pmfo_rct.py`: 三dataset、五arms dense-horizon gate与failure attribution；
- `remote/run_stage_c_step7b_pmfo_rct.sh`: 3090固定GPU-worker 15-run matrix；
- `sync_stage_c_step7b_pmfo_rct_results.sh`: 轻量同步并重算Step7B gate。
- `analyze_stage_c_step4_operator_geometry.py`: 读取Step7B A6/PMFO checkpoints，审计effective operator
  rank、function-family dimension、fixed partition boundaries与history-to-root interface；不训练模型。
- `check_stage_c_fpmo_step5_theory.py`: 构造任意长度interval transform，验证FPMO orthogonality、exact A6
  embedding、native prefix restriction、scale factorization与T720 parameter/function-space budget；不训练模型。
- `check_stage_c_plgo_step5_theory.py`: 构造Restricted-Global Nested Basis，验证global DCT subspace、
  interval-local complements、arbitrary-prefix restriction、A6 exact morphism，并审计overcomplete frame与
  independent-group full-affine no-go；CPU-only，不训练模型。
- `check_stage_c_plgo_step6_design.py`: 验证Projective Atom Functional的atomwise subset invariance、prefix
  synthesis、rank/parameter boundary与candidate-control matrix；只作Step6 design audit，不训练模型。
- `check_stage_c_japo_step6_design.py`: 冻结JAPO的双full-rank expert bank、history-atom joint router、
  matched same-bank controls、初始化与分阶段effectiveness gate；只验证Step6 tensor/gradient/contract，
  不实现forecast model、不训练模型。
- `check_stage_c_sc1_japo_step7a.py`: 五profiles × 七arms的210 prefix、35 gradient、paired initialization、
  patch-block rewrite与horizon-path production gate；不训练forecast model。
- `check_stage_c_sc1_japo_checkpoint_invariants.py`: synthetic或trained A6/JAPO checkpoint的prefix、
  from-scratch、initialization、patch与expert contract审计。
- `analyze_stage_c_sc1_japo_e2e.py`: validation-only completeness、paired hashes、same-bank controls、
  seed2021 staged gate与冻结two-seed mean gate分析。
- `remote/run_stage_c_sc1_japo_e2e.sh`: 3090 workload-aware JAPO单seed 35-run end-to-end screen；seed2022完成后
  自动执行seed2021/2022 frozen mean gate；禁止test。
- `run_stage_c_sc1_d7_descriptor_sufficiency.py`: 从冻结A6 memory训练free-M0与GEO/PERM/RANDOM
  compact/matched PAF heads；使用validation batches16-23，不更新forecast model、不读取test。
- `analyze_stage_c_sc1_d7_descriptor_sufficiency.py`: 审计105-fit completeness、parameter/projectivity invariants、
  descriptor gain、free-M0 gap与fit-holdout memorization hard gates。
- `remote/run_stage_c_sc1_d7_descriptor_sufficiency.sh`: 三GPU workload-aware D7 runner；
- `sync_stage_c_sc1_d7_descriptor_sufficiency_results.sh`: 同步D7 raw artifacts并本地独立重算decision。
- `check_stage_c_sc1_d8_step7a.py`: 五profiles × 七arms的shape、prefix、gradient、RGNB descriptor与
  patch-block exact rewrite本地gate；不训练forecast model。
- `check_stage_c_sc1_d8_checkpoint_invariants.py`: 重载trained A6/PAF checkpoint，审计from-scratch、
  projectivity、无冻结参数与patch artifact数值合同。
- `analyze_stage_c_sc1_d8_e2e.py`: 五dataset × 七arms validation-only dense-horizon gate、matched descriptor
  attribution与optimization/pathology审计。
- `remote/run_stage_c_sc1_d8_e2e.sh`: 3090 workload-aware 35-run end-to-end joint-training screen；禁止test。
- `run_stage_c_sc1_d2_diagnostic.py`: 从冻结A6 checkpoints提取`memory [B,C,P,D]`，训练rank/full-affine、
  dense nonlinear、true-scale grouped与random grouped head-only probes；不更新forecast model、不读取test。
- `analyze_stage_c_sc1_d2_diagnostic.py`: 计算rank、generic nonlinearity与scale alignment的paired gains，
  区分core3 precheck和formal5 hard gate。
- `remote/run_stage_c_sc1_d2_diagnostic.sh`: 三套ready datasets的D2 core3 parallel precheck；
- `sync_stage_c_sc1_d2_diagnostic_results.sh`: 同步D2 raw metrics/history/metadata并本地重算gate。
- `analyze_stage_c_five_profile_extension.py`: ETTh1/ETTm2三阶段validation-only natural profile选择与
  stability审计；parameter count不参与选择。
- `remote/run_stage_c_five_profile_extension.sh`: 14-run profile extension matrix，按A/B/C阶段顺序执行并可续跑；
- `sync_stage_c_five_profile_extension_results.sh`: 轻量同步extension artifacts并本地独立重算三阶段结论。
- `remote/run_stage_c_sc1_d2_formal5.sh`: workload-aware五dataset × 三checkpoint seeds × 11 arms hard gate；
- `sync_stage_c_sc1_d2_formal5_results.sh`: 同步formal5 artifacts并本地独立重算最终problem decision。
- `run_stage_c_sc1_d3_crossed_diagnostic.py`: 复用D2 frozen-memory contract，只训练
  `random basis × random group`缺失cell的45个head-only fits。
- `analyze_stage_c_sc1_d3_crossed_diagnostic.py`: 合并D2/D3四cell，先聚合structure seeds，再以15个
  dataset-checkpoint units审计basis main effect、conditional effects与interaction。
- `remote/run_stage_c_sc1_d3_crossed_diagnostic.sh`: 五dataset workload-aware D3 runner；
- `sync_stage_c_sc1_d3_crossed_results.sh`: 同步D3 raw artifacts并结合既有D2 raw独立重算gate。
- `run_stage_c_sc1_d4_structured_basis.py`: seven-basis × random-group frozen-memory worker，记录八个prefix
  horizons与fit-target geometry；
- `analyze_stage_c_sc1_d4_structured_basis.py`: standard-basis noninferiority、locality、exact-balancing与D3
  replication gates；
- `remote/run_stage_c_sc1_d4_structured_basis.sh`: workload-aware 315-fit remote diagnostic；
- `sync_stage_c_sc1_d4_structured_basis_results.sh`: 同步raw artifacts并本地重算D4 decision。
- `analyze_stage_c_sc1_d9_history_support_operator.py`: 从15个frozen natural A6 checkpoints精确合成
  `W = learned_temporal_basis @ learned_basis_coeff.weight`，审计history patch scale与future RGNB support
  coupling；不读取data split、不训练head。
- `remote/run_stage_c_sc1_d9_history_support_operator.sh`: 在3090服务器执行CPU-only D9-A audit，并在launch
  record中保留GPU preflight、commit与protocol boundary。
- `sync_stage_c_sc1_d9_history_support_operator_results.sh`: 只同步D9-A CSV/JSON/report等轻量结果，不同步原始
  baseline checkpoints。
- `run_stage_c_sc1_d10_raw_scale_identifiability.py`: 从official train/validation raw windows构造history DCT ×
  future RGNB capacity-matched ridge probes；输出七组matrix与binary 2×2 cells，不读取test。
- `analyze_stage_c_sc1_d10_raw_scale_identifiability.py`: 聚合binary/detail-monotone metrics、paired controls、
  6! mapping permutations与frozen gates。
- `remote/run_stage_c_sc1_d10_raw_scale_identifiability.sh`: 三GPU workload-aware五dataset D10 runner；
- `sync_stage_c_sc1_d10_raw_scale_identifiability_results.sh`: 同步D10 raw cells、metadata与aggregate decision。
- `run_stage_c_sc1_d11_future_component_responsibility.py`：在五数据集A6 checkpoints上提取short/long
  prefix risk的total gradients与exact future-component responsibilities；
- `analyze_stage_c_sc1_d11_future_component_responsibility.py`：聚合seed/split/basis controls并应用strict
  directional、support-specific、generic与magnitude-only gates；
- `remote/run_stage_c_sc1_d11_future_component_responsibility.sh`：3090三GPU dataset-major runner；
- `sync_stage_c_sc1_d11_future_component_responsibility_results.sh`：同步raw/aggregate evidence。

历史 runner/analyzer 已移入 `scripts/archive/`，不得作为当前研究入口。新增脚本必须服务active ledger中明确的
next action，并同步对应experiment protocol与code explanation。
