# Active Scripts

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
- `run_stage_c_sc1_d2_diagnostic.py`: 从冻结A6 checkpoints提取`memory [B,C,P,D]`，训练rank/full-affine、
  dense nonlinear、true-scale grouped与random grouped head-only probes；不更新forecast model、不读取test。
- `analyze_stage_c_sc1_d2_diagnostic.py`: 计算rank、generic nonlinearity与scale alignment的paired gains，
  区分core3 precheck和formal5 hard gate。
- `remote/run_stage_c_sc1_d2_diagnostic.sh`: 三套ready datasets的D2 core3 parallel precheck；
- `sync_stage_c_sc1_d2_diagnostic_results.sh`: 同步D2 raw metrics/history/metadata并本地重算gate。

历史 runner/analyzer 已移入 `scripts/archive/`，不得作为当前研究入口。下一步新增脚本只能服务
`docs/experiments/stage-c-pmfo-rct-step7-protocol.md`，并须同步 code explanation。
