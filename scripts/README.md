# Active Scripts

StageC active entrypoints：

- `evaluate_stage_c_natural_baseline_test.py`: 从冻结 checkpoint 计算 dense-horizon test metrics；
- `analyze_stage_c_natural_baseline_test.py`: 汇总逐 seed结果并审计 completeness/test leakage；
- `sync_stage_c_natural_baseline_test_results.sh`: 同步并分析远端 baseline reference；
- `remote/run_stage_c_natural_baseline_test.sh`: 3-dataset baseline evaluator；
- `remote/check_529lab_3090_gpus.sh`: 远端 GPU preflight；
- `check_project_structure.py`: 最小仓库结构检查。

历史 runner/analyzer 已移入 `scripts/archive/`，不得作为当前研究入口。下一步新增脚本只能服务
`docs/experiments/stage-c-pmfo-pir-problem-diagnostic.md`，并须同步 code explanation。
