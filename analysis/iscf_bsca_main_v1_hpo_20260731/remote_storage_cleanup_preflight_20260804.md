# Remote HPO Storage Cleanup Preflight

## Scope

- remote root=`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo`；
- quota before cleanup=`201G / 200G soft / 220G hard`；
- HPO root before cleanup约43G；
- user authorization=`2026-08-04 clean remote result folder before new experiments`。

## Frozen deletion set

1. 七个H1/H2/H3A/H3B/H4J/H4K/H4L `_resource_smoke`目录，约4.24GiB；
2. 157个non-selected `pcsd_test_audit_diagnostics.npz`，约32.26GiB。

## Frozen retention set

- 八个current dataset-level selected trial diagnostics；
- 165份`test_audit_metrics_by_target_horizon.csv`；
- 165份`test_audit_invariants.json`；
- 全部training checkpoints、effective configs、logs、environment、initialization contracts、manifests与local/remote analysis ledgers；
- remote worktree及其三份unrelated dirty CSV。

预计释放约36.5GiB。删除项不可直接恢复，但可由对应smoke/formal evaluation重新生成；正式MSE/MAE与checkpoint provenance保持可复核。
