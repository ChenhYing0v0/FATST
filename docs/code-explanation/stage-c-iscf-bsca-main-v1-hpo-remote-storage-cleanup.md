# ISCF-BSCA-MAIN-v1 Remote HPO Storage Cleanup

## Functional boundary

`scripts/remote/cleanup_iscf_bsca_main_v1_hpo_storage.sh`只作用于固定路径`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo`。脚本不访问remote Git worktree、dataset目录或其他experiment families。

默认`MODE=audit`只统计七个已知`_resource_smoke`目录与formal-test `pcsd_test_audit_diagnostics.npz`。`MODE=apply`还要求精确confirmation token，并在删除前断言165个diagnostic files、8个selected diagnostics以及165份逐H metrics/invariants完整。

## Retention contract

清理删除七个resource-smoke目录和157个non-selected dense diagnostics。八个当前dataset-level selected trials的dense diagnostics继续保留；全部formal-test `test_audit_metrics_by_target_horizon.csv`、`test_audit_invariants.json`、logs、analysis ledgers、manifests、training artifacts与checkpoints均保留。

删除后脚本强制验证：resource-smoke目录为0、selected diagnostics为8、metrics/invariants仍各165。被删除的resource-smoke checkpoints和dense diagnostic arrays无法直接恢复，只能通过重新执行对应smoke/formal evaluation重建；paper-facing MSE/MAE与checkpoint provenance不依赖这些被删除文件。
