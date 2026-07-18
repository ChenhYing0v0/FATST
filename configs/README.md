# Active Configs

- `stage_c_mechanism_control_natural_dataset_profiles.json`: 后续所有 mechanism/control 的冻结 contract；
- `stage_c_dataset_profile_calibration_r2.json`: contract 引用的 validation-only profile provenance。
- `stage_c_post_pcc_step6.json`: SIFF-v1/MCCA-v1 Step6冻结tensor、same-mass assignment、controls、gates与rollback。
- `stage_c_siff_equal_attribution_v2.json`: `SIFF_EQUAL` 的10-arm EQUAL-context归因矩阵，冻结四层评估、
  seven hard comparisons、Phase-A/confirmation规模及Step7A-only授权。
- `paper_facing_evaluation_protocol.json`: paper-facing four-horizon scorecard、checkpoint selector、test-informed
  边界与统一四层机制评估规则。

contract hash 为 `254d85d47a9e5b7c212f8a8b88decf17a0328a1ea1df324c9cc65be4c672a50c`。
不得根据 test 或新 mechanism 重新选择 dataset profile。旧 configs 位于 `configs/archive/`。
