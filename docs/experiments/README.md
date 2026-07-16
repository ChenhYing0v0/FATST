# Active Experiment Protocols

- `stage-c-d14-output-coupling-granularity.md`: D14-A1 three-seed dual-carrier confirmation已通过；5/5 stable
  crossing且strict/instance oracle显著为正；GroupedMLP仍低于A6-LBF，故只授权D14-B返回Step4-6，
  implementation/method/test=false；
- `stage-c-d14-conditional-patch-memory-headroom.md`: 已降级为`D14-P auxiliary_not_scheduled`；仅保留future
  decoder-interface ablation，不占paper slots、不决定mainline；
- `stage-c-d13-rolling-origin-revision-efficiency.md`: `deferred_next_paper`；forecast-revision idea的未来problem
  protocol，保留但当前不执行，入口见根目录`New-idea.md`；
- `stage-c-natural-baseline-test-protocol.md`: 已完成的 post-freeze test reference；
- `stage-c-pmfo-pir-problem-diagnostic.md`: 已完成的 Step 2-3 D1 protocol；
- `stage-c-pmfo-rct-step7-protocol.md`: 已关闭的 PMFO-RCT Step 7 protocol；
- `stage-c-sc1-d2-operator-structure-diagnostic.md`: 已完成并关闭的 D2 formal5 problem diagnostic；
- `stage-c-sc1-d3-crossed-basis-group-diagnostic.md`: 已完成的 Step 2/3 paired 2×2 basis-group diagnostic；
- `stage-c-sc1-d4-structured-basis-mechanism-diagnostic.md`: 已完成并回滚Step 2/3的standard-basis、locality与
  exact-balancing diagnostic；
- `stage-c-sc1-d5-conditioning-locality-frontier-diagnostic.md`: 当前Step 2/3 local conditioning headroom
  diagnostic；只作problem existence，不是method training。
- `stage-c-sc1-d6-horizon-support-interaction-confirmation.md`: 当前Step 2/3 disjoint-validation confirmation；
  检验short/long horizon的support-scale crossing。

完整结果写入 `analysis/`；ledger 只保留 decision、status、next action 与链接。Phase0-Phase5 protocols
及已完成/被取代的 StageC calibration protocols 已移入
`docs/archive/pre-stage-c-reset-20260713/experiments/`。
