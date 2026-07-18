# CCSF tau0.25 formal Phase-A Step7B prelaunch

## 1. What we plan to test

本轮冻结候选为`SC1-SIFF-v2-CCSF-v1-tau25`。它不复用temperature pilot checkpoint，而是对10个arms在
Weather、ETTm1、ETTm2、ETTh1、ETTh2上以seed2021重新端到端训练，共50 runs。每个run用validation
H96/H192/H336/H720 mean MSE选择checkpoint；冻结checkpoint只在随后读取official test，产生200个标准
dataset-horizon cells。

当前步骤是11-step loop的formal Phase-A Step7B prelaunch。effectiveness gate尚未开始；confirmation seeds
2022/2023保持未授权。

## 2. Why this matrix matters

论文主张需要同时区分两条机制：

1. CCSF contrast path是否优于v1、same-capacity zero contrast、permuted contrast和independent scopes；
2. RELCAL是否在CCSF上优于EQUAL、旧SIFF loss-only与standardized teacher，并形成非冗余interaction。

因此只比较full CCSF与A6不足以归因。Step6冻结的10项hard comparisons全部保留；A6_MEASURE是主要
effectiveness baseline，pilot validation margin不参与机制判定。

## 3. Construction and artifacts

- config：`configs/stage_c_siff_ccsf_v1_tau25_formal_candidate.json`；
- runner：`scripts/remote/run_stage_c_siff_ccsf_tau25_phase_a.sh`；
- checkpoint evaluator：`scripts/evaluate_stage_c_pcsd_cf_checkpoint.py`；
- four-layer analyzer：`scripts/analyze_stage_c_siff_ccsf_tau25_phase_a.py`；
- prelaunch checker：`scripts/check_stage_c_siff_ccsf_tau25_step7b.py`；
- manifest：`prelaunch/formal_phase_a_manifest.csv`；
- machine-readable gate：`prelaunch/prelaunch_gate.json`。

evaluator新增保存`probe_policy`、`probe_base_policy`、`probe_base_logits`、`probe_correction_logits`和
`probe_contrast_descriptor`。这些张量来自同一次forward，不读取future label构造policy；label只用于离线核算
best-arm match、skill alignment与allocation效果。`test_access_date`由每个run在实际进入test evaluator时写入，
prelaunch config中的null不会被当作实际访问日期。

## 4. Metric meanings

paper-facing层仍是20个test cells上的MSE/MAE。matched attribution层执行Step6的10项hard comparisons与
interaction gate。internal层包括：

- `oracle_gain_percent`：逐row/bin选择最优arm相对fused loss的剩余headroom；
- `pairwise_arm_nrmse`：arms是否给出不同forecast，而非collapse；
- `policy_normalized_entropy`：policy是否退化为单arm或近uniform；
- `best_arm_accuracy_gain_points`：final contrast-corrected policy相对同模型base policy的best-arm命中提升；
- `policy_skill_centered_alignment`：scope内centered policy与negative absolute error的相关；
- `allocation_gain_over_uniform_percent`：final policy融合相对uniform融合的probe MSE收益；
- `contrast_correction_rms_ratio`：correction logits相对base logits的RMS幅度；
- `prefix_projectivity_gap`：full-domain forecast与prefix调用的一致性。

这些diagnostics不能挽救negative official-test effectiveness gate。

## 5. Prelaunch result

[Fact] 本地prelaunch为15/15 categories pass：50-run/200-cell矩阵、10项hard comparisons、全部50条CLI
parse、test授权、contract hashes、tau0.25 freeze、pilot checkpoint no-reuse、runtime repair、三batch remote
resource smoke合同、checkpoint non-mutation、CCSF evaluator synthetic tensors与four-layer analyzer synthetic
decision均通过。

[Decision] `remote_phase_a_authorized=true`。remote resource smoke通过后可启动正式50-run矩阵；本结果仅证明
实验合同和工具一致，不构成模型有效性证据。

## 6. Failure attribution and rollback

- local/remote runtime或artifact contract失败：`optimization_or_numeric_pathology`或tooling fault，回Step7A；
- Step6 comparisons、threshold或claim-control map无法无损实现：回Step6；
- official-test main performance失败：关闭exact candidate或按internal evidence判断readout design fault；
- performance通过但controls失败：只保留`performance_partial_pass`，回对应attribution层；
- internal path inactive：`design_fault_suspected`，不能把正向性能直接提升为paper-core机制；
- Phase A全部通过也只授权Step9 review，confirmation seeds不会自动启动。
