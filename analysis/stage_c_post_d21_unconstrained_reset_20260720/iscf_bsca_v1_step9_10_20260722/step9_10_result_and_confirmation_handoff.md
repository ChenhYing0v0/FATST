# ISCF-BSCA-v1 Step9/10：single-seed frozen formal test

## 结论

Decision=`performance_partial_pass_pending_confirmation_seed`。ISCF-BSCA-v1 在冻结的 seed2021 official-test matrix 上刚好通过全部 preregistered performance gates：macro MSE/MAE gain=`+0.3104%/+0.4902%`，15/20 cells、3/5 datasets、3/4 horizons MSE positive。结果足以把 BSCA 从 design candidate 升为有初步性能证据的 contribution candidate，但单 seed 与明显 dataset heterogeneity 不足以记为 `passed_core_candidate`。

## Protocol 与 test role

| Field | Value |
|---|---|
| `test_access_date` | 2026-07-22 |
| `user_authorization` | 用户明确授权 Step4–7A、five-run training 与 one frozen formal test |
| `candidate_version` | `ISCF-BSCA-v1-seed2021-frozen` |
| `checkpoint_retrained` | true；五个 candidate checkpoints 均为本候选 from-scratch training |
| `checkpoint_mutated_during_test` | false；runner 对每项 SHA256 before/after exact equality |
| `checkpoint_selection` | validation mean MSE over H96/H192/H336/H720 |
| `test_role` | primary mechanism effectiveness and paper benchmark |
| `matrix_complete` | true；5 datasets × 4 horizons × MSE/MAE |
| `test_informed` | true |

五个 candidate checkpoint SHA256 记录在 `primary/run_audit.csv`。Candidate 与 EQUAL 的 encoder、PCSD initialization、coordinate、partition 和 SIFF basis hashes 在每个 dataset 均逐项相等；参数量、forward graph、optimizer、seed、data、checkpoint rule 与 evaluation protocol 相同，唯一训练差异是 BSCA uniform anchor。

## 统计量定义

- `cell gain (%) = 100 * (EQUAL - BSCA) / EQUAL`，正值表示 BSCA 更好。
- `macro gain`：20 个 dataset-horizon cell gain 的等权平均，不按样本量加权。
- `dataset win`：该 dataset 四个 horizon 的 MSE gain 均值大于 0。
- `horizon win`：该 horizon 五个 dataset 的 MSE gain 均值大于 0。
- `policy normalized entropy`：对 `policy_row_bin_usage:[N,8,5]` 的 scope entropy 以 $\log 5$ 标准化后求均值。
- `policy usage max/min`：上述 policy 在 row 与 future-bin 上求均值后，五个 scopes 的最大/最小 marginal usage。
- `pairwise arm L1`：`probe_arms:[256,5,720]` 的十个 scope-pair absolute difference 均值。
- `oracle headroom`：每个 probe coordinate 选择 squared-error 最小 arm 所得 oracle MSE，相对 fused MSE 的改善百分比。它只表示未兑现互补性，不是可部署性能。

## Layer 1：paper-facing effectiveness

| Surface | MSE gain | MAE gain | Wins |
|---|---:|---:|---:|
| validation | +0.6490% | +0.4492% | 5/5 datasets；4/4 horizons |
| official test | **+0.3104%** | **+0.4902%** | 15/20 cells；3/5 datasets；3/4 horizons |

Test dataset mean MSE gains：Weather `+0.8736%`、ETTm1 `+2.2467%`、ETTh1 `+0.1743%`、ETTh2 `-0.0048%`、ETTm2 `-1.7375%`。Test horizon mean gains：H96 `+1.2543%`、H192 `+0.0502%`、H336 `-0.2255%`、H720 `+0.1628%`。

关键负向结果必须保留：ETTm2 在 H192/H336/H720 分别为 `-2.6467%/-3.7139%/-1.1281%`，形成显著 validation/test reversal；ETTh2 基本 tie，H336/H720 略负。不能选择性只报告 Weather/ETTm1。

## Layer 2：matched mechanism attribution

ISCF-EQUAL 是 exact same-architecture/no-anchor control，且五个 datasets 的所有 initialization hashes 完全配对。BSCA 没有新增参数、capacity、router 或 inference path，因此 observed difference 可归因于 train-time uniform anchor 所改变的 joint optimization/co-adaptation，而不是参数量或随机初始化。

该证据支持“balanced co-adaptation 有效”，但不能支持“BSCA 使 policy 学到更强 sample-wise conditional routing”。事实上 policy 更接近 uniform；generic load-balancing primitive 也不是 novelty claim。

## Layer 3：internal mechanism health

| Metric, five-dataset mean | BSCA | EQUAL | Interpretation |
|---|---:|---:|---|
| policy normalized entropy | 0.9983 | 0.7913 | anchor 确实产生 broad pointwise access |
| marginal policy usage max | 0.2042 | 0.2528 | 五 scopes 使用接近 0.2 |
| pairwise arm L1 | 0.1165 | 0.1219 | arms 未 collapse，但 diversity 略降约 4.4% |
| oracle headroom | 32.5557% | 33.0144% | 大量 complementarity 保留，略低于 EQUAL |

所有 test tensors finite，prefix/readout/protocol invariants 5/5 pass。Mechanism path因此是“防止 policy-mediated gradient starvation、让所有 scopes 持续参与 joint learning”，而不是“提高 routing specialization”。Weather/ETTm1 的性能改善与高 entropy 同时出现；ETTm2 也高度 uniform 却 test 反转，说明 balance 不是 dataset-independent guarantee。

## Layer 4：failure attribution 与边界

本轮不是 failure；exact v1 通过冻结 gate。最准确状态为 `performance_partial_pass_pending_confirmation_seed`：

- [Strong Evidence] exact matched seed2021 上 objective effect 为正，且达到全部 gate。
- [Uncertainty] macro MSE 仅高出 threshold 0.0104 percentage points；ETTm2 material negative；单 seed 无法排除 optimizer variance。
- [Narrative boundary] 可声称 training-time balanced scope co-adaptation 的初步有效性；不可声称 learned specialization、universal gain、generic load-balancing novelty或多 seed robustness。

若 confirmation seeds 保持 macro positive、至少 3/5 datasets 且 ETTm2 不再形成 dominant reversal，可升级为 `passed_core_candidate`。若跨 seed 均值回到 tie/negative，应归因为 `optimization_variance_or_dataset_heterogeneity_explains`，回 Step4 收窄 claim，而不是否定 fixed ISCF architecture。

## Protocol faults（诚实记录）

首次 validation diagnostics 因 design config 缺 `coupling_scales` 在 probe postcondition 报错；training/checkpoints 已完成且未变，v0.1 只补 evaluator schema并做 SHA-preserving replay。首次 formal-test invocation 因 generic authorization schema 缺字段产生 `test_access_authorized=false`，3/5 artifacts 被标记 `pass=false`，未读取 partial metrics；修复后同一 candidate/version 完成 5/5，未重训或改 gate。这两项均是 `diagnostic_protocol_fault_predecision`，不是 model pathology。

## artifacts

- `primary/summary.json`：machine decision。
- `primary/comparison_cells.csv`：40 validation + test comparison rows。
- `primary/internal_health.csv`：candidate/control internal statistics。
- `primary/run_audit.csv`：checkpoint hashes、objective、initialization pairing 与 invariants。
- `remote_records/`：initial/replay training 与 successful formal-test launch records。

## 下一步

当前 task 授权已全部执行。下一步建议冻结 confirmation seeds 2022/2023，只训练 BSCA 五 datasets，并复用已有 seed2022/2023 EQUAL official-test artifacts；不得调 lambda、按 dataset/horizon 改 objective 或根据本次 test 选择 cell。该额外 training/formal test 尚未授权，本报告不自动启动。
