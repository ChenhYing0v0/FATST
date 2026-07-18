# SC1-SIFF-v2-EQ-ATTR Step 6 归因矩阵冻结

## 1. 当前节点

| Field | Content |
| --- | --- |
| `current_step` | Step 6 complete；Step 7A local implementation authorized |
| `candidate_version` | `SC1-SIFF-v2-EQ-ATTR-v1` |
| `problem` | `SIFF_EQUAL` 已取得正向 test performance，但 ordered scale coordinate 的独立贡献尚未超过 EQUAL-context specificity controls |
| `existence_evidence` | fair re-audit 中 `SIFF_EQUAL` 相对 A6 为 `+1.6436%` MSE、`+0.9084%` MAE；内部 arms 未 collapse，但既有 PCC-context controls 不能完成归因 |
| `idea` | 在相同 EQUAL training context 下，分解 harmonic measure、equal-skill supervision、SIFF architecture 与四类 specificity control |
| `theory_check` | measure/equal decomposition 与 EQUAL-context controls 可分别回答 objective、architecture、ordering、scale partition 与 cross-arm interaction；不再用旧 PCC-context negative 代替当前归因 |
| `design` | 10 arms × 5 datasets × seed 2021，共 50 runs / 200 standard-horizon test cells |
| `narrative_gate` | conditional pass；只有七项 hard comparisons 全部通过，SIFF 才能进入 attributable paper-core candidate |
| `effectiveness_gate` | 每项 hard comparison 均要求 MSE macro gain ≥ `0.3%`、dataset wins ≥ `3/5`、horizon wins ≥ `3/4`、cell wins ≥ `11/20`；三项 main comparisons 另要求 MAE macro gain ≥ `0%` |
| `artifacts` | `configs/stage_c_siff_equal_attribution_v2.json`、`step6_gate.json` |
| `decision` | Step 6 checker `16/16` pass；仅授权 Step 7A，本阶段 remote/test 均为 false |

## 2. 为什么必须重新冻结

先前的 fair re-audit 已证明 `SIFF_EQUAL` 是当前最佳 performance carrier，但还没有证明收益来自 SIFF 所声称的
ordered scale coordinate。尤其是：

1. 旧的 `constant/permuted/Q1-wide/independent` controls 主要处于 PCC context，而 PCC 在本次公平重评估中会
   伤害 `SIFF_EQUAL`；它们不能直接解释 EQUAL context 下的正收益。
2. 内部诊断显示 `SIFF_EQUAL` 仍有 `6.394%` oracle headroom、pairwise arm NRMSE `0.133`、policy entropy
   `0.812`，且 MSE/MAE 均为正。这排除了“所有 arms 完全退化为同一输出”的简单解释，但不能完成因果归因。
3. 因此，当前状态是 `performance_positive / attribution_blocked`，不能直接补 seeds，也不能把内部健康度当成
   architecture specificity。

## 3. 十个冻结 arms

### 3.1 主效应与 decomposition

| Arm | Decoder | Objective | 作用 |
| --- | --- | --- | --- |
| `A6_FULL` | LBF | original full objective | paper-facing baseline |
| `A6_MEASURE` | LBF | harmonic measure only | 分离 harmonic measure effect |
| `PCSD_MEASURE` | PCSD | harmonic measure only | SIFF 的 matched architecture parent |
| `PCSD_EQUAL` | PCSD | equal-skill | 分离 equal-skill supervision effect |
| `SIFF_MEASURE` | ordered SIFF | harmonic measure only | 检查 SIFF 是否在普通 measure 下独立成立 |
| `SIFF_EQUAL` | ordered SIFF | equal-skill | 当前 candidate |

这六个 arms 形成三个不同问题：

1. `A6_MEASURE - A6_FULL`：harmonic measure 本身是否解释了收益；
2. `PCSD_EQUAL - PCSD_MEASURE` 与 `SIFF_EQUAL - SIFF_MEASURE`：equal-skill supervision 是否是必要条件；
3. `SIFF_MEASURE - PCSD_MEASURE` 与 `SIFF_EQUAL - PCSD_EQUAL`：SIFF architecture 在两种 objective context 下的
   增量。

`SIFF_MEASURE - PCSD_MEASURE` 是 claim-narrowing comparison，不是 hard gate。若它失败而
`SIFF_EQUAL - PCSD_EQUAL` 通过，论文 claim 必须收紧为“equal-skill-trained scale field”，不能声称
objective-robust SIFF architecture。

### 3.2 EQUAL-context specificity controls

| Arm | 与 `SIFF_EQUAL` 的唯一关键差异 | 反驳的替代解释 |
| --- | --- | --- |
| `SIFF_CONSTANT_EQUAL` | ordered coordinate 改为 constant | generic multi-arm capacity |
| `SIFF_PERMUTED_EQUAL` | ordered scale assignment 被 permutation | 任意离散 arm identity |
| `SIFF_Q1_WIDE_EQUAL` | scale partition 改为单一 wide partition | 只是更宽的 local field |
| `SIFF_INDEPENDENT_EQUAL` | 移除 ordered cross-scale coupling | independent-arm ensemble |

四个 controls 与 `SIFF_EQUAL` 使用相同 dataset profiles、seed、checkpoint selector、objective family、rank budget 与
from-scratch joint training。它们全部属于 hard attribution gate。

## 4. 四层统一评估

### Layer 1：paper-facing effectiveness

`SIFF_EQUAL` 必须分别超过：

1. `A6_FULL`；
2. `A6_MEASURE`；
3. `PCSD_EQUAL`。

每项 comparison 独立应用冻结的 MSE gate；三个 main comparisons 还要求 MAE macro 不为负。不能用一项较大的
gain 抵消另一项失败。

### Layer 2：matched mechanism attribution

`SIFF_EQUAL` 必须分别超过四个 EQUAL-context controls。七项 hard comparisons 必须全部通过，才可把正收益
归因于完整 SIFF chain。

### Layer 3：internal mechanism health

冻结以下诊断：

- 所有 arm tensors、weights 与 losses 必须 finite；
- prefix projectivity maximum gap ≤ `1e-7`；
- oracle-positive datasets ≥ `4/5`；
- pairwise arm NRMSE macro ≥ `0.05`；
- policy entropy macro 位于 `[0.20, 0.95]`；
- nonconstant component RMS ratio ≥ `0.01`；
- ordered-vs-constant scale-component probe NRMSE ≥ `0.01`。

这些量用于判断 arms 是否 collapse、policy 是否近乎 uniform/one-hot、ordered component 是否实际进入输出。它们
不能替代 Layer 1/2。

### Layer 4：failure attribution

| 结果组合 | Decision |
| --- | --- |
| effectiveness pass，control attribution fail | `partial_pass_attribution_blocked`；回 Step 4/6 收紧或重构 claim |
| effectiveness + attribution pass，internal health fail | `design_fault_suspected`；回 Step 4 检查 intervention/readout |
| effectiveness fail，但 oracle/headroom 为正 | 关闭 exact candidate；内部 diagnostic 不得挽救 performance gate |
| 四层全部通过 | 才授权 seeds 2022/2023 confirmation |
| 出现 numeric/pathology | 只修复 exact implementation/protocol，不作方向级拒绝 |

## 5. Test-informed 边界

该 candidate 由既有 test audit 触发，因此从冻结起明确标记 `test_informed=true`。禁止声称它基于 untouched
holdout，也禁止按 dataset、horizon 或 cell 调参。

- Phase A：10 arms × 5 datasets × seed 2021 = 50 runs / 200 test cells；
- Confirmation：只在 Phase A 七项 hard comparisons 全部通过后，运行 seeds 2022/2023；
- Confirmation 完整规模：100 runs / 400 test cells；
- Phase A 与 confirmation 之间不得修改 model、objective、profile、checkpoint rule、gate 或 control definition。

## 6. Code-theory consistency

现有 `train_repo.py` 已支持 LBF 的 `measure_only`、PCSD/SIFF 的 `measure_only/equal_skill`，并已有
ordered/constant/permuted/Q1-wide/independent SIFF modes。四-horizon validation checkpoint selector 与正式 test
audit path 也已存在。

因此 Step 7A 不应修改 SIFF 数学公式，主要工作是：

1. 生产 runner 与 10-arm config wiring；
2. scale-component artifact 的稳定导出；
3. 四层 analyzer 与 machine-readable decision；
4. construction、initialization、forward、gradient、parameter matching 和 diagnostic completeness 的 local gate。

若任一 arm 无法从现有 production path 构造，必须回到 Step 6 修正 protocol，不能静默删除 control 或替换为较弱
proxy。

## 7. 决定

Step 6 protocol checker 为 `16/16`，profile hash 为
`80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a`。

当前只授权 `SC1-SIFF-v2-EQ-ATTR` 进入 Step 7A local implementation。remote launch、official test access 与
confirmation 均保持 false。
