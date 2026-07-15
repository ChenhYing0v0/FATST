# SC1-JAPO Step 8 Seed-2021 Validation Screen

## Decision Summary

| Field | Result |
| --- | --- |
| `matrix` | 5 datasets × 7 arms × seed2021；35/35 complete |
| `split` | validation-only；dense H1..720 |
| `decision` | `seed2021_inconclusive_run_seed2022_only` |
| `JOINT vs A6 macro MSE` | `-1.3754%`；0/5 datasets positive |
| `JOINT vs same-bank median` | `-0.0780%`；2/5 positive |
| `immediate_fail` | false |
| `provisional_pass` | false |
| `numeric/protocol pathology` | none detected |
| `next_action` | 不改design，只补seed2022；two-seed mean复用原threshold |

## 1. What We Planned To Test

本轮不是测试“两个experts是否比A6容量更大”，而是测试完整JAPO mechanism是否同时满足：

1. `JOINT-GEO > A6`：新decoder具备实际forecast effectiveness；
2. `JOINT-GEO > UNIFORM`：收益不是双expert capacity或ensemble effect；
3. `JOINT-GEO > HISTORY/ATOM`：history与atom interaction都必要；
4. `JOINT-GEO > PERM/RANDOM`：canonical RGNB geometry不是任意descriptor坐标；
5. 上述比较在五个datasets上具有一致方向，而非单dataset偶然性。

Step 6预注册：若结果既不满足provisional pass，也未达到严重immediate-fail，则状态为`inconclusive`，只能
原样补seed2022；不得修改router、rank、init、epoch或objective。

## 2. Artifact And Protocol Audit

- 35/35 `metrics_by_target_horizon.csv`均含完整H1..720；
- 35/35 effective configs均为full-H720 pointwise L1、best-val、`final_split=val`、test=false；
- 35/35 checkpoints均from-scratch joint training，frozen parameter tensors为0；
- paired Encoder、expert-bank与RGNB basis hashes全部通过；
- trained prefix最大误差`8.345e-7`，patch-block rewrite最大误差`1.907e-6`；
- 所有metrics、training logs、routing diagnostics与checkpoint invariants均finite。

[Fact] 没有NaN、divergence、prefix破坏、hash mismatch、frozen replacement或validation/test混用。因此该结果可
用于exact design判断，不属于`diagnostic_invalid_for_direction_rejection`。

## 3. Dataset-Level Results

所有improvement均定义为$100(1-\mathrm{candidate}/\mathrm{reference})$，正值更好。

| Dataset | JOINT MSE AUC | vs A6 MSE | vs A6 MAE | vs same-bank median | short H1-96 | middle H97-336 | long H337-720 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 1.187002 | -1.417% | -0.887% | -0.064% | -3.851% | -1.858% | -0.903% |
| ETTh2 | 0.438423 | -2.338% | -1.048% | -0.650% | -1.493% | -2.500% | -2.351% |
| ETTm1 | 0.670153 | -0.539% | +0.274% | +0.681% | +1.449% | -0.418% | -0.780% |
| ETTm2 | 0.199523 | -2.479% | -0.469% | -0.528% | -7.293% | -3.567% | -1.603% |
| Weather | 0.497622 | -0.103% | -0.279% | +0.170% | -3.013% | -0.113% | +0.338% |

[Strong Evidence] 单seed下JAPO没有超过A6：MSE为0/5正向，且ETTh2/ETTm2约差2.3%–2.5%。正向信号只存在于
ETTm1 MAE/short segment与Weather long segment，不能支撑full candidate。

## 4. Same-Bank Attribution

| Reference | JOINT macro improvement | Positive datasets |
| --- | ---: | ---: |
| UNIFORM | -0.397% | 2/5 |
| HISTORY | -0.100% | 2/5 |
| ATOM | -0.386% | 2/5 |
| JOINT-PERM | +0.275% | 3/5 |
| JOINT-RANDOM | +0.138% | 3/5 |
| per-dataset control median | -0.078% | 2/5 |

[Strong Evidence] canonical descriptor相对PERM/RANDOM有很小的正向均值，但JOINT没有超过UNIFORM、HISTORY或
ATOM。当前收益不能归因到完整history-atom interaction。

[Important Boundary] `capacity_control_explains` hard failure尚未触发，因为预注册条件要求same-bank median
$\le0$且正向datasets不超过1；当前为`-0.078%`、2/5。差距非常小，seed变化可能改变符号，所以必须补seed2022。

## 5. Routing And Optimization Diagnostics

JOINT-GEO trained normalized entropy：

| Dataset | Entropy | Mean expert usage | JOINT epochs / A6 epochs |
| --- | ---: | ---: | ---: |
| ETTh1 | 0.998837 | 0.5125 / 0.4875 | 7 / 8 |
| ETTh2 | 0.999258 | 0.4962 / 0.5038 | 7 / 7 |
| ETTm1 | 0.996697 | 0.4856 / 0.5144 | 8 / 8 |
| ETTm2 | 0.998420 | 0.5043 / 0.4957 | 6 / 10 |
| Weather | 0.993263 | 0.4724 / 0.5276 | 11 / 17 |

[Strong Evidence] router平均仍非常接近uniform；最低entropy也有`0.9933`。这不证明每个history-atom pair完全
相同，但说明总体specialization很弱。

[Hypothesis] exact E2 design的主要风险不是numerical instability，而是router intervention过弱或优化后仍接近
uniform mixture。ETTm2与Weather的JOINT还比A6更早early-stop，optimization sensitivity可能放大性能差距。

[Self-Critique] 高entropy并不自动等于机制无效：连续soft routing可以在接近0.5时产生有意义的小幅operator变化；
当前也只有一个seed。因此不能据此修改temperature、init scale或加入auxiliary specialization loss。

## 6. Frozen Gate Evaluation

### Immediate fail

- JOINT vs A6：`-1.375%`，虽0/5正向，但未达到$\le-10\%$；
- JOINT vs same-bank median：`-0.078%`、2/5正向，不满足正向$\le1$；
- numeric/protocol pathology：false。

因此`immediate_fail=false`。

### Provisional pass

- JOINT vs A6要求macro $>0$且4/5正向：失败；
- JOINT vs每个control要求macro $>0$且至少3/5正向：UNIFORM/HISTORY/ATOM失败；
- vs same-bank median要求至少`+1%`且4/5正向：失败。

因此`provisional_pass=false`。

## 7. Failure Attribution And Decision

- `hypothesis_false`：未授权；只有一个seed；
- `intervention_point_wrong`：仍可能，但未由single-seed直接证明；
- `readout_or_head_design_wrong`：suspected，原因是router under-specialization与A6 0/5；
- `optimization_or_numeric_pathology`：numeric false；optimization sensitivity suspected；
- `capacity_control_explains`：hard gate未触发，但UNIFORM当前略优于JOINT。

[Decision] 本轮为稳定但负向的`inconclusive`结果。严格执行预注册协议：保持E2/K256/G32、initialization、
objective、profiles与seven arms不变，只补seed2022。随后在two-seed mean上复用provisional-pass thresholds：

- 若two-seed mean通过，才补seed2023；
- 若不通过，停止该exact JAPO design并完成Step 4 attribution；
- 不允许在seed2022之前或之后依据边缘结果调router/temperature/loss。
