# ISCF-v0 SAC Step 8 Validation Artifact Audit and Test Handoff

## 1. 审计问题与边界

本报告只回答两个问题：冻结的25个SAC control trainings是否完整、可复现且具备进入formal test的协议条件；
validation现象是否暴露明显的protocol/numeric pathology。它不以validation MSE/MAE通过或拒绝scope architecture，
也不授权official test。

当前candidate仍为`ISCF-v0 conditional paperization candidate`，active paper-core method仍为none。
`formal_test_access_authorized=false`，本轮test access为0。

## 2. Artifact构造与完整性

| Item | Result |
| --- | --- |
| remote commit | `78cbcf47e1cb5f6d24a01ac5ad8fea8b0deebbb9` |
| output root | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_iscf_v0_sac_v1` |
| launch / finish | `2026-07-21T18:58:40+08:00` / `2026-07-21T20:24:32+08:00` |
| new training runs | `25/25` |
| new checkpoints | `25/25`，均已计算SHA256 |
| new validation files | `25/25` |
| effective validation matrix | `60 runs × 4 horizons = 240 rows` |
| effective run audit | `60/60 ok` |
| formal-test artifacts | `0/25` new；未访问test |
| log scan | 无Traceback、OOM、NaN或Inf命中 |

`raw_lite/`只同步metrics、training log、effective config、initialization contract和model diagnostics；
remote checkpoints继续保留在repo外output root，未提交大权重文件。25个checkpoint的存在性和SHA256已在remote只读核验，
formal test runner仍需在test前后逐run核验nonmutation。

## 3. Analyzer修正与统计定义

现有Step9/10 analyzer原本要求test artifacts齐全，无法诚实处理training-only checkpoint。为此新增
`--validation-only`：只读取validation metrics、effective config、initialization contract与diagnostics，输出
`validation_readiness.json`，不调用official-test `decide()`。

审计同时发现一个checker-specific bug：A6_FULL不使用PCSD，但历史effective config保留默认
`pcsd_partition=canonical`，而SAC role将其标成`control`。旧checker会把15个A6 references误判为partition mismatch。
修正后只对`partition=control`的非PCSD reference跳过该无意义字段；ISCF canonical/random partition检查保持严格。

对任意candidate $c$、reference $r$、metric $m$，cell gain定义为

$$
g_{d,s,h}^{(m)}=100\left(1-\frac{m_{c,d,s,h}}{m_{r,d,s,h}}\right).
$$

`macro_gain_percent`是60个dataset-seed-horizon cell gain的算术平均；`dataset_wins`、`horizon_wins`和
`positive_seed_macros`分别先在对应切片内取平均，再统计严格大于0的切片数。这里所有量均为validation observation，
不能替代预注册official-test gate。

## 4. Validation observations

| Comparison | MSE macro | MAE macro | MSE cells | Datasets | Horizons | Seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ISCF-v0 over Q1-WIDE | `+1.0704%` | `+0.7538%` | `46/60` | `4/5` | `4/4` | `3/3` |
| canonical over RANDOM-PARTITION | `-0.1823%` | `-0.3075%` | `23/60` | `2/5` | `1/4` | `1/3` |
| ISCF-v0 over A6_FULL context | `+2.4181%` | `+1.4277%` | `51/60` | `5/5` | `4/4` | `3/3` |

Q1-WIDE comparison在validation上方向清晰：MSE的H96/H192/H336/H720分别为
`+1.8795/+1.1279/+0.9526/+0.3214%`，three-seed macros均正。它初步支持“independent scope maps不只是
shared-width capacity”的解释，但Weather MSE为`-0.0955%`，且该结论仍可能发生validation→test reversal。

canonical vs RANDOM在validation上不支持temporal contiguity/nesting：MSE仅H96为`+0.0445%`，其余三个horizons
均负；MAE在5/5 datasets、4/4 horizons和3/3 seeds的macro均负。这是必须完整报告的negative lead，但按项目协议，
validation只用于checkpoint selection与health，不能据此关闭ISCF或触发rollback。

## 5. Four-layer interpretation

1. `paper_facing_effectiveness`：缺失。official test 0/25，不能作Step9/10 decision。
2. `matched_mechanism_attribution`：validation上Q1 favorable、RANDOM unfavorable；只作预警，不是gate。
3. `internal_mechanism_health`：15/15 dataset-seed pairs通过。canonical/random Encoder与PCSD initialization匹配、
   active parameters完全相同、partition hashes不同；Q1 parameter gaps与预注册值一致。
4. `failure_attribution`：当前为`unresolved_pending_formal_test`。没有numeric/protocol pathology；若official test复现
   RANDOM failure，归因将是`temporal_scope_structure_not_supported`，而不是optimization fault。

反方解释是：validation negative可能是随机partition真正提供了regularization，也可能只是checkpoint-level noise；
TSAF已有validation→test反转先例，因此不能从当前排序外推official-test结论。

## 6. Decision与下一动作

Decision=`formal_test_ready_pending_user_authorization`。

这只表示artifact/protocol readiness通过，不表示任一primary comparison通过。下一步必须取得独立user authorization，
再把config切换为formal-test authorized状态、commit/push、remote fast-forward，并仅运行一次冻结的
`FORMAL_TEST_ONLY=1` 25-run new matrix。完成后由现有analyzer联合35个historical references作60-run/240-cell
official-test审计。test前不得改candidate、rank、partition seed、objective、checkpoint或gates。

若Q1-WIDE或RANDOM任一official-test primary gate失败，ISCF-v0回到strong carrier/control portfolio；不进行
seed、rank、partition、loss、router或requested-H rescue。

