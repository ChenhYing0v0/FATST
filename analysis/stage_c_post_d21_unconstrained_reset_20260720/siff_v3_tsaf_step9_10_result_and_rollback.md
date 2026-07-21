# SC1-SIFF-v3-TSAF-v1 Step 9/10 Result and Rollback

## 1. Decision summary

| Field | Result |
| --- | --- |
| `current_step` | Step 9/10 complete；return Step 2/4 within SIFF-first paperization |
| `candidate_version` | `SC1-SIFF-v3-TSAF-v1` |
| `paper_facing_effectiveness` | fail |
| `matched_mechanism_attribution` | fail |
| `internal_mechanism_health` | pass，但不能覆盖effectiveness failure |
| `machine_decision` | `close_exact_candidate_effectiveness_fail_rollback_step2_or_4` |
| `research_decision` | `close_tsaf_v1_shared_field_design_keep_siff_v2_immutable_parent` |
| `active_method` | none；SIFF-v2仍是immutable paperization parent，不把control post-hoc升级为method |
| `confirmation_authorized` | false |

[Fact] 冻结的45-run/180-cell official-test matrix完整，TSAF相对`A6_MEASURE`和SIFF-v2 parent的primary
effectiveness gates均失败；ordered field、ordered scale、target-coordinate和shared-field四个归因问题也均未得到
正向支持。

[Strong Evidence] 失败不是NaN、checkpoint mutation、request leakage、inactive field或arms collapse造成的。
TSAF确实学出了非恒定allocation与非零scale component，但这些内部活动没有转化为test收益。

[Decision] 关闭exact `TSAF-v1 shared target-scale allocation field`。不补confirmation seed，不做temperature、rank、
width、readout、loss或per-dataset/horizon rescue。D22-C的target-coordinate information-access problem evidence和
SIFF-v2 identity均不被本结果推翻。

## 2. Protocol validity

- user authorization：2026-07-21，覆盖25个new from-scratch runs与一次完整formal test；
- training commit：`6cef063ecfa4cc12aaa3eb0e5e1bbbfcca42092b`；
- formal-test commit：`4cc96f21e23c159e37757c66ec2e5c68358c5718`；
- config SHA256：`32ec229c66c5683a48793b5bbde3c4f6e99790e4f6e05e4076ec6284c351e748`；
- profile SHA256：`80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a`；
- checkpoint selector：validation mean MSE over `{96,192,336,720}`；
- training：25/25 complete；formal test：25/25 new checkpoints complete；20/20 historical references reused；
- `run_audit.csv`：45/45 `status=ok`、45/45 `protocol_pass=true`、45个unique checkpoint hashes；
- paired initialization：每个dataset的9 arms只有一个`encoder_initialization_hash`；
- 25/25 new checkpoints均`checkpoint_retrained=true`、test authorization true、test date `2026-07-21`，且
  test前后SHA256不变；
- 25个new runs全部finite并由early stopping结束；best epoch范围1--10，median为3。

聚合`summary.json`中的顶层`test_access_date`为空，是analyzer没有把per-run invariant date提升到顶层的metadata
serialization omission。25份`test_audit_invariants.json`均明确记录`2026-07-21`，因此不是unauthorized test或
artifact缺失。

## 3. Statistics and roles

对每个dataset × horizon cell，comparison gain定义为

$$
g_{d,h}=100\times\frac{M_{\mathrm{reference},d,h}-M_{\mathrm{candidate},d,h}}
{M_{\mathrm{reference},d,h}}.
$$

`macro_gain_percent`是20个cells的$g_{d,h}$算术平均，正值表示candidate更好。`cell_wins`直接计数
$g_{d,h}>0$；`dataset_wins`和`horizon_wins`分别先在对应四个horizons或五个datasets内平均$g$再计数。

- validation只用于共同checkpoint selection和split-transfer audit；
- official test承担paper-facing effectiveness与matched attribution；
- internal diagnostics只回答路径是否活跃，不能挽救negative test gate；
- historical reused arms只作冻结references；25个new arms均为matched end-to-end joint training。

## 4. Paper-facing effectiveness

### 4.1 Primary comparisons

| Comparison | MSE macro gain | MSE wins | MAE macro gain | MAE wins | Gate |
| --- | ---: | --- | ---: | --- | --- |
| TSAF vs `A6_MEASURE` | -1.2854% | 9/20 cells；2/5 datasets；0/4 horizons | -1.3146% | 5/20；1/5；0/4 | fail |
| TSAF vs SIFF-v2 parent | -1.0422% | 7/20；1/5；0/4 | -0.9183% | 4/20；1/5；0/4 | fail |

20-cell raw means同样给出一致排序：`A6_MEASURE` MSE/MAE为`0.308118/0.345504`，SIFF-v2 parent为
`0.308663/0.346449`，TSAF为`0.312599/0.350151`。TSAF只在ETTm1的dataset-mean MSE上略优于parent，
但MAE仍略差；最大退化来自ETTh1，dataset-mean MSE相对parent约`-4.26%`。

TSAF相对较弱的`A6_FULL` MSE为`+0.6192%`，但MAE为`-0.0019%`。这不能替代已冻结的primary
`A6_MEASURE` gate，也不能从较弱carrier中选择有利reference。

### 4.2 Validation-to-test transfer

| Comparison | Validation MSE gain | Test MSE gain | Interpretation |
| --- | ---: | ---: | --- |
| TSAF vs SIFF-v2 parent | +0.7700% | -1.0422% | clear reversal |
| TSAF vs `A6_MEASURE` | +0.3335% | -1.2854% | clear reversal |
| TSAF vs categorical target-only | +0.4769% | -1.0191% | field benefit does not transfer |
| TSAF vs independent target-only | -0.0167% | -1.2785% | shared field never beats independent control |

因此不能把validation positive当作机制成立，也不能通过换checkpoint score或读取test后再选epoch修复。

## 5. Matched mechanism attribution

| Question | MSE macro gain | Wins | Decision |
| --- | ---: | --- | --- |
| ordered field vs categorical target-only | -1.0191% | 7/20；2/5 datasets；0/4 horizons | fail |
| ordered scale vs permuted scale | -0.0796% | 6/20；2/5；0/4 | fail |
| target-coordinate vs global | -0.0405% | 9/20；3/5；1/4 | fail |
| shared field vs independent target-only | -1.2785% | 4/20；0/5；0/4 | fail；strict superiority亦fail |

categorical target-only相对SIFF-v2 parent为MSE `-0.0246%`、MAE `-0.0397%`，实质上持平但没有正向
effect。capacity-matched independent target-only相对parent则为MSE `+0.2383%`、MAE `+0.0898%`，12/20 MSE
cells、4/5 datasets和3/4 horizons正向。

这个independent-control信号需要谨慎解释：其MSE低于预注册primary threshold `+0.3%`，只有seed2021，且它在
实验前被定义为capacity/control arm而不是paper method。它可以保留为新的Step2/4 weak lead，但不得在看到test
后直接改名、补seed或升级为SIFF-v3。

## 6. Internal mechanism health

| Dataset | Arm diversity NRMSE | Target-scale surface std | Ordered/permuted allocation NRMSE | Normalized entropy | Scale-component RMS | Request gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Weather | 0.1900 | 0.1638 | 0.2882 | 0.8016 | 0.1931 | 0 |
| ETTm1 | 0.1637 | 0.1086 | 0.2759 | 0.9155 | 0.1225 | 0 |
| ETTh1 | 0.3570 | 0.0263 | 0.1444 | 0.9943 | 0.2003 | 0 |
| ETTh2 | 0.0778 | 0.0197 | 0.1405 | 0.9969 | 0.0280 | 0 |
| ETTm2 | 0.0829 | 0.0483 | 0.2686 | 0.9813 | 0.0363 | 0 |

所有预注册aggregate health gates通过。这里的含义仅是：arms有差异、allocation不是常数、permutation会改变
allocation、scale component非零且requested-H invariance保持。ETTh1/ETTh2 entropy接近1，说明这两个dataset的
allocation接近uniform；即使如此，它也不是numeric failure，因为输出与gradient仍finite且nonzero。

## 7. Failure attribution

1. `optimization_or_numeric_pathology`：不支持。45/45 protocol valid，所有test tensors finite，checkpoint未变；
2. `capacity_control_explains`：不是“更大capacity带来TSAF收益”，因为TSAF没有收益，且parameter-matched
   independent control显著优于shared field；
3. `hypothesis_false`：可以拒绝“当前sample-shared ordered target-scale field能稳定改进SIFF”的exact hypothesis；
   但independent target-only的微弱正信号使更广义的history-free target allocation仍为unresolved，而非方向级false；
4. primary attribution：`readout_or_head_design_wrong` / `intervention_point_wrong`。ordered-scale和target-coordinate
   semantics在当前shared allocation readout中没有提供可转移收益，permuted/global controls几乎等价或略好；
5. rollback：关闭TSAF-v1 exact design，回Step2/4重新判断paper problem与机制边界，不回Step7做参数修补。

## 8. Paper and execution consequences

- SIFF-v2继续作为immutable performance-near paperization parent；本结果不把其历史attribution failure改写为pass；
- TSAF不进入paper-core、confirmation或baseline benchmark阶段；
- 当前没有active successor method；SC-MNB仍是supporting source/control inventory，65-run execution仍false；
- 不从局部有利cells、`A6_FULL` comparison或internal-health gates包装新contribution；
- 下一研究节点必须先完成新的Step2/4 narrative/design gate。若审计independent target-only weak lead，必须创建新的
  test-informed candidate identity和完整matched protocol，不能把本次control arm post-hoc晋升；
- 不设计第二loss/router，不恢复CCSF、D17-D21、seed/rank/readout/representation rescue。

## 9. Artifacts

- machine summary：`siff_v3_tsaf_step9_10_result/primary/summary.json`；
- formal comparisons：`siff_v3_tsaf_step9_10_result/primary/comparison_summary.csv`与
  `comparison_cells.csv`；
- all 180 test cells：`siff_v3_tsaf_step9_10_result/primary/test_metrics_standard_horizons.csv`；
- mechanism diagnostics：`siff_v3_tsaf_step9_10_result/primary/mechanism_health.csv`；
- checkpoint/protocol audit：`siff_v3_tsaf_step9_10_result/primary/run_audit.csv`；
- supervisor、formal-test launch与per-run validation/invariants：`siff_v3_tsaf_step9_10_result/raw_lite/`。
