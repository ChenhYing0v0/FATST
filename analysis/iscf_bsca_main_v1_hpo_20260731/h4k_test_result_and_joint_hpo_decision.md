# ISCF-BSCA-MAIN-v1 H4K Formal-Test Result and Joint-HPO Decision

## 1. Decision summary

| Field | Value |
| --- | --- |
| `date` | 2026-08-04 |
| `formal_test` | 24/24 checkpoints；96/96 standard-horizon cells complete |
| `checkpoint_immutability` | 24/24 pass |
| `test_artifact_invariants` | 24/24 pass；no ABORT |
| `joint_search_pool` | H1--H4K；117 dataset-level trials |
| `joint_selected_leads` | MSE 15/28；MAE 15/28；combined 30/56 |
| `success_gate` | fail：20/28、20/28、40/56均未达到 |
| `decision` | `H4K_complete_continuous_improvement_no_new_leads_gate_fail_return_step6` |

## 2. Formal-test audit

H4K formal test以commit `d687428e1a2ef6882d32813f1fe4bf68b236c537`于09:21:23启动，09:27:33完成，wall time约6分10秒。Manifest SHA256=`61446262b9ed645fb1e2ebd233a0dd88c7d1743ff80fa8ed0fc2a5a52e2747ab`，完整矩阵为24 checkpoints × `{96,192,336,720}`，每cell同时生成MSE/MAE。

Audit结果：24/24 dense 1--720 metric artifacts、24/24 invariants与24/24 diagnostic archives完整；checkpoint pre/post SHA256一致；没有ABORT、partial publication、NaN、Inf或schema/provenance错误。结束后GPU0/1/2均回到18 MiB idle，quota约195/220 GiB。首次remote preflight因人工填写错误的完整commit字符串在code gate处拒绝，未访问test；使用实际commit重跑preflight后通过，随后只执行一次formal test。

## 3. Frozen 117-trial joint selector result

H4K与既有93 trials合并后，frozen 1% joint-mean guard与balanced leading-cell selector得到：

| Quantity | H4J selected | H4K selected | Change |
| --- | ---: | ---: | ---: |
| MSE leading cells | 15/28 | 15/28 | 0 |
| MAE leading cells | 15/28 | 15/28 | 0 |
| Combined leading cells | 30/56 | 30/56 | 0 |
| Seven-dataset cell-macro MSE | 0.262748 | 0.262696 | +0.0199% improvement |
| Seven-dataset cell-macro MAE | 0.308815 | 0.308726 | +0.0286% improvement |

合法dataset-level selector、unrestricted single-profile selector和逐cell diagnostic oracle均为30/56。这意味着117个已测试profiles内不存在能够增加published-target leading cells的trial；结果不是selector tie-break造成的。

相对TimeAlign Table 6的published three-run means，当前single-seed test-tuned row的cell-macro MSE/MAE分别低2.447%/0.548%。在五个published models加本方法的逐cellranking中，MSE为15个rank 1、11个rank 2、2个rank 3；MAE为15个rank 1、13个rank 2。该比较仍是published context，不是matched attribution，也不能消除run-aggregation差异。

## 4. Targeted weakness analysis

Joint selector只替换两个dataset profiles：

- ETTm2：`H4J rank64` → `H4K rank64_dropout2`。Four-H mean MSE/MAE改善0.1267%/0.1120%，但仍为0/8 leading cells。H96--H336的MSE gaps仍为4.35%、3.80%、2.94%，MAE gaps为2.68%、1.95%、1.69%；H720最接近，MSE gap=0.365%，MAE gap=0.010%。
- Weather：`H4J timealign_dropout3` → `H4K current_lr5e5_patch24_dropout0`。Four-H mean MSE/MAE改善0.0236%/0.1107%，仍为2/8 leading cells；领先项仍是H720 MSE/MAE。H96--H336的MSE gaps为1.70%、1.11%、0.75%，MAE gaps为2.94%、2.74%、1.56%。

H720全局仍为MSE 1/7、MAE 3/7、combined 4/14，未达到局部6/14目标。ETTm2>=4/8与Weather>=6/8两个局部gates同样失败。Solar `patch2_lr3e4`虽降低four-H mean MSE，但MAE退化并未进入joint selection；ECL、ETTh1、ETTh2和ETTm1均保留此前profiles。

## 5. Four-layer decision and rollback

- `paper_facing_effectiveness=strong_aggregate_competitor_but_30_of_56_gate_fail`；
- `matched_mechanism_attribution=unchanged_pending_Main_II_and_ablations`；
- `internal_mechanism_health=24_of_24_test_and_checkpoint_invariants_pass`；
- `failure_attribution=search_space_performance_shortfall_not_selector_or_numeric_failure`。

H4K产生了连续值上的微小改善，但没有任何新的leading cell，不能宣称达到预注册的per-cell SOTA目标。Rollback=Step 6 HPO search-space design。Automatic H4L=false；若继续，必须另行冻结更有区分度的dataset-level search contract，优先围绕ETTm2 `rank64_dropout2`、Weather short-horizon gaps与H720缺口，而不是重复当前邻域或改用per-horizon profiles。

Decision=`H4K_complete_continuous_improvement_no_new_leads_gate_fail_return_step6`。
