# SC1-SIFF-v3-TSAF Step7B Prelaunch Audit

## 1. Long-stage record

| Field | Content |
| --- | --- |
| `current_step` | Step7B prelaunch complete；等待独立remote/test authorization |
| `problem` | 在不恢复requested-H或sample-conditioned router的条件下，TSAF能否形成可归因、可复现的formal matrix？ |
| `existence_evidence` | SIFF-v2有可观performance但direct policy attribution不足；TSAF Step7A 26/26 local pass |
| `idea` | future-coordinate × ordered-log-scale shared allocation field；history只进入五个SIFF arms |
| `theory_check` | Bayes boundary不把requested horizon作为额外信息；TSAF改变finite-capacity allocation bias而非Bayes target |
| `design` | 9 effective arms × 5 datasets；4个历史references复用，5个new arms from-scratch joint training |
| `narrative_gate` | Step4-6 conditional pass；single SIFF/TSAF decoder contribution，不把equal-skill写成第二contribution |
| `effectiveness_gate` | 尚未执行；必须使用完整official-test 45-run/180-cell effective matrix |
| `artifacts` | config、25-job runner、reference manifest、initialization/gradient gates、four-layer analyzer |
| `decision` | `step7b_prelaunch_pass_waiting_remote_and_test_authorization` |

## 2. Matrix freeze

正式effective matrix是9 arms × 5 datasets × seed2021 = 45 runs。由于4个历史arms满足相同profile、seed、
four-horizon validation selector与checkpoint hash契约，实际新增训练量为25 runs：

| Effective arm | Source | Policy information access | Role |
| --- | --- | --- | --- |
| `a6_full` | reused | n/a | carrier control |
| `a6_measure` | reused | n/a | strong objective control |
| `siff_v2_direct_parent` | reused `siff_equal` | history + target coordinate | immutable parent |
| `pcsd_equal` | reused | history + target coordinate；no SIFF scale coordinate | source-family control |
| `siff_categorical_target_only` | new | target coordinate only | no ordered scale geometry |
| `tsaf` | new | target coordinate × ordered scale | candidate |
| `tsaf_permuted_scale` | new | target coordinate × permuted scale | scale semantics control |
| `tsaf_no_target_global` | new | scale only；target-shared | future-coordinate control |
| `siff_independent_target_only` | new | target coordinate only；independent scale fields | shared-field control |

历史`SIFF independent`不能复用，因为它使用`direct` policy，包含history-conditioned allocation；用它替代
`independent target-only`会把information-access差异混入shared-field attribution。

formal scorecard固定H96/H192/H336/H720、MSE/MAE。25个new runs对应100个new test cells，加上20个reused
references对应80 cells，effective matrix共180 cells。partial reporting禁止。

## 3. Reference reuse audit

source audit为
`analysis/stage_c_siff_equal_attribution_step9_20260718/primary/run_audit.csv`，SHA256为
`64d89c07ee5151b6dcdd50d6eee60fad15f5df03fdbf6a68fb4847772f08e366`。

- local lite artifacts：20/20含test metrics、invariants、effective config与initialization contract；
- remote checkpoints：20/20存在；
- remote逐文件SHA256与frozen run audit：20/20一致；
- profile SHA256：`80912741...278990a`；
- checkpoint selector：validation mean MSE over `{96,192,336,720}`；
- 这些references不重新训练，也不作frozen replacement；它们只是完整end-to-end historical runs的复用。

## 4. Capacity and initialization audit

旧independent matched ranks是为history-conditioned direct parent计算的，不能直接继承。Step7B按TSAF
`active_forward_parameters`重新求nearest integer rank：

| Dataset | Independent rank | TSAF active params | Independent active params | Relative gap |
| --- | ---: | ---: | ---: | ---: |
| Weather | 115 | 1,982,097 | 1,975,025 | 0.3568% |
| ETTm1 | 115 | 1,954,289 | 1,947,217 | 0.3619% |
| ETTh1 | 109 | 3,553,041 | 3,546,425 | 0.1862% |
| ETTh2 | 115 | 1,982,097 | 1,975,025 | 0.3568% |
| ETTm2 | 106 | 6,697,809 | 6,705,245 | 0.1110% |

五个new arms在每个dataset使用相同seed时encoder initialization hash只有1个；TSAF、permuted与global的
trainable parameter initialization hash也只有1个。permuted semantics只通过buffer改变scale mapping，global只在forward
移除target variation。

## 5. Early-gradient and CLI gate

machine gate结果为15/15 cases、10/10 categories：

- 25/25 CLI jobs解析为冻结的readout/policy/objective/rank；
- 5/5 new arms在first backward都有finite/nonzero SIFF field与policy-output gradient；
- zero-output initialization完成一次optimizer step后，5/5 policy input path在second backward均为finite/nonzero；
- 25-job runner通过`bash -n`与dry-run；
- 当前config下normal launch exit code 3；
- evaluator synthetic smoke与four-layer analyzer synthetic smoke均通过；
- 未读取dataset、未训练、未访问official test。

该two-step检查只证明gradient path可达，不证明optimization会学习到有用allocation。

## 6. Statistics and artifact semantics

formal analyzer预注册以下统计量：

1. cell gain：$100(1-M_{candidate}/M_{reference})$；正值表示candidate更好；
2. macro gain：20个dataset × horizon cells的cell gain算术平均，不是先聚合metric再取比；
3. dataset win：该dataset四个horizons的mean gain大于0；
4. horizon win：该horizon五个datasets的mean gain大于0；
5. cell win：20个cells中gain大于0的数量；
6. arm diversity：`probe_arms [N,S,T]`各arm pair RMSE除以arms RMS后的均值；
7. target-scale surface std：`policy_row_bin_usage [N,Bin,S]`先对rows取均值，再对bin × scale求std；
8. scale-order sensitivity：candidate与permuted的mean allocation surface之NRMSE；
9. allocation entropy：scope distribution entropy除以$\log S$；
10. scale component contribution：nonconstant `scale_component_contribution[:,1]` RMS除以fused RMS；
11. request invariance：evaluator记录full-domain output与requested prefix crop的maximum absolute gap。

训练artifacts必须包含checkpoint、training log、validation metrics、effective config、initialization contract与model
diagnostics。formal test另需test metrics、test invariants与diagnostic NPZ；test前后checkpoint SHA256必须不变。

## 7. Validation/test roles and gates

- validation只用于20 epoch内的early stopping与four-horizon mean-MSE checkpoint selection；
- validation不得pass/reject TSAF，也不得选择dataset/horizon-specific variant；
- official test在未来另行授权后，才作为`paper_facing_effectiveness` primary gate；
- 当前candidate已`test_informed=true`，完整negative cells必须报告；
- seed2022/2023 confirmation仍held。

primary effectiveness要求TSAF相对`a6_measure`与parent分别满足：MSE macro至少+0.3%、dataset wins至少3/5、
horizon wins至少3/4、cell wins至少11/20、MAE macro非负。categorical、permuted、global三个matched controls分别要求
MSE macro严格为正且dataset wins至少3/5。shared field相对independent至少macro严格为正；达到+0.3%才可claim
strict superiority，否则最多claim compact structured bias。

## 8. Failure attribution

- candidate与categorical均低于parent：`history_free_allocation_hypothesis_false`，rollback Step2/4；
- categorical提升而TSAF不提升：`scale_field_allocation_design_wrong`，只关闭exact TSAF-v1；
- TSAF超过parent但不超过A6_MEASURE：`performance_partial_pass`，只能作为SIFF ablation；
- permuted/global解释gain：`capacity_or_coordinate_control_explains`，rollback Step4/6；
- numeric/gradient/checkpoint异常：`optimization_or_numeric_pathology`，修复Step7 exact protocol，不作方向级拒绝。

禁止用frozen replacement作方向拒绝，也禁止回到CCSF、region/covariance/temperature、seed/width/rank/readout sweep作
post-hoc rescue。

## 9. Read-only remote preflight

2026-07-21只读检查：

- remote repo：`/home/yingch/projects/FATST`，branch `main`；
- remote worktree仍有3份历史analysis CSV修改，与本次路径不重叠；不得清理、stash或覆盖；
- conda `moe`：torch `2.9.0+cu128`，CUDA available；dataset root存在；
- GPU 0/1/2均为RTX 3090，memory used 18 MiB、free 24107 MiB、utilization 0%；
- 未发现active `train_repo.py`或TSAF runner；
- parent root共有50 checkpoints，其中本次4个reference arms为20/20且hash一致。

本步骤没有`git pull`、resource-batch smoke、remote training或official test。后续若用户另行授权，必须先提交推送、remote
`git pull --ff-only`，重新读取GPU状态，再执行两arm resource smoke；remote worktree的三份历史修改必须原样保留。

## 10. Decision

[Strong Evidence] Step7B production/prelaunch contract完整：15/15 local cases、20/20 reference hashes、25个new jobs、
capacity/initialization/gradient/runner/analyzer边界均通过。

[Fact] 这不是performance结果，也没有形成`passed_core_candidate`。当前权限仍为：

`Step7B prelaunch=true / remote training=false / official test=false / confirmation=false`。

[Decision] `SC1-SIFF-v3-TSAF-v1`进入`step7b_prelaunch_pass_waiting_remote_and_test_authorization`；下一步只能在独立
授权后执行remote resource smoke与25-run seed2021 training。不得将GPU空闲或synthetic smoke写成method evidence。
