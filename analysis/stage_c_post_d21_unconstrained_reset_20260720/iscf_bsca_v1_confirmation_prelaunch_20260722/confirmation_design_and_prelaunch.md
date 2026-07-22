# ISCF-BSCA-v1 seeds2022/2023 confirmation design

## 当前结论

`current_step=Step 6 confirmation freeze / Step 7B prelaunch pass`。用户于2026-07-22回复“继续按计划推进实验”，授权上一轮handoff中明确提出的seeds2022/2023 confirmation training与single frozen formal-test matrix。模型、objective、lambda schedule、datasets、checkpoint selector和official-test horizons均不变。

## problem

Seed2021 BSCA相对EQUAL official-test MSE/MAE=`+0.3104%/+0.4902%`，刚好通过原gate，但ETTm2=`-1.7375%`且macro MSE仅比threshold高0.0104 percentage points。当前关键问题不是再设计机制，而是区分`stable objective effect`与`optimizer-seed variance / dataset heterogeneity`。

## existence_evidence

- Seed2021 candidate/control exact paired initialization，20/20 cells与internal health完整。
- FCC已存在seeds2022/2023 × five datasets的10个ISCF-EQUAL checkpoints、validation/test metrics、invariants与diagnostic NPZ；remote completeness audit=10/10。
- 新confirmation只需训练10个BSCA runs，不重训control，不改变test-informed边界。

## idea

在完全冻结的BSCA-v1上改变random seed，测试balanced co-adaptation的effect direction、effect size与dataset heterogeneity。该阶段不是新method，也不增加loss/router。

## theory_check

如果uniform anchor确实稳定改善joint optimization，则在paired seed control下，跨seed macro direction应保持positive；若seed2021收益主要来自optimizer noise，则新增seeds会使macro趋近tie或negative。Initialization hashes必须在每个seed/dataset与对应EQUAL exact paired，才能把差异归因到objective。

## design

- New：seeds2022/2023 × five datasets × BSCA = 10 trainings。
- Reused controls：FCC的10个same-seed ISCF-EQUAL artifacts。
- Effective analysis：seeds2021/2022/2023 × five datasets × H96/H192/H336/H720 = 60 test cells per metric。
- Checkpoint：validation mean MSE over four standard horizons；test不选checkpoint。
- Scheduling：seed2022使用GPU0/1，seed2023使用GPU2；两个Weather首先并行，避免slow dataset串行。
- Formal test：10/10 training+validation diagnostics完整后，执行一次10-run confirmation formal-test matrix；checkpoint SHA before/after必须相等。

## narrative_gate

不重新评估component novelty；该阶段只验证已通过conditional narrative gate的BSCA contribution是否具备three-seed robustness。Generic balancing仍不作为novelty claim。

## effectiveness_gate

### Direction robustness

Three-seed macro MSE/MAE均`>0`；至少2/3 seeds、3/5 datasets、3/4 horizons的mean MSE gain为正。

### Paper-core promotion

除上述条件外，three-seed macro MSE `>=+0.3%`；最差dataset mean MSE gain `>-2%`；ETTm2 mean MSE gain `>=-1%`；artifacts、paired initialization、checkpoint nonmutation与internal health全部通过。

Internal health冻结为candidate entropy `>=0.95`、candidate/reference pairwise-arm-L1 ratio `>=0.5`、candidate oracle headroom `>=10%`。这些diagnostics不能覆盖negative effectiveness。

## failure_attribution

- Promotion pass：`passed_core_candidate_ready_for_paper_consolidation`。
- Direction pass但promotion fail：effect direction稳定，但effect size或heterogeneity不足；保留contribution candidate并收窄claim。
- Direction fail且无pathology：`optimization_variance_or_dataset_heterogeneity_explains`，回Step4收窄claim，不否定fixed ISCF architecture。
- Numeric/protocol pathology：只修复protocol或Step7 implementation，不作direction rejection。

## artifacts

- Config：`configs/stage_c_iscf_bsca_v1_confirmation.json`
- Runner：`scripts/remote/run_stage_c_iscf_bsca_v1_confirmation.sh`
- Analyzer：`scripts/analyze_stage_c_iscf_bsca_confirmation.py`
- Checker：`scripts/check_stage_c_iscf_bsca_confirmation.py`

## decision

Decision=`confirmation_step7b_prelaunch_pass_remote_resource_smoke_next`。Local checker确认10-job dry-run、10/10-before-test hard guard、generic test authorization contract与10 reused references完整；remote launch前仍须commit/push、fast-forward、GPU/process audit与one-job Weather resource smoke。
