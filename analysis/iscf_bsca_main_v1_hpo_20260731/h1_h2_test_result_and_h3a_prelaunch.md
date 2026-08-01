# ISCF-BSCA-MAIN-v1 H1/H2 Test Result and ECL/Solar H3A Prelaunch

## 0. Decision summary

| Field | Value |
| --- | --- |
| `current_step` | H1/H2 Step 9 official-test ranking complete；test-informed H3A Step 6--8 prelaunch |
| `source_candidate_version` | `ISCF-BSCA-MAIN-v1-hpo-h1-h2-20260801` |
| `test_matrix` | 40/40 checkpoints；160/160 standard-horizon cells |
| `primary_seed` | 2021 |
| `ECL_decision` | aggregate target passed；one exact-profile budget-extension trial only |
| `Solar_decision` | 2.17% aggregate gap remains；eight-trial orthogonal H3A |
| `H3A_candidate_version` | `ISCF-BSCA-MAIN-v1-ecl-solar-h3a-test-informed-20260801` |
| `decision` | launch 9 H3A train/validation jobs，then direct complete official-test audit |

## 1. Problem

H1/H2需要在8 datasets上从每dataset五个profiles中选择一个共享四个horizons的main profile，并重点判断ECL和Solar是否达到TimeAlign ICLR 2026 Table 6的published competitiveness target。用户允许在两者上使用official-test four-H aggregate继续调参并扩展training budget，但仍禁止per-horizon、per-cell、per-seed或metric-specific选择。

## 2. Existence evidence

[Fact] 40/40 checkpoints与160/160 standard-horizon cells完整。Analyzer验证所有dense test CSV为H1--H720、invariant与NPZ完整、candidate/trial/profile/seed provenance一致、test前后checkpoint SHA256一致，errors为空。

### 2.1 Selected profiles over all datasets

| Dataset | Selected trial | Phase | Test mean MSE | Test mean MAE |
| --- | --- | --- | ---: | ---: |
| ETTh1 | `ETTh1__h1_timealign` | H1 | 0.395575 | 0.420327 |
| ETTh2 | `ETTh2__h2_lr5e4` | H2 | 0.307332 | 0.365116 |
| ETTm1 | `ETTm1__h2_table5_capacity` | H2 | 0.330699 | 0.363879 |
| ETTm2 | `ETTm2__h2_hybrid` | H2 | 0.250849 | 0.310389 |
| Weather | `Weather__h1_timealign` | H1 | 0.216574 | 0.249031 |
| ECL | `ECL__h1_timealign` | H1 | 0.151191 | 0.245509 |
| Solar | `Solar__h1_timealign` | H1 | 0.196157 | 0.218267 |
| Exchange | `Exchange__h2_lookback336` | H2 | 0.398836 | 0.425081 |

H2提供了ETTh2、ETTm1、ETTm2和Exchange的test winner；ETTh1、Weather、ECL、Solar由H1 TimeAlign-source profile胜出。这说明validation winner不能替代test-tuned profile selection。

### 2.2 ECL

| Trial | H96 | H192 | H336 | H720 | Mean MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `h1_timealign` | 0.116722 | 0.136832 | 0.155733 | 0.195475 | **0.151191** |
| `h2_dropout3` | 0.117206 | 0.137860 | 0.156526 | 0.193985 | 0.151394 |
| `h2_intermediate_capacity` | 0.116836 | 0.136800 | 0.156202 | 0.196661 | 0.151625 |
| `h2_lookback336` | 0.116795 | 0.137701 | 0.158908 | 0.201286 | 0.153673 |
| `h1_conservative` | 0.130415 | 0.150602 | 0.171957 | 0.219656 | 0.168157 |

TimeAlign published coherent row为MSE `0.126/0.143/0.158/0.189`，mean=`0.154`。ISCF-BSCA当前mean低1.82%，并在H96/H192/H336更低；H720仍高0.006475。ECL aggregate SOTA-competitive target已通过，因此H3A只保留一个exact test-best profile的45-epoch extension，用于检查原best epoch 20/20的budget boundary，不扩张ECL search。

### 2.3 Solar

| Trial | H96 | H192 | H336 | H720 | Mean MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `h1_timealign` | 0.180708 | 0.194491 | 0.203222 | 0.206208 | **0.196157** |
| `h2_dropout1` | 0.191974 | 0.205792 | 0.215267 | 0.212102 | 0.206284 |
| `h2_lookback336` | 0.184253 | 0.205108 | 0.217693 | 0.224730 | 0.207946 |
| `h2_ff512` | 0.197823 | 0.212210 | 0.218919 | 0.214718 | 0.210918 |
| `h1_conservative` | 0.182345 | 0.205539 | 0.223864 | 0.232330 | 0.211020 |

TimeAlign published row为MSE `0.172/0.189/0.204/0.202`，mean=`0.192`。当前ISCF-BSCA mean高2.17%，H336低0.000778，但H96/H192/H720仍高。因此Solar尚未达到target。

[Strong Evidence] Solar的validation winner `h2_dropout1`在test上比`h1_timealign`差5.16%；`d_ff=512`和`seq_len=336`也分别差7.53%和6.01%。这些方向关闭，不能再按validation趋势组合。H3A从test-best profile出发，搜索budget、lr、stronger regularization、mode rank、effective batch和patch granularity。

## 3. Idea and theory check

H3A不修改ISCF-BSCA architecture family、objective、five scopes、partition、readout mode或inference graph。它只改变同一end-to-end joint-training path的ordinary capacity/optimization hyperparameters。每个trial仍由validation four-H mean MSE选择checkpoint；为了响应用户要求，不做validation profile ranking，9/9 artifacts完整后直接执行H3A official-test four-H scorecard。

[Theory check] ECL只延长exact winner的cosine schedule，必须从头训练而不能覆盖H1 checkpoint。Solar八个profiles为one-factor邻域：budget45、lr3e-4、dropout0.4、weight decay0.05、mode rank64、effective batch16、patch2和patch4。它们不是Cartesian search，避免在一次test exposure后无限追加组合。

## 4. H3A design and resource schedule

- ECL：1 trial；Solar：8 trials；total=9；seed2021；
- all trials：max_epochs=45，patience=10，from scratch；
- global queue：ECL budget extension first，随后Solar八个profiles；
- three GPUs；full launch前执行all-job two-batch resource smoke；
- output root：`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/ecl_solar_h3a`；
- estimated wall clock：约4--6小时，实时以logs为准；
- training complete gate：9/9 artifacts、four-H validation、finite、checkpoint hash；
- direct test gate：训练完整后冻结9-row checkpoint manifest，直接测试所有9个profiles并保留全部负结果。

## 5. Success, failure and rollback gates

- ECL：若budget45低于0.151191，更新shared profile；否则保留H1 winner并停止ECL HPO。
- Solar：若H3A winner低于0.192，达到当前coherent published target；若仍高于0.192，依据H3A one-factor response决定是否冻结一次H3B interaction batch。
- H3B如启用，必须是新的test-informed version、完整冻结，并以不超过4个profiles为默认上限；不得按单horizon组合。
- 任一NaN/Inf/OOM/provenance/hash failure阻断该batch test；infrastructure failure只允许相同profile/hash重跑。
- validation改善但test反转必须记录；不得删除negative trial。
- 达到target后停止该dataset的HPO，不为追求更好数字无限扩张。

## 6. Narrative gate

`passed_as_test_informed_same_architecture_hpo`。该工作只兑现main-table performance，不构成新mechanism contribution。论文必须披露test-tuned/test-informed、single-seed与TimeAlign 3-run published mean的差异。

## 7. Effectiveness gate

- ECL=`aggregate_target_pass_single_seed_test_tuned`；
- Solar=`competitive_but_target_gap_2.17_percent_H3A_required`；
- all-eight main candidate=`performance_partial_pass_pending_targeted_H3A_and_baseline_completion`。

这些结果是paper-facing effectiveness evidence，不是matched mechanism attribution；ablation和transfer仍需独立完成。

## 8. Failure attribution

Solar当前gap归为`hyperparameter_optimization_incomplete_with_validation_test_reversal`，不是architecture failure。ECL当前无failure。若H3A/H3B均不能关闭Solar gap，回到Step 6重估profile freedom与main claim，不新增paper-core mechanism，也不使用per-cell profile掩盖失败。

## 9. Artifacts and decision

- result root：`test_audit_result/`；
- completeness：`test_audit_result/test_audit_completeness.json`；
- all trials：`test_audit_result/all_trial_scorecard.csv`；
- ranking：`test_audit_result/profile_ranking.csv`；
- selected profiles：`test_audit_result/selected_profiles.json`；
- H3A config：`configs/iscf_bsca_main_v1_hpo_ecl_solar_h3a.json`。

Decision=`H1_H2_test_complete_ECL_target_pass_Solar_H3A_authorized_prelaunch`。
