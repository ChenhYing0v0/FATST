# ISCF-BSCA-MAIN-v1 HPO Tooling

## 1. Role and Boundary

本工具链服务paper-facing `ISCF-BSCA-MAIN-v1`超参数调优，不修改冻结的
ISCF-BSCA architecture forward path，也不替换exact `ISCF-BSCA-v1` ablation
anchor。H0 dataset audit和H1 two-anchor matrix已经完成；H2已冻结为每dataset
三个additional profiles，因此每dataset共有五个total trials。

核心选择顺序固定为：

```text
train one trial
-> validation four-H mean MSE selects checkpoint
-> freeze checkpoint hash
-> complete all frozen trials
-> official-test four-H mean MSE ranks profiles per dataset
-> freeze one profile shared by H={96,192,336,720}
```

Official test不得选择epoch、checkpoint、seed或per-horizon profile。

## 2. Dataset Support

`baselines/timealign_official/train_repo.py`的`OFFICIAL_PRESETS`新增：

- ECL：`Dataset_Custom`，321 channels，hourly；
- Solar：`Dataset_Solar`，137 channels；
- Exchange：`Dataset_Custom`，8 channels，daily。

这些preset只提供可构造的source-informed defaults；最终main profile仍由HPO
contract决定。`Dataset_Solar.__getitem__`同时修正`seq_y_mark`长度，使其与
`seq_y`而不是`seq_x`一致。

`scripts/audit_iscf_bsca_paper_datasets.py`在任何training前记录：

- exact file path、SHA256、bytes、rows和channels；
- CSV的`date`/`OT` contract、timestamp monotonicity、duplicates和cadence；
- NaN/Inf及constant channels；
- chronological split boundaries与train-only scaler fit rows；
- 可选train/validation loader batch shapes。

该audit不构造test loader，也不计算test metrics。

## 3. H1/H2 Tensor and Training Flow

每个job从历史窗口

$$
X \in \mathbb{R}^{B\times720\times C}
$$

开始。`timealign-token-mlp` encoder由profile中的`patch_num`、`d_model`、
`d_ff`、`dropout`和`layer_norm`控制；`siff-independent-scope-control`
readout沿用五个sharing scopes和dataset/job指定的`mode_rank`。最终一次生成

$$
\hat Y \in \mathbb{R}^{B\times720\times C},
$$

再以full-crop读取H96、H192、H336和H720。

H1包含8 datasets × 2 source-audited anchors：

- `h1_conservative_anchor`：原5 datasets复用exact-v1 natural profile；新3
  datasets使用bounded conservative start；
- `h1_timealign_source_prior`：使用对应TimeAlign official encoder/profile
  setting；Exchange使用已披露的ETTh1-derived bootstrap。

所有job固定seed2021、BSCA objective、five scopes、canonical partition和
joint end-to-end training。差异只来自config中显式记录的hyperparameters。

H2的24个jobs以H1 job作为`base_trial_id`，再通过显式`overrides`冻结局部邻域。
Runner在启动时materialize完整job；profile hash基于materialized job，而不是仅对
override计算。H2允许的变化只有lookback、patch count、encoder capacity、dropout、
learning rate和固定的30-epoch/patience-7预算。Architecture invariants保持不变。

## 4. Provenance and Optimizer

`train_repo.py`新增以下provenance arguments，它们进入
`effective_config.json`：

- `hpo_trial_id`；
- `hpo_profile_id`；
- `hpo_profile_hash`；
- `hpo_config_hash`；
- `hpo_search_space_hash`。

`AdamW`原先隐式使用PyTorch默认`weight_decay=0.01`。现在通过
`--weight-decay`显式连接到optimizer并记录；默认值仍为`0.01`，因此旧命令语义
不变。

## 5. Runner and Artifacts

`scripts/remote/run_iscf_bsca_main_v1_hpo.sh`同时支持H1和H2 config：

- `MODE=dry-run`：输出phase-specific frozen manifest和hash；
- `MODE=data-audit`：执行新三dataset H0 audit；
- `MODE=resource-smoke`：执行two-batch construction/resource canary；
- `MODE=train`：执行phase-specific full-budget train/validation；
- `MODE=status`：统计完整job。

H2通过`scripts/remote/run_iscf_bsca_main_v1_hpo_h2.sh`固定config和repo-external
output root。`CANARY_ONLY=1`在H1为6 jobs，在H2为9 jobs。Runner采用global
shared queue，空闲GPU领取剩余最长job，不按dataset或arm静态配对。每个H2 job的
`max_epochs`和`early_stopping_patience`来自materialized config，不再由runner
硬编码。

每个trial目录至少包含：

- `checkpoint.pt`；
- `training_log.csv`；
- `metrics_by_target_horizon.csv`；
- `effective_config.json`；
- `initialization_contract.json`；
- `model_diagnostics.json`。

## 6. Analysis and Selection

`scripts/analyze_iscf_bsca_main_v1_hpo.py`能解析H1完整jobs或H2
`base_trial_id + overrides` profiles，只接受完整standard horizons，输出：

- `trial_ledger.jsonl`；
- `trial_scorecard.csv`；
- `profile_aggregates.csv`；
- `hpo_completeness.json`。

只有使用`--require-test`且全部frozen trials都有完整test scorecard时，才继续
生成`profile_ranking.csv`和`selected_profiles.json`。排序依次使用：

1. lower four-H mean official-test MSE；
2. lower four-H mean validation MSE；
3. lower trainable parameter count；
4. lexical `profile_id`。

MAE完整报告，但不参与默认selector。

## 7. Code-Theory Consistency

Intended theory是：main-table model应在保持ISCF-BSCA computation contract不变
的前提下，通过dataset-level encoder/optimization profile释放架构性能。代码通过
固定readout mode、objective、scopes、partition和training path，仅搜索显式
hyperparameters来实现这一边界。

仍然只是proxy的部分：

- H1只有两个anchors，不能宣称已经找到最优profile；
- TimeAlign设置只是source prior，不保证适配ISCF-BSCA；
- 新dataset的frequency/source identity必须由remote H0 file audit确认；
- test-tuned结果不构成untouched-holdout generalization estimate。

若H2出现OOM/NaN/Inf、validation checkpoint provenance不完整、base-config hash
不一致，或trial hash不一致，则H2 gate失败并回到local protocol/resource repair。
H2现已24/24完成，逐checkpoint artifact、selector和SHA256 audit通过。H1与H2合并为40-row frozen manifest后才进入official-test ranking。

## 8. Complete Test-Audit Path

`scripts/build_iscf_bsca_main_v1_hpo_test_manifest.py`把H1的16个和H2的24个validation-selected checkpoints冻结为一个manifest。每一行记录phase、dataset、trial/profile、best epoch、validation four-H mean MSE、parameter count、test前checkpoint SHA256、只读training artifact目录和独立test artifact目录。

`scripts/evaluate_stage_c_pcsd_cf_checkpoint.py`在test模式下先读取并验证authorization，再创建artifact目录、加载model或构造test loader。未授权config必须在任何test artifact写入和loader access之前fail closed。

`scripts/remote/run_iscf_bsca_main_v1_hpo_test_audit.sh`提供：

- `MODE=dry-run`：只验证config/manifest hash并输出40 jobs；
- `MODE=preflight`：验证40个training artifacts、validation四-H完整性、provenance和checkpoint hash，不构造test loader；
- `MODE=test`：在三GPU global queue上读取40个checkpoint，输出独立test artifacts，并核对test前后hash；
- `MODE=status`：只统计具有720-row metrics、pass invariant和diagnostic NPZ的完整jobs。

任一checkpoint mutation、evaluator failure或artifact invariant failure都会写入`ABORT` sentinel，阻止worker领取新job。初次启动要求core test artifacts为0；恢复必须显式设置`ALLOW_RESUME=1`。

`scripts/analyze_iscf_bsca_main_v1_hpo_test_audit.py`仅在40/40 trials和160/160 standard-horizon cells均完整时生成profile ranking。输出包含所有trial scorecard、aggregate、ranking、selected profile和test audit ledger；partial matrix不会产生选择结果。

该runner/analyzer现同时支持config显式给出的非Cartesian manifest大小与`profiles_per_dataset`。ECL/Solar H3A使用9-row manifest（ECL 1、Solar 8）和36个standard-horizon cells，复用相同的authorization、atomic publication、provenance与checkpoint immutability gates；H1/H2原40-row contract保持不变。

ECL和Solar后续允许test-tuned扩展training budget，但仍使用validation选择每个trial的checkpoint，并按official-test four-H aggregate选择一个dataset-level shared profile。该后续搜索必须使用新的candidate version和完整保留的trial ledger。

## 9. ECL/Solar H3A

`configs/iscf_bsca_main_v1_hpo_ecl_solar_h3a.json`是在H1/H2完整official-test scorecard后建立的test-informed candidate version。ECL当前aggregate已优于TimeAlign published target，只运行一个exact test-best profile的45-epoch extension。Solar当前test-best仍有2.17%差距，八个one-factor profiles分别检查expanded budget、learning rate、dropout、weight decay、decoder rank、effective batch和patch granularity。

`scripts/remote/run_iscf_bsca_main_v1_hpo.sh`现在优先读取`remote_<PHASE>_training_authorized`，因此H3A仍复用同一个global queue、artifact contract与training path。`scripts/remote/run_iscf_bsca_main_v1_hpo_ecl_solar_h3a.sh`只固定H3A config和repo-external output root，不改变runner逻辑。

H3A training期间仍为`official_test_mode=false`、`final_evaluation_split=val`。9/9 checkpoints完成后不做耗时validation profile ranking，只检查selector/artifact/hash完整性，随后为H3A另行冻结test manifest并直接执行完整four-H official-test audit。

## 10. Main II H5A Result Selector

`scripts/analyze_iscf_bsca_main_v1_h5a_result.py`只在generic formal-test analyzer已经确认
H5A `48/48` checkpoints与`192/192` standard-horizon rows完整后运行。输入仍是每个
dataset/profile在H96、H192、H336、H720上的MSE/MAE；dense H1--720 diagnostics不进入
profile selection。

Analyzer将H1--H4M的189个历史profiles与H5A的48个新profiles合并，但只对
ETTh1、ECL、Solar执行重选。每个profile的8个metric cells先按统一three-decimal
`ROUND_HALF_UP`与冻结Main II七个external systems比较。Dataset级候选必须同时满足：

$$
\overline{\mathrm{MSE}}_{p}\leq1.005\,
\overline{\mathrm{MSE}}_{\mathrm{current}},\qquad
\overline{\mathrm{MAE}}_{p}\leq1.005\,
\overline{\mathrm{MAE}}_{\mathrm{current}}.
$$

Eligible profiles依次按best-cell数量、top-2-cell数量、相对external best的mean
normalized regret、four-H mean MSE/MAE、validation selector、parameter count和
profile ID排序。如果第一名没有严格增加该dataset的best-cell数量，则显式保留当前
profile。该逻辑确保一个dataset最终只有一个profile服务四个horizons，不允许per-H、
per-metric、per-seed或per-cell rescue。

输出包括完整`all_profile_main_ii_ranking.csv`、三dataset共12行的
`selected_profile_scorecard.csv`和`h5a_selection_result.json`。JSON单独报告
ETTh1/ECL/Solar best-cell gates、Solar MAE coverage、target/global projected counts与
mean guard。Analyzer不会修改Main I或Main II table；表格替换仍需单独授权。

Code-theory边界是：H5A只测试冻结ISCF-BSCA architecture内的dataset-level
hyperparameter choice能否改善Main II best-cell coverage。即使gate通过，也只能支持
test-tuned benchmark性能，不建立untouched-holdout或mechanism-attribution结论；若完整
matrix、checkpoint provenance或target artifact hash不一致，分析会fail closed而不产生
selector结果。

## 11. ETTh1 H5B Artifact Gate and Frozen Selector

H5B的training artifact checker与manifest builder把36个ETTh1 trials固定为一个完整
formal-test block。`scripts/check_iscf_bsca_main_v1_h5b_training_artifacts.py`逐trial核对
checkpoint、validation four-H selector、effective config、numeric health和training-stage
test=0；`scripts/build_iscf_bsca_main_v1_h5b_test_manifest.py`随后记录36个validation-selected
checkpoint的pre-test SHA256。只有36个hash全部唯一且manifest hash匹配时，H5B formal
runner才允许构造test loader。

`scripts/remote/run_iscf_bsca_main_v1_hpo_etth1_h5b_test_audit.sh`复用generic three-GPU
atomic test runner，但固定H5B config、manifest和repo-external output root。一次完整execution
生成36份720-row dense metrics、invariant和diagnostic artifact；
`scripts/check_iscf_bsca_main_v1_hpo_etth1_h5b_test_audit.py`要求36/36 jobs、144/144
standard rows、0 temporary files和checkpoint hash immutability同时通过。

`scripts/analyze_iscf_bsca_main_v1_h5b_result.py`只在上述complete gate之后运行。其输入列
及含义为：

- `test_mse` / `test_mae`：checkpoint在指定standard horizon prefix上的official-test metric；
- `test_mean_mse_4h` / `test_mean_mae_4h`：同一profile四个standard horizons的算术平均；
- `main_i_best` / `main_ii_best`：按冻结external comparison surface和three-decimal
  `ROUND_HALF_UP`计算的8个metric cells中的best数量；
- `main_ii_top2`：同一口径下进入best或second的cell数量；
- `mean_*_ratio_to_h5a`：候选four-H mean除以当前H5A ETTh1 mean；
- `eligible`：MSE和MAE ratio都不超过`1.003`；
- `eligible_rank`：eligible profiles按Main II best、Main I best、Main II top-2、mean
  MSE/MAE、validation score、parameter count及profile ID依次排序。

Selector显式把H5A current profile加入候选池；只有H5B winner严格提高Main II best-cell
数量才替换current。最终选择`h5b_seq640_p20`，Main II ETTh1 best由2/8提高到4/8。
Analyzer只写ranking、selected scorecard和decision JSON，不修改paper tables。

Code-theory边界是：H5B验证expanded context/patch profile能否改善冻结ISCF-BSCA的
paper-facing ETTh1 performance，不承担mechanism attribution。Official test只选择一个
dataset-level profile，不选择epoch、checkpoint、seed或单独horizon；single-seed和
test-informed属性必须随结果披露。

## 12. ETTh1 H5C Refined-Interaction Matrix

H5C复用generic HPO runner，只新增`configs/iscf_bsca_main_v1_hpo_etth1_h5c.json`和
wrapper `scripts/remote/run_iscf_bsca_main_v1_hpo_etth1_h5c.sh`。Base profile是H5B
selected `L640/p20`；54个jobs均通过显式overrides materialize完整effective config，runner
仍输出trial/profile/config/search-space hashes并保持official-test mode关闭。

`scripts/check_iscf_bsca_main_v1_hpo_etth1_h5c.py`完成三类检查：首先核对Main I/Main II
comparison CSV及H5B evidence hashes；其次materialize H1/H2/H4J/H4K/H5A/H5B共61个
ETTh1 historical jobs，并以dataset、context、patch、width、regularization、optimizer、
rank、normalization和budget组成effective fingerprint，要求54个H5C profiles与历史集合
零重复；最后检查54-job group counts、`seq_len % patch_num == 0`、fixed capacity/effective
batch、training test=0与generic runner dry-run。

H5C没有新的tensor path：输入仍为`[B,L,C]`，encoder生成history-conditioned states，
five-scope decoder一次输出`[B,720,C]`，validation从同一field裁剪H96/H192/H336/H720。
变化只在`L`、patch geometry、LR、dropout、weight decay与mode rank。Code-theory边界是
搜索H5B winner附近的hyperparameter interactions；任何性能提升仍是test-tuned benchmark
evidence，不是new architecture或BSCA mechanism attribution。

H5C formal-test path复用H5B工具但将hard-coded trial count改为config-driven：training
artifact checker读取`matrix.expected_training_runs`与phase-specific authorization key；manifest
builder显式接收`--phase H5C --expected-trials 54`；formal-contract checker从config读取runs、
cells、phase、guard reference和next-extension key。H5B原36/144 contract有独立regression
check，确保向后兼容。H5C test runner仍复用generic atomic queue，目标artifact为每profile
720-row dense metrics、invariant JSON和diagnostic NPZ，216 standard rows只从完整dense artifacts
聚合得到。

`scripts/analyze_iscf_bsca_main_v1_h5c_result.py`在54/54与216/216 completeness gate通过后，
从冻结Main I/Main II comparison surfaces计算共同三位小数口径的best/top-2 counts。它先用
H5B mean MSE/MAE `1.002x`双guard过滤，再按Main II best、Main I best、Main II top-2、
mean MSE/MAE、validation score、参数量和ID排序。输出包含全部profile ranking、best H5C
scorecard、最终保留profile scorecard及machine decision。若H5C best-cell数没有严格超过H5B，
selector必须保留H5B，不能因事后mean improvement改写primary objective，也不修改paper tables。

## 13. ETTh1 H5D History Audit and Interaction Matrix

`scripts/analyze_iscf_bsca_main_v1_etth1_hpo_history.py`materialize H1/H2/H4J/H4K/
H5A/H5B/H5C的ETTh1 configs，并按trial ID连接六套complete test scorecards。它从冻结Main II
surface计算三位小数best thresholds，为115个profiles输出hyperparameters、8个metric cells、
four-H means、best-cell count及H192/H336 normalized gap。任何missing/duplicate trial或cell都会
hard fail；输出只作H5D design evidence，不改变historical selection。

H5D仍复用generic runner，tensor path不变：`[B,L,C] -> history encoder -> five-scope decoder
-> [B,720,C] -> four prefix crops`。48个profiles只改变actual batch size、LR、p19/p21
geometry和mode rank，且都固定dropout0、width32、weight decay0.01、LayerNorm on与120/24
budget。`scripts/check_iscf_bsca_main_v1_hpo_etth1_h5d.py`核验115个historical fingerprints、
48个new fingerprints、six group counts、frozen hashes、divisible geometry与test=0 dry-run。
Formal test flag明确为false，防止training完成后自动访问test或修改paper tables。
