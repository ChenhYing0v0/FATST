# ISCF-BSCA Paper-Facing Experiment Consolidation and Prelaunch Gate

## 0. Document Status

| Field | Content |
| --- | --- |
| `protocol_id` | `ISCF-BSCA-PAPER-EXP-v2` |
| `date` | `2026-07-31` |
| `candidate` | ablation anchor=`ISCF-BSCA-v1`；main candidate=`ISCF-BSCA-MAIN-v1` |
| `current_step` | ISCF-BSCA-MAIN-v1 HPO complete；Main I 140-row published block audited |
| `decision` | `HPO_frozen_Main_I_published_140_rows_complete_competitive_not_full_SOTA_request_staged_baseline_authorization` |
| `architecture_search` | false |
| `test_tuned_hpo_project_principle` | authorized and frozen；HPO execution authorized |
| `local_protocol_patch_authorized` | true for ISCF-BSCA-MAIN-v1 HPO tooling |
| `scoped_timealign_exchange_patch` | authorized and implemented locally；unlaunched |
| `remote_training_authorized` | true for frozen H0/H1 and bounded H2 HPO |
| `official_test_hpo_authorized` | true after complete frozen H2 training matrix |
| `formal_test_authorized` | HPO complete；Main I/II baseline formal test=false |
| `machine_readable_protocol` | `configs/iscf_bsca_paper_experiment_protocol.json` |

截至2026-08-02，`ISCF-BSCA-MAIN-v1` H1/H2/H3A/H3B共53个trials与全部formal tests已完成；8 selected checkpoints/32 cells冻结且不再重训。Main I的TimeAlign Table 6目标published block也已形成140/140 rows。当前未启动任何baseline或Main II remote job；其source/protocol patch、training与formal test需重新分级授权。

## A. Current Authoritative v2 Plan

本节是当前权威实验计划；后续`H1–H9`保留2026-07-31首次consolidation的
artifact/hash审计证据，但其中5-dataset main-table范围、未调优
`ISCF-BSCA-v1`进入Main I/II、baseline删减和旧资源计数均已被本节取代。

### A1. Candidate Separation

[Decision] 从本版开始严格区分两个实验身份：

| Identity | Architecture | Hyperparameters | Paper role | Existing result role |
| --- | --- | --- | --- | --- |
| `ISCF-BSCA-v1` | exact frozen | exact confirmation config | 5-dataset core ablation anchor only | Full与`w/o BSCA`已有three-seed evidence可复用 |
| `ISCF-BSCA-MAIN-v1` | same frozen architecture family；no architecture search | official-test tuned, one profile per dataset and shared across four H | Main I、Main II、transfer reference、efficiency、fresh mechanism diagnostics | 8 selected checkpoints/32 cells frozen and reusable；do not retrain |

因此，现有confirmation不能再作为Main I或Main II中的最终ISCF-BSCA行；它只
证明exact ablation anchor上BSCA相对`ISCF-EQUAL`的小幅方向稳定收益。
`ISCF-BSCA-MAIN-v1`的调优不得改变核心architecture claim。项目现固定采用
`test_tuned/test_informed` paper-benchmark原则：允许按dataset使用four-H
official-test aggregate选择profile，但不得按horizon、seed、metric或单个table
cell选择配置。

### A2. Frozen Eight-Dataset Main Contract

Main I和Main II统一扩展为：

```text
ETTh1, ETTh2, ETTm1, ETTm2, Weather, ECL, Solar, Exchange
```

并继续冻结：

- horizons=`{96,192,336,720}`；
- metrics=`MSE, MAE`，所有预注册cell完整报告；
- current primary seed=`2021`；
- optional extension seeds=`{2022,2023}`，仅在single-seed完整矩阵结束且时间允许
  时按完整experiment block补跑；
- unified checkpoint selector=四个standard horizons的mean validation MSE；
- 每个dataset只选一个unified profile，禁止per-horizon hyperparameter tuning；
- validation only用于每个trial的early stopping与checkpoint selection；
- official test用于在冻结search space内按dataset选择hyperparameter profile，
  以及最终paper-facing effectiveness/reporting；
- 每个dataset的profile selector为四个H的mean official-test MSE；MAE完整报告但
  默认不参与selector；
- 所有尝试过的config及完整MSE/MAE必须保留，结果明确标记
  `test_tuned/test_informed`，不得声称untouched holdout；
- 不选择性报告有利dataset、horizon、seed或baseline。

5-dataset ablation仍使用原集合
`ETTh1, ETTh2, ETTm1, ETTm2, Weather`和exact
`ISCF-BSCA-v1` contract，不因Main candidate调优而重跑已完成的Full/Equal
confirmation。

### A3. ISCF-BSCA-MAIN-v1 Test-Tuned HPO

这是当前最高优先级实验工作，但仍需Tier A/B分级授权后执行。

#### A3.1 Search boundary

允许搜索的是训练与encoder/profile超参数，不是architecture search：

- `seq_len/lookback`、patch-related encoder settings、`d_model`、`d_ff`、
  dropout；
- learning rate、weight decay、batch size/gradient accumulation、scheduler；
- frozen architecture内既有loss coefficients的有限候选值；
- dataset-specific profile allowed，horizon-specific profile forbidden。

TimeAlign encoder设置只作为source-audited search prior；不得机械复制上游代码。
其published test结果可用于建立search prior或baseline context，但不能替代本地
trial。每个被采纳的上游设置必须记录source commit、原始dataset/profile、映射到
本仓库tensor contract的理由及被拒绝的差异。

每个HPO trial先由four-H mean validation MSE选择checkpoint，再在official test
上导出四个H的MSE/MAE。每dataset按four-H mean test MSE排序trial并冻结唯一
profile；test不得选择epoch、checkpoint或seed。

#### A3.2 Staged tuning procedure

| Stage | Scope | Selector / gate | Test access |
| --- | --- | --- | --- |
| H0 data/protocol parity | ECL、Solar、Exchange loader、split、channels、scaler、metric与一条comparable baseline | exact metadata audit + local smoke | false |
| H1 anchor establishment | 8 datasets；current profile + source-audited TimeAlign-inspired encoder anchors | finite train/val smoke；new datasets先得到可比较、非病态结果 | false |
| H2 bounded coarse test-tuned search | 8 datasets，seed2021；预先冻结有限search space与budget | validation选择每trial checkpoint；four-H mean official-test MSE选择dataset profile；MAE/numeric health完整记录 | requires separate Tier B2 authorization |
| H3 optional stability confirmation | 每dataset的top-2 profiles，seeds2022/2023 | only if time allows after complete seed2021 matrix；只评估稳定性，不重选profile | false |
| H4 config freeze | 每dataset选唯一profile；冻结effective config、commit、seed、validation checkpoint selector、test HPO selector与hash contract | complete trial ledger；no missing dataset；no horizon-specific selection | test results already disclosed as tuning evidence |
| H5 selected-profile confirmation/reporting | selected profiles × 8 datasets × seed2021 | checkpoint selected from validation only；additional seeds optional | requires Tier B3/C authorization |

“达到SOTA”是本轮HPO的paper-facing性能目标。允许official test在冻结budget内
选择dataset-level profile，但接受标准必须是完整four-H aggregate，而不是针对
某个baseline或有利cell逐项追赶。搜索预算耗尽后应冻结并报告最优profile及全部
negative trials；若仍未达到目标，再决定收窄claim或另立architecture candidate，
不得通过隐去失败trial制造SOTA结论。

#### A3.3 New-dataset readiness

ECL、Solar、Exchange在进入H2前必须各完成：

1. dataset identity、split boundary、channel count、frequency、missing-value与
   scaler audit；
2. current/anchor profile的local smoke；
3. 至少一个official或已核验published protocol下的comparable reference；
4. four-H validation selector与MSE/MAE导出一致性；
5. resource smoke后才冻结HPO budget。

#### A3.4 H1 implementation cursor

2026-07-31用户已明确授权开始按本计划推进`ISCF-BSCA-MAIN-v1` HPO。当前冻结：

- H1=`8 datasets × {conservative anchor, TimeAlign source prior} × seed2021`
  共16 jobs；
- 新dataset canary=`ECL/Solar/Exchange × 2 anchors`共6 jobs；
- H1 training仍由four-H validation mean MSE选择checkpoint，test jobs=0；
- H2总budget上限为每dataset 5 trials（含H1两项），additional profiles必须在
  H1资源结果后冻结；
- H2使用final-budget training，最终被test aggregate选中的seed2021 checkpoint
  直接复用，不另行随机重训。

实现入口：

- `configs/iscf_bsca_main_v1_hpo.json`；
- `scripts/audit_iscf_bsca_paper_datasets.py`；
- `scripts/remote/run_iscf_bsca_main_v1_hpo.sh`；
- `scripts/analyze_iscf_bsca_main_v1_hpo.py`；
- `scripts/check_iscf_bsca_main_v1_hpo.py`。

H0及launch gate随后已执行：新三dataset audit pass、6/6 canary pass、16/16
resource smoke pass。Full H1于commit `7361d9e`启动，remote orchestrator
PID=`545400`，test jobs=0。Execution record：
`analysis/iscf_bsca_main_v1_hpo_20260731/h0_h1_authorization_and_launch.md`。

### A4. Main Results I — SOTA-Oriented Horizon-Specific Table

Main I是论文主表之一，比较一个tuned unified `ISCF-BSCA-MAIN-v1`与多种
horizon-specific systems。baseline按来源分层，避免把published values、
official-native reproduction与本地matched runs混成一种证据。

| Family | Baselines | Evidence route | Current status |
| --- | --- | --- | --- |
| linear / mixing | AMD, TimeMixer, DLinear | TimeMixer/DLinear优先转录TimeAlign Table 6；AMD与缺失dataset cells转official reproduction | `source_audit_required` |
| Transformer-based | SimpleTM, iTransformer, PatchTST | iTransformer/PatchTST优先转录TimeAlign Table 6；SimpleTM与缺失dataset cells转official reproduction | `source_audit_required` |
| recent official-native | TimePerceiver, SRSNet, TimeAlign | TimeAlign优先使用其ICLR 2026 Table 6并复现缺失Exchange；其余使用official fixed-H scripts | `source_patch_required` |
| paper method | ISCF-BSCA-MAIN-v1 | one tuned unified checkpoint per dataset/seed outputs all four H | `reusable`；8/8 selected profiles frozen |

官方metadata校正：

- AMD：AAAI 2025，official code=`https://github.com/TROUBADOUR000/AMD`；
- SimpleTM：ICLR 2025，official code=`https://github.com/vsingh-group/SimpleTM`；
- TimePerceiver：NeurIPS 2025，official code=
  `https://github.com/efficient-learning-lab/TimePerceiver`；
- SRSNet：NeurIPS 2025 Spotlight，official code=
  `https://github.com/decisionintelligence/SRSNet`；
- TimeAlign：ICLR 2026，official code=
  `https://github.com/TROUBADOUR000/TimeAlign`。

Main I的published-result primary source改为TimeAlign ICLR 2026 Table 6，
`PDT_final.pdf`降为secondary transcription cross-check，不再向主表直接供值。
TimeAlign Table 6提供four-H MSE/MAE和Avg.，source datasets为
ETTm1/ETTm2/ETTh1/ETTh2/Weather/Electricity/Traffic/Solar。映射到本论文
8-dataset contract后：

- covered desired datasets=ETTh1、ETTh2、ETTm1、ETTm2、Weather、ECL、Solar；
- missing desired dataset=Exchange；
- covered selected baselines=TimeAlign、TimeMixer、DLinear、iTransformer、
  PatchTST；
- absent selected baselines=AMD、SimpleTM、TimePerceiver、SRSNet。

因此Main I evidence route冻结为：

1. 从TimeAlign Table 6转录上述5 models × 7 datasets × 4 H=140个display
   cells，并进行双人/双脚本式transcription verification；
2. AMD、SimpleTM、TimePerceiver、SRSNet使用各自official repositories在
   8 datasets × 4 H × seed2021复现，共128个fixed-H checkpoints；
3. 对TimeAlign、TimeMixer、DLinear、iTransformer、PatchTST补跑Exchange，
   使用seed2021，共20个fixed-H checkpoints；
4. official reproduction合计148个checkpoints/cells；与140个published cells
   合并为9 baselines × 8 datasets × 4 H=288个Main I baseline display cells。

仓库已有`baselines/timealign_official` source tree及ETT、Weather、ECL、Solar、
Traffic scripts。用户已单独授权并完成
`baselines/timealign_official/scripts/Exchange.sh` local patch：使用custom
loader、8 channels、daily frequency、ETTh1-derived bootstrap settings、
seed2021与four-H loop。脚本已实现但尚未运行；其超参数不是tuned Exchange
profile。

TimeAlign Table 1 caption称input length从`{336,512,720}`搜索，而同页
Implementation Details称look-back grid为`{96,192,336,512,720}`。该差异必须
在Tier A source audit中通过official scripts、supplement或作者配置解析；在解析
前不得把published rows描述为与本地profile完全matched。相比PDT固定`L=96`，
TimeAlign仍是更合适的primary published source，但published rows仍不进入
matched mechanism attribution、local seed variance或本地latency比较。
TimeAlign paper声明每个setting对3个random seeds取平均，而当前official
reproduction只跑seed2021；该差异必须在table note中披露。若时间允许，再将
seeds2022/2023作为完整block扩展，不得只补表现有利的model/dataset/horizon，
且新增seed不能触发hyperparameter reselection。

Main I的主claim边界是“在预注册8-dataset/four-H scorecard上达到
SOTA-competitive或优于所列systems”，具体措辞由完整结果决定；不得预写
“unqualified SOTA”。

### A5. Main Results II — Matched Unified Benchmark

Main II也使用8个数据集和tuned `ISCF-BSCA-MAIN-v1`：

| Role | Arms | Training contract | Status |
| --- | --- | --- | --- |
| paper method | ISCF-BSCA-MAIN-v1 | 8 datasets × seed2021；one checkpoint per dataset | `reusable`；8 checkpoints/32 cells frozen |
| matched unified adaptation | DLinear-Unified, PatchTST-Unified | same data/objective/four-H selector/seeds；one checkpoint per dataset | `source_patch_required` |
| repo-native unified reference | A6_FULL | same four-H selector；existing 5 datasets reusable，3 new datasets retrain | `partially_reusable` |
| native varied-horizon context | ElasTST-native | native selector/protocol，separately labelled | `source_patch_required` |

DLinear-Unified与PatchTST-Unified是matched unified **system benchmark**，不是
exact same-backbone mechanism attribution。external native结果只承担对应
protocol角色。architecture attribution仍由exact ablation和two-backbone
end-to-end transfer完成。

### A6. Ablation, Transfer, Efficiency and Diagnostics

#### Core ablation

- datasets=原5个；
- Full=`ISCF-BSCA-v1` exact config；
- `w/o BSCA=ISCF-EQUAL` exact existing result；
- 其余三个exact with/without controls先按same initialization class、
  objective、selector和seed2021进行end-to-end joint training；
- historical frozen replacement、warm-start与cross-swap只可作diagnostic，
  不作方向级结论。

#### Decoder transfer

- backbones=DLinear-style与PatchTST-style；
- arms=Original、+ISCF、+ISCF-BSCA；
- datasets=原5个，current seed=`2021`，four-H validation selector；
- 使用经同一test-tuned原则选择的backbone-specific profiles，不使用未调优
  `ISCF-BSCA-v1` ablation超参数；
- 所有方向级结果必须end-to-end joint training。

#### Efficiency and mechanism diagnostics

- efficiency主对象改为tuned `ISCF-BSCA-MAIN-v1`与可本地复现的关键baseline；
- measured latency/memory只比较同一hardware/software/input batch contract；
- published-only rows只转录其报告的Params/FLOPs，不能混入本地实测latency；
- tuned main checkpoints必须重新导出scope usage、prediction diversity、
  oracle headroom、gradient access和future-region signatures；
- 现有exact-v1 internal health保留为ablation-anchor evidence，不能替代
  tuned-main diagnostics。

### A7. Revised Cell Manifest and Resource Boundary

不含HPO exploratory runs的deduplicated final-training plan：

| Block | Unique checkpoint slots | Existing reusable metric evidence | New after authorization | Seed-level four-H cells |
| --- | ---: | ---: | ---: | ---: |
| tuned ISCF-BSCA-MAIN-v1 | 8 | 8 | 0 | 32 |
| Main I official reproduction补齐 | 148 | 0 | 148 | 148 |
| DLinear/PatchTST unified | 16 | 0 | 16 | 64 |
| A6_FULL | 8 | 5 | 3 | 32 |
| ElasTST-native | 8 | 0 | 8 | 32 |
| 5-dataset exact ablation：5 arms | 25 | 10 | 15 | 100 |
| transfer incremental +ISCF/+ISCF-BSCA | 20 | 0 | 20 | 80 |
| **Phase-1 total** | **233** | **23** | **210** | **488** |

说明：

- transfer Original与Main II中的DLinear/PatchTST unified checkpoint复用，不
  重复计数；
- TimeAlign Table 6提供140个published display cells；148个缺失model/dataset
  cells由single-seed official reproduction补齐，合计288个baseline display
  cells；
- 现有额外30个seed2022/2023 BSCA/EQUAL/A6 metric-evidence checkpoints保留为
  optional robustness evidence，不计入seed2021 primary matrix，也不重跑；
- HPO run数不计入233。H0/H1完成resource smoke后，必须先冻结每dataset候选数、
  epoch cap、early-stop与总GPU-hour cap，再请求Tier B；
- 不为“凑SOTA”无限扩张search space。

调度仍采用global workload-aware queue，但8数据集priority改为：

```text
Weather / ECL / Solar first
then ETTm1 / ETTh1 / ETTm2
then Exchange / ETTh2
```

实际顺序须由H1 resource smoke校正；每轮填充三张3090，避免slow/fast arm配对
导致空闲。每个HPO stage完成后审计validation checkpoint provenance、完整test
scorecard、numeric health与预算消耗，再决定是否进入下一stage。

### A8. Gates and Rollback

1. **HPO completeness**：8/8 dataset各有唯一selected profile；每trial checkpoint
   完全由four-H validation MSE选择，profile完全由four-H official-test mean MSE
   选择；trial ledger完整、无per-H config、无missing metadata。
2. **Main I completeness**：每个保留row标明`published_transcribed`或
   `official_reproduced`；8×4 MSE/MAE完整，协议差异可见。
3. **Main II completeness**：所有local matched arms在seed2021下8 checkpoints
   与8×4 cells完整；native context不冒充matched attribution。
4. **Ablation/transfer**：full matrix和negative cells完整；frozen replacement
   不进入方向级gate。
5. **SOTA wording**：完成test-tuned search和protocol comparability audit后，才
   按实际wins、macro means决定`SOTA`、`competitive`或negative wording；必须
   同时披露结果为test-tuned，不作untouched-holdout表述。
6. **Seed robustness wording**：three-seed extension未完成前，main/transfer/new
   ablation只能声称single-seed完整矩阵，不能声称cross-seed robustness。
7. **Rollback**：
   - new-dataset data/protocol不一致→回H0，不启动HPO；
   - HPO numeric/OOM不稳定→缩小同一预注册space并回H1/H2，不改architecture；
   - test aggregate排名不稳定→`comparison_unresolved`，不得以单个H/cell决胜；
   - 冻结budget内最优test-tuned profile未达SOTA→完整报告并收窄claim，或另立
     新candidate后重新冻结search contract；
   - matched/ablation/transfer失败→按four-layer failure attribution返回
     Step 4–6，不自动追加loss/router。

### A9. Tiered Authorization After This Revision

- **Tier A — local protocol/source patch**：8-dataset loaders/configs、HPO
  manifest/runner、official baseline adapters、prediction/efficiency export；
- **Tier B1 — remote H0/H1 smoke and comparable anchors**：仅新数据集与资源
  校准；
- **Tier B2 — test-tuned HPO**：按冻结budget执行seed2021 H2；每trial由validation
  选checkpoint并访问official test作profile ranking；H3 additional seeds仅在
  时间允许时另行扩展；
- **Tier B3 — selected-profile confirmation**：H4冻结后才执行；
- **Tier C — complete test-tuned reporting audit**：全部checkpoint/hash、trial
  ledger与matrix完整后，单独授权汇总paper-facing结果。

截至本版，用户已授权ISCF-BSCA-MAIN-v1的Tier A、Tier B1与bounded Tier B2
HPO（包括完整H2 matrix后的official-test profile ranking）。Tier B3
selected-profile confirmation与Tier C final reporting audit仍为`false`。
执行仍必须依次通过local gate、focused commit/push、remote commit/GPU/data
preflight、6-job canary和16-job H1；不得把授权理解为跳过这些gates。

## Historical v1 Consolidation Detail

以下章节保留首次artifact、selector与hash审计，便于追溯。除已审计的existing
checkpoint facts外，其5-dataset main范围、baseline exclusions、Main I/II直接
复用exact-v1和资源计数均已被Section A取代。

## H1. Executive Decision

[Fact] exact `ISCF-BSCA-v1`已经是冻结paper-core candidate，不再进行architecture
search。现有three-seed confirmation只证明BSCA相对same-architecture
`ISCF-EQUAL`有small but directionally robust gain：

- three-seed macro MSE gain=`+0.3541%`；
- three-seed macro MAE gain=`+0.3073%`；
- seed means=`3/3` positive；
- dataset means=`4/5` positive，ETTm2=`-0.6506%`；
- horizon means=`4/4` positive。

[Fact] 上述结果不能替代Main I、Main II、完整core ablation或decoder transfer。
Introduction v0.9中的“优于horizon-specific systems、核心组件有效、decoder可迁移”
仍是provisional claims。

[Decision] paper-facing minimal sufficient matrix冻结为：

1. Main I primary standard：DLinear-Specific、PatchTST-Specific；
2. Main I modern native context：TimePerceiver-native、SRSNet-native；
3. Main II matched unified：DLinear-Unified、PatchTST-Unified，并列现有
   A6_FULL reference；
4. Main II native varied-horizon context：ElasTST-native；
5. core ablation：Full、w/o BSCA以及三个exact end-to-end controls；
6. transfer：DLinear-style与PatchTST-style两类backbone，各比较Original、
   +ISCF、+ISCF-BSCA；
7. efficiency与必要mechanism diagnostics从上述checkpoints和predictions派生，
   不新增无关training arms。

[Decision] iTransformer、TimeMixer和CATS在结果出现前预先排除，不能因未来结果
强弱改变该集合。排除不是对模型质量的否定，而是避免与已保留角色重复并控制
matrix规模。

## H2. Frozen Evaluation Contract

### 2.1 Dataset, horizon, metric and split

- datasets：`ETTh1, ETTh2, ETTm1, ETTm2, Weather`；
- horizons：`96, 192, 336, 720`；
- primary metric=MSE，secondary metric=MAE；每个预注册cell同时报告；
- local claim-critical seeds：`2021, 2022, 2023`；
- external native baselines：保留各自官方seed与native selector，并显式披露；
- validation只用于checkpoint selection、ordinary hyperparameter choice与
  implementation smoke；
- official test只评估结果，不选择epoch、checkpoint、seed、profile、input length
  或dataset-specific设置。

### 2.2 Checkpoint selectors

| Contract | Selector |
| --- | --- |
| local unified / matched / ablation / transfer | mean validation MSE over `{96,192,336,720}` |
| local horizon-specific | each H independently selects minimum validation MSE |
| ElasTST-native | native `val_weighted_ND`；不冒充FATST four-H selector |
| TimePerceiver-native | native per-H validation selector；test access从training epoch移除 |
| SRSNet-native | native per-H validation selector；先完成metric-equivalence |

任何`historical-best-validation-h720-mse` checkpoint即使已有四个test replay，也
不得进入matched table。若需要该architecture，只能在当前selector下end-to-end
重训。

### 2.3 Seed and macro aggregation

Aggregation在结果前冻结：

1. local three-seed arms先保留每个`seed × dataset × horizon`的MSE/MAE；
2. table display cell先在同一`arm × dataset × horizon`内对三个seeds取算术均值，
   seed standard deviation放Appendix；
3. paired local gain在相同seed、dataset、horizon上计算relative gain，再对全部
   60 paired cells等权取macro mean；dataset、horizon与seed win counts从同一组
   paired gains派生；
4. 20个dataset-horizon display cells不按dataset长度、channel数或horizon长度
   加权；
5. external native arm只有一个官方seed，作为point estimate单独展示，不与local
   seeds pooling，也不进入three-seed direction gate；
6. 不报告“best seed”，不以test选择要展示的seed。

### 2.4 Official-test boundary

本项目official test已被多次治理性使用，所有未来formal evaluation必须标记
`test_informed=true`。每次formal test前必须冻结并记录：

- candidate/source commit和effective config；
- dataset profiles、seeds、selector与完整comparison matrix；
- checkpoint hash和`checkpoint_retrained`；
- `test_access_date`、`user_authorization`与`test_role`；
- 完整positive/negative cells和rollback consequence。

## H3. Existing Artifact and Checkpoint Audit

### 3.1 Reuse boundary

本地保留的是audited metrics、hash、effective metadata与remote provenance，
不是checkpoint binaries的完整本地archive：

```text
artifact_binary_local=false
artifact_metrics_local=true
checkpoint_hash_recorded=true
remote_checkpoint_existence_currently_verified=false
reuse_scope=existing_audited_metrics_and_completed_test_cells
```

因此，现有MSE/MAE table cells可以复用；若未来需要重新导出raw predictions、
latency或新增diagnostics，必须先只读确认remote checkpoint仍存在并重新核对hash。

### 3.2 Family-level audit

| Family | Checkpoints | Cells per metric | Selector | Test access | Completeness | Status / allowed use |
| --- | ---: | ---: | --- | --- | --- | --- |
| ISCF-BSCA-v1 | 15 | 60 | four-H mean val MSE | 2026-07-22 | 15/15 missing=0；hash/nonmutation pass；MSE/MAE/internal health齐全 | `reusable`；Full/main/diagnostics |
| ISCF-EQUAL | 15 | 60 | four-H mean val MSE | 2026-07-18, 21, 22 | 15/15 protocol pass；MSE/MAE齐全 | `reusable`；exact `w/o BSCA` |
| A6_FULL | 15 | 60 | four-H mean val MSE | 2026-07-18, 21 | 15/15 protocol pass；MSE/MAE齐全 | `reusable`；unified carrier/reference，not ISCF attribution |
| A6_MEASURE | 5 | 20 | four-H mean val MSE | 2026-07-18 | seed2021 only | 20 existing metric cells reusable as historical reference；primary paper matrix excluded |
| DLinear intro search | 20 | validation only | exploratory visualization contract | no formal test | prediction artifacts complete for figure role；formal hash/source/test manifest不完整 | formal table `excluded_with_reason`；DLinear-Specific必须重训 |
| historical SIFF/PCSD/PCC controls | mixed | mixed | some four-H，some H720-only | historical | diagnostic artifacts exist | core ablation `excluded_with_reason` unless exact identity proven |
| frozen replacement/FRSC/D24 probes | mixed | diagnostic only | frozen/co-adapted | historical | conditional compatibility only | `excluded_with_reason` |
| external native baselines | 0 local reproductions | 0 | native | none in this workstream | source/protocol blockers remain | `source_patch_required` |

### 3.3 Audited checkpoint hashes

Dataset order in every row is
`ETTh1, ETTh2, ETTm1, ETTm2, Weather`.

#### ISCF-BSCA-v1

| Seed | SHA-256 by dataset order |
| --- | --- |
| 2021 | `ebcd73b95f2bfb7c9b5e5dc16920aafbb63964668f7b4670f1327ab7276368ff`; `dff40f86680125244fa0ed1a3eb28d9afa00f23f60304277d97b9d6f60db8d0f`; `9b99248d53cbbe66796a1fa66cd7bb6d8f03ffc7ad54ff5420f143f5812b3a26`; `4eba8291d320e23bf26988c7aa0ebbe864e64fde38ddde14c11696801cbba354`; `fb700182522d778b97cb1f082e309986022a507c61c25106be47e6cbc4df7ed6` |
| 2022 | `0109bf17f57b7fb32c3138adeb138d6e07ae9a19b40f42b6ff2ce6459c0a525e`; `a2d86658dc8eae83b440eb65b1a0592a473434c2d84df946a942c31271cc7d4b`; `88857e9b7ef24558768a8af6de293725cdfafe2c3c69b43fe1f4adf4eb0d5cf8`; `4343366de5772dc0951ee846290f10db74cd5bdca6461132a01de9198ac55185`; `da0fe15f8a93347855ca9d5d3db1df766d61837550e3327a056381477cebfc86` |
| 2023 | `56643bd0b13c24debcdf02cb929e69189a3eb838d9e7282051fe6506ccda6c33`; `a9fce0075202af82f220c471a4054567f079f900644669acbb01633de758d6be`; `30d707dde08e51c1c001fc4aa19700f11747a93e439b04c1da2363e0ad545fc1`; `7ef538be4e1d61f9a8a81d7fdfcd79c514095203e4f51ea077f6fcfa502eca31`; `11b02e861afc0e0584f5d961fc40183ce83ae642aa3690eabfb4a6159e9c7b4b` |

#### ISCF-EQUAL

| Seed | SHA-256 by dataset order |
| --- | --- |
| 2021 | `6cc42d65b2ae48a1f5f55e3c79df96820641664f4b4668a7298d2f6108e53643`; `a4c7d5d4818c08e316cc26a8025c39b072bb4ead53119aed021da616cd7ddd31`; `6bd4d86c41c901be55708e0d09ae64bd8be8a428a5c53fa3b5d7ba4955a629b1`; `f63d649b96d6eda29e137b91990e1370c8716fc131c712d740aae7902195b40a`; `d94e4c2cc8769b45c50d44c567132c91414979b752eaf17244f38c446a713d64` |
| 2022 | `0fa2fafc9b9cdcb7c119083ae7dff5126d5d4f079d43df3c49273291c1d68915`; `8cbf406ab53ad7c0e9861a044fb5bae5cef6f067eb6bb76b7f6f6f829d17ced7`; `3f789a30b5c41d12621c4c7b07b0f06413ef0489b8becc99c077fabfc500878d`; `b0129c64d08ed7b4dd58ff654013ee6abd9a796376e1abae44b93a24a093167c`; `1b8c3afd87fe2558e00846098547ae04b2505675be33e630e9023a0f36bca5ec` |
| 2023 | `e5b4fcbeb038e2fe205e5709cbc2119f15f2aadb268e35a3dd14e19e332eaef1`; `2570c77be1f2bffd07ec01c0168854b1f1eddc0b0401183489662f1266203e72`; `0b3f474ff0dff021e400445c4d0023b263aa00a5e0fcf7f19473984f68d92cf6`; `8225ec46c60db6d19afc91a754f9b489c641c80f6916651ea4cb35bd115c8373`; `f6db9fb750ced7265606e68d3420e46413776a94e5581eea0a1e34b6c9e0b862` |

#### A6_FULL

| Seed | SHA-256 by dataset order |
| --- | --- |
| 2021 | `608023e3564640e7c0fc00c29edadc19c614fbe3a46683537ee7eff1089fa967`; `b6178da4e329647e88bb5b340650926a4d86506eb06f5ff02831dae7cf27eacf`; `cbb57f5c5f32ba2889f6ef9c0aaf165103b319fc64c288df843ac1bb4a0fcef3`; `80667ea06869d886243687a83935e14d2d22f1400de2b61464e7f56449470668`; `14b572a18f8ecc12dab40fc0f7c16e10953df982155b71eec6bfa77788f1073d` |
| 2022 | `69a92934f3a1ce3a13c4ec8210547de8f0ac4fd89d2f64f876e11b2049478ea0`; `8f3c230c6eb25d51efbd13345985623ad0eca94f22a4074d6dc574892b610e36`; `8e3377b949fdd3e886ab3705ba85761bb39d2adaf89f3f86fbc9ad5a74e8f174`; `ed78f3e812f77e95ea0843b70ab187f46af89d10cd777f0847991dfa434b185a`; `4d6a1068167a90ae79bd2fd5fcd8e5799a96e7fac20e39ef1015ec5e9dffc36c` |
| 2023 | `5972554fa0a50778e0e151731c75db9f4a78b7485d7927117aa822d403a21239`; `463f952607407c76945973a7590ad7296bdda38837be04f2d74f863105673d3d`; `57017c1d08c18d3933d5c756b775d6b5dffa7f5bb20becf24ae5a44bf81a6a91`; `c096ceaebc0eaa6d6e229028fbb9faf52dd39309cfc2be659d8bca5ec966a57e`; `a4db097e48f0923c3a5ff303b27ecd5c0d0d6e771c1cea9991d1f5e7b1b0d493` |

#### A6_MEASURE secondary reference

Only seed2021 exists. In the same dataset order, hashes are
`5cfa4eb8bd58d0c524afe747aa58665e18e9ef0490ff082d5355d9ba059c6e69`,
`501587f8cc26578b791273bdd5d0058b35b53edd2fa83f94421f071964192005`,
`2492d7c6f840fa69117201b957f7dc6b37bcaea0df44ad26f479b1e2495dae03`,
`5d89a48bad84e9c61e8c6e2acd10848f25e62a4b65241db769bffdfc596f9a05`
and
`d55d81b75735056a9016f6f3217127bca9c0c71cdcb5ac18045b1736c7846489`.
They remain historical secondary references and are excluded from the primary
matrix because the other two preregistered seeds do not exist and the arm is
not claim-critical.

## H4. Baseline Role Consolidation

### 4.1 Frozen roles

| Role | Retained arms | Training contract | Paper use |
| --- | --- | --- | --- |
| `horizon_specific_standard` | DLinear-Specific；PatchTST-Specific | four independently trained H-specific models per dataset；3 seeds | Main I primary standard panel |
| `matched_unified_adaptation` | DLinear-Unified；PatchTST-Unified | one model/dataset/seed；same split、four-H selector and unified supervision；end-to-end | Main II matched unified system benchmark；also transfer Original controls；not exact ISCF architecture attribution |
| `native_single_weight_varied_horizon` | ElasTST-native | one native checkpoint/dataset；native seed/selector | separate native varied-horizon context；not matched attribution |
| `modern_native_fixed_h_accuracy` | TimePerceiver-native；SRSNet-native | official per-H model/selector/seed plus test-hygiene patch | separate modern accuracy panel；not matched attribution |

A6_FULL is retained as a zero-new-cost repo-native unified carrier/reference,
but it is not a same-backbone attribution control for DLinear/PatchTST.

### 4.2 Pre-result exclusions

| Candidate | Status | Reason |
| --- | --- | --- |
| iTransformer | `excluded_with_reason` | PatchTST already supplies a structurally distinct Transformer-style standard and transfer backbone；no current exact source/protocol/artifacts；reviewer-requested Appendix fallback only |
| TimeMixer | `excluded_with_reason` | multi-scale prior-art boundary remains cited, but it does not add a claim-critical control beyond retained families；adding specific+unified arms would expand cost without isolating ISCF output-side sharing |
| CATS | `excluded_with_reason` | it was mandatory for historical D22-C future-query attribution, not for frozen ISCF-BSCA paper core；TimePerceiver covers the closest current target-position accuracy context；restore only if the claim changes or reviewer requires it |

### 4.3 Source/protocol blockers

| Arm | Blocking work before training |
| --- | --- |
| DLinear-Specific | promote exploratory local carrier to formal source/protocol identity；freeze per-H selector、effective config、hash/schema and no-test training path |
| PatchTST-Specific | audit official commit/license；map dataset/split/channel；freeze native profiles and artifact schema |
| DLinear-Unified | implement future-step-indexed unified contract and four-H selector after local patch authorization |
| PatchTST-Unified | source-informed unified adaptation with exact tensor/parameter contract after local patch authorization |
| ElasTST | resolve `limit_train_batches=10` semantics；prove split/channel/metric mapping；preserve native selector |
| TimePerceiver | remove training-epoch test access；freeze input length before test；export raw predictions/targets |
| SRSNet | executed-file license trace；prove normalized metric equivalence；add prediction export；remove hard-coded scheduling assumptions |

## H5. Per-Table-Cell Manifest

The machine-readable protocol encodes each block as an exact Cartesian product
of `arm × dataset × horizon × seed`. Every member is one table cell and inherits
the stated status. No cell may be silently omitted or assigned a different
status. The expanded complete surface is:

- 345 unique training checkpoint slots；
- 45 completed formal metric-evidence checkpoints reusable for existing cells；
- checkpoint binary reuse remains unverified until a remote existence/hash audit；
- 300 new checkpoints blocked before training；
- 900 unique standard-horizon seed-cells；
- 1,800 reported metric scalars：900 primary MSE + 900 secondary MAE。

### 5.1 Main I

| Arm | Product | Runs | Cells | Current status | Post-patch action |
| --- | --- | ---: | ---: | --- | --- |
| ISCF-BSCA-v1 | 5 datasets × 3 seeds；one unified checkpoint gives 4 H | 15 | 60 | `reusable` | no rerun |
| DLinear-Specific | 5 × 4 H × 3 seeds | 60 | 60 | `source_patch_required` | `retrain_required` |
| PatchTST-Specific | 5 × 4 H × 3 seeds | 60 | 60 | `missing` | Tier A source/protocol landing，then `retrain_required` |
| TimePerceiver-native | 5 × 4 H × native seed2025 | 20 | 20 | `source_patch_required` | train under native accuracy role |
| SRSNet-native | 5 × 4 H × native seed2021 | 20 | 20 | `source_patch_required` | train under native accuracy role |

Main I must use two panels: primary standard-protocol comparison and modern
native accuracy context. Native rows cannot be pooled into matched attribution.

### 5.2 Main II

| Arm | Product | Runs | Cells | Current status | Attribution role |
| --- | --- | ---: | ---: | --- | --- |
| ISCF-BSCA-v1 | 5 × 3 seeds × 4 H outputs | 15 | 60 | `reusable` | paper method |
| DLinear-Unified | 5 × 3 seeds × 4 H outputs | 15 | 60 | `missing` | matched unified system benchmark |
| PatchTST-Unified | 5 × 3 seeds × 4 H outputs | 15 | 60 | `missing` | matched unified system benchmark |
| A6_FULL | 5 × 3 seeds × 4 H outputs | 15 | 60 | `reusable` | repo-native reference only |
| ElasTST-native | 5 × native seed1 × 4 H outputs | 5 | 20 | `source_patch_required` | native varied-horizon context only |

### 5.3 Core ablations

| Variant | Exact intervention | Runs | Cells | Status |
| --- | --- | ---: | ---: | --- |
| Full ISCF-BSCA | frozen exact v1 | 15 | 60 | `reusable` |
| w/o BSCA | same architecture；remove train-only anchor；ISCF-EQUAL | 15 | 60 | `reusable` |
| w/o Independent Scope-Conditioned Forecasting | remove independent scope-specific history projections while retaining allocation and BSCA wherever algebraically defined；capacity delta reported | 15 | 60 | `missing` |
| w/o Target-Conditioned Scope Allocation | replace target-conditioned allocation by preregistered information-free equal allocation while retaining all scope slices and BSCA-compatible training path | 15 | 60 | `missing` |
| w/o Multiple Sharing Scopes | one preregistered single-scope decoder，end-to-end from same initialization class；no frozen replacement | 15 | 60 | `missing` |

Historical `siff_equal` changes more than independent projection and uses
`equal_skill`; `siff_constant_equal` is not the exact fixed-allocation arm；
Q1-WIDE、random/permuted partitions and PCSD controls are
`excluded_with_reason=historical_secondary_control_not_exact_paper_ablation`.

### 5.4 Decoder transfer

| Backbone | Original | +ISCF | +ISCF-BSCA | Current status |
| --- | --- | --- | --- | --- |
| DLinear-style | reuse DLinear-Unified block | 15 runs / 60 cells | 15 runs / 60 cells | all exact local arms currently `missing` |
| PatchTST-style | reuse PatchTST-Unified block | 15 runs / 60 cells | 15 runs / 60 cells | all exact local arms currently `missing` |

The complete transfer table is 90 runs/360 cells, but its 30 Original runs are
the same Main II checkpoints. Incremental transfer cost is therefore 60 runs/
240 cells. All six rows are end-to-end joint training. Frozen replacement,
warm-start and cross-swap are excluded from direction-level conclusions.

### 5.5 Efficiency and necessary diagnostics

Efficiency adds no training runs. It has 45 explicit
`system × dataset` profiling units from these nine systems:

1. DLinear-Specific-system；
2. PatchTST-Specific-system；
3. TimePerceiver-native-system；
4. SRSNet-native-system；
5. DLinear-Unified；
6. PatchTST-Unified；
7. ISCF-BSCA-v1；
8. A6_FULL；
9. ElasTST-native。

On seed2021/native checkpoints, every unit reports:

- parameter count and stored parameter total for all supported horizons；
- training GPU-hours from run records；
- warmed-up single-request and all-horizon service latency；
- peak inference memory；
- checkpoint/model count；
- exact CHPC contract；
- BSCA train-only inference equivalence。

Necessary diagnostics are limited to:

1. reuse existing ISCF-BSCA internal health；
2. compute naive unified penalty from Main I/Main II predictions；
3. compute horizon-specific CHPD/NCHPD from frozen Main I predictions and
   contrast it with ISCF-BSCA architectural CHPC；
4. report scope usage/entropy、prediction diversity、oracle headroom and
   per-region improvement from the frozen full checkpoints。

No new neutral-scale training, random-partition sweep, scope-count sensitivity
or router search enters the prelaunch matrix.

## H6. Resource Estimate and Workload-Aware Schedule

### 6.1 Deduplicated counts

| Block | Effective runs | Reused | New | Seed-cells |
| --- | ---: | ---: | ---: | ---: |
| existing ISCF-BSCA / ISCF-EQUAL / A6_FULL | 45 | 45 | 0 | 180 |
| Main I DLinear/PatchTST specific | 120 | 0 | 120 | 120 |
| Main II DLinear/PatchTST unified | 30 | 0 | 30 | 120 |
| three new exact core ablations | 45 | 0 | 45 | 180 |
| transfer incremental +ISCF/+ISCF-BSCA | 60 | 0 | 60 | 240 |
| ElasTST/TimePerceiver/SRSNet native | 45 | 0 | 45 | 60 |
| **Unique total** | **345** | **45** | **300** | **900** |

### 6.2 Planning estimate

[Estimate] Before source-specific resource smokes:

- new training：approximately `225–675 GPU-hours`，central estimate around
  `360 GPU-hours`；
- three RTX 3090s at 70–80% effective utilization：approximately `4–13 days`，
  central estimate around `6–7 days`；
- storage：approximately `40–220 GB` depending on raw predictions and optimizer
  retention；reserve at least `250 GB`；
- efficiency profiling：approximately `3–8 additional GPU-hours`。

These are planning ranges, not promised runtimes. After authorization, every
source family must first run a representative train/validation resource smoke;
epoch time、peak memory and artifact size then replace the range.

### 6.3 Three-GPU scheduling

Use a global dynamic longest-processing-time queue:

1. dataset priority starts `Weather → ETTm1 → ETTh1/ETTm2 → ETTh2`；
2. initialize GPUs 0/1/2 with different long arm/dataset/seed jobs；
3. any free GPU takes the longest remaining compatible job；
4. do not pair a slow and fast arm and wait for both；
5. use short DLinear jobs to fill long-tail gaps；
6. isolate external environments by wave to avoid dependency conflicts；
7. record GPU、start/end、peak memory、epoch time、command、environment、
   checkpoint hash and output path。

Planned waves after authorization:

1. Wave A：exact core ablations + Main II + transfer，because they directly gate
   the component and portability claims；
2. Wave B：Main I DLinear/PatchTST specific；
3. Wave C：ElasTST → TimePerceiver → SRSNet after each source/protocol blocker
   passes；
4. Wave D：efficiency and diagnostics on frozen checkpoints；
5. Wave E：only after separate authorization, one complete frozen formal-test
   audit。

## H7. Success, Failure Attribution and Rollback

### 7.1 Universal completeness gate

No claim passes unless:

- every preregistered dataset/horizon/seed cell is present；
- MSE and MAE plus negative cells are reported；
- checkpoint hash nonmutation and numeric health pass；
- validation selected the checkpoint without test information；
- no row, seed, dataset or horizon is removed after seeing results。

### 7.2 Performance gates

For local three-seed claim-critical comparisons, reuse the frozen v1 gate:

- macro MSE gain `>= +0.3%`；
- macro MAE gain `> 0`；
- dataset MSE wins `>=3/5`；
- horizon MSE wins `>=3/4`；
- seed MSE wins `>=2/3`；
- minimum dataset mean MSE gain `> -2%`；
- ETTm2 mean MSE gain `>= -1%`。

Application:

- Main I freezes the narrower headline
  “outperforms the evaluated standard-protocol horizon-specific baselines”；
  it requires Full to pass against both frozen primary standard families.
  Otherwise use `competitive with` or a named per-family statement. The broader
  unqualified “outperforms horizon-specific forecasters” is prohibited even
  when the two standard rows pass；TimePerceiver/SRSNet are reported separately
  as native single-seed point-estimate context；
- Main II paper-facing system-benchmark statement requires Full to pass against
  DLinear-Unified、PatchTST-Unified and A6_FULL. DLinear/PatchTST only match the
  unified evaluation/training contract, not the backbone, so they do not decide
  exact ISCF architecture attribution. That attribution is decided by the exact
  core ablations and same-backbone transfer controls；ElasTST/native rows do not
  decide matched attribution；
- “each component is effective” requires Full to pass against each of the four
  exact without controls. BSCA versus EQUAL is already complete and is not
  rerun；
- portability requires, within both backbones, `+ISCF > Original` and
  `+ISCF-BSCA > +ISCF` under the same completeness/gain gate. One-backbone pass
  is only backbone-dependent evidence；
- TimePerceiver/SRSNet/ElasTST decide paper positioning under their disclosed
  native contracts, not ISCF mechanism attribution。

### 7.3 Four-layer decision

Every Step 9–10 report must separate:

1. `paper_facing_effectiveness`；
2. `matched_mechanism_attribution`；
3. `internal_mechanism_health`；
4. `failure_attribution`。

Positive performance without matched attribution is
`performance_partial_pass`. Internal activity cannot rescue a negative
effectiveness gate.

### 7.4 Rollback

| Failure | Attribution / rollback |
| --- | --- |
| Main I or Main II negative without pathology | return to paper claim boundary and Step 4–6 narrative；do not restart architecture search automatically |
| one exact ablation fails | remove or narrow that component claim；audit `capacity_control_explains` versus `hypothesis_false`；no post-test router/lambda rescue |
| transfer fails on one backbone | `intervention_point_wrong` or `readout_or_head_design_wrong` for that adaptation；does not reject native ISCF-BSCA |
| numeric divergence, >100% degradation, broken hash or incomplete matrix | `optimization_or_numeric_pathology` / protocol invalid；return Step 7 and rerun the entire affected preregistered block |
| external split/metric/license/source equivalence unresolved | `comparison_unresolved`；not a baseline model failure |
| validation/test reversal | report completely；do not reselect checkpoint；any redesign becomes a new `test_informed` candidate and returns to narrative/design gate |

## H8. Tiered Authorization Gate

The current decision authorizes no execution. Required approvals are separate:

### Tier A — local protocol/source patch

Requested scope:

- implement and audit exact Main I/Main II/ablation/transfer contracts；
- apply minimal external test-hygiene patches；
- add manifest/completeness/diff checkers；
- prefer an external prediction-export wrapper for SRSNet；if the native tree
  must be instrumented, freeze the minimal diff and prove the training graph,
  optimizer、scheduler and selector are unchanged；
- run local CPU/CUDA construction and train/validation-only smokes；
- no remote matrix and no official test。

Pass condition: source commits/licenses、dataset mapping、tensor/parameter
identity、selectors、CLI matrix、artifact schema、no-test smoke and resource
estimate all pass.

### Tier B — remote train/validation

Requested only after Tier A report passes:

- inspect `nvidia-smi`；
- commit/push the authorized source state；
- remote `git pull`；
- run the 300 new training checkpoints in workload-aware waves；
- stop after validation-selected checkpoint hashes and completeness audit；
- no official test。

### Tier C — formal official-test audit

Requested only after all training cells are complete and immutable:

- freeze candidate/source/config/checkpoint hashes and full matrix；
- execute one test-only complete audit；
- report all cells and four evidence layers；
- no test-driven checkpoint/profile/seed selection。

## H9. Historical Gate Conclusion

[Decision] `E0 existing-artifact audit=pass`。

[Decision] `E1 baseline consolidation and minimal matrix=pass`。

[Decision] `E2 prelaunch design=conditional_pass_with_source_protocol_blockers`。

[Blocked by authorization] Tier A local protocol/source patch、Tier B remote
train/validation and Tier C formal official test remain independently false.
The next action is to request Tier A only; Tier B and Tier C must not be bundled
into implied approval.
