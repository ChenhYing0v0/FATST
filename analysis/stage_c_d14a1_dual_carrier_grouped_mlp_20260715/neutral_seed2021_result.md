# StageC D14-A1 Neutral Seed2021 Result

## 0. Decision Card

| Field | Value |
| --- | --- |
| `current_step` | Step 8-10 neutral screen complete；A6 sensitivity running |
| `role` | `diagnostic_only` problem-existence screen |
| `carrier` | `neutral_raw`：normalized history `[B,C,720]`，无learned encoder/global basis |
| `matrix` | 5 datasets × 8 arms × seed2021；40/40 complete |
| `split` | train + validation only；test=false |
| `decision` | `neutral_problem_pass_authorize_a6_sensitivity` |
| `paper_method` | false；D14-B=false；confirmation seeds held |

## 1. What Was Tested And Why

D14-A0的linear RRR scales几乎产生同一预测，不能判断output-sharing scope是否真实重要。A1让point/block/global
scales拥有不同的nonlinear hidden-bank sharing topology，同时通过parameter matching与exact full-affine containment
排除基础linear capacity差异。

neutral carrier直接把720点raw history交给decoder，因此结果不受“A6 encoder/profile围绕global basis decoder共适配”
的影响。它是方向级primary gate；A6只在neutral通过后回答paper carrier compatibility。

## 2. Artifact Construction And Alignment Repair

每个run从头训练full-H720 pointwise L1，以best-validation H720 MSE checkpoint生成validation row-level artifacts：

- `row_bin_mse/mae [N,3]`：每个window-channel在short `[0,144)`、mid `[144,360)`、long `[360,720)`的loss；
- `persistence_row_bin_mse [N,3]`：相同行的persistence control；
- `probe_predictions/probe_targets [1024,720]`：检查不同arms是否学成不同函数。

首次聚合暴露official validation loader默认shuffle，不同arm的row order不一致。该结果在统计前即被hard invariant拦截，
没有形成错误结论。checkpoint evaluator随后改为`shuffle=False`的dataset-sequential loader，40个checkpoint只重算
diagnostics、不重训；`row_order=dataset_sequential`成为analyzer硬条件。本地重算与远端gate完全一致。

此问题归因为`diagnostic_artifact_alignment_fault`，不是architecture、scale hypothesis或optimization failure。

## 3. Metric Definitions

- `carrier skill`：train-only selected fixed canonical scale相对persistence的validation MSE gain；
- `function separation`：canonical arm pair prediction RMSE相对target RMS的median，门槛0.5%；
- `crossing`：同一scale pair在三个future bins中出现至少0.1%的双向胜负；
- `oracle gain`：逐sample × bin选择最佳canonical scale，相对train-only selected fixed scale的headroom；
- `contiguity gain`：canonical contiguous oracle相对matched random-partition oracle的gain。

oracle是problem headroom，不是可实现forecast performance；只有未来D14-B证明history可预测winner后才可能兑现。

## 4. Returned Results

| Dataset | train-selected fixed scale | carrier gain | prediction disagreement median | crossing | oracle gain | canonical vs random |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| ETTh1 | 360 | 43.6535% | 12.4032% | yes | 11.2336% | 0.6579% |
| ETTh2 | 720 | 14.8485% | 19.0754% | yes | 6.3521% | 0.7026% |
| ETTm1 | 360 | 51.0500% | 11.3844% | yes | 5.3383% | 0.7103% |
| ETTm2 | 48 | 30.7374% | 26.0511% | yes | 10.0390% | 1.8778% |
| Weather | 360 | 51.5834% | 9.3290% | yes | 5.4136% | 0.5240% |
| **Macro / count** | dataset-dependent | **5/5 pass** | **5/5 pass** | **5/5** | **7.6753%** | **0.8945%, 5/5 positive** |

所有invariants通过，最大decoder parameter relative gap为0.1471%，无non-finite或severe degradation。

## 5. Interpretation

[Strong Evidence] 在一个不含A6/global-basis bias的有效E2E carrier上，不同coupling scales确实学成显著不同函数，
五数据集都出现future-bin crossing，且sample × bin oracle headroom远高于0.5%门槛。D14-A0的0.0586% oracle是
旧linear intervention过弱造成的假阴性，而不是scale problem不存在。

[Strong Evidence] train-only best fixed scale跨dataset不一致（48、360、720均出现），canonical grouping在5/5上
优于matched random grouping，支持“future temporal neighborhood的sharing scope有结构意义”，而非generic多头容量。

[Uncertainty] 当前只有seed2021；oracle不可由inference直接使用；neutral raw head不等于最终paper carrier。因此结果只
授权A6-natural sensitivity，不授权PCSD/CCRL、D14-B、test或paper claim。

## 6. Causal Boundary And Next Decision

A6-natural使用围绕global basis decoder冻结的encoder/profile。若A6也通过，说明problem evidence能迁移到当前paper
carrier；若A6 flat/fail，neutral正证据仍保留，结论只能是`carrier_interface_or_profile_incompatibility`，不能拒绝
scale hypothesis。只有A6完成后才决定是否运行confirmation seeds；runner不会自动启动它们。

## 7. Artifacts

- local independent gate：`local_reanalysis_neutral_raw_seed2021/gate.json`；
- per-dataset metrics：`local_reanalysis_neutral_raw_seed2021/dataset_metrics.csv`；
- remote raw mirror：`raw/`（git-ignored，约211 MiB）；
- design/local contract：`source_theory_design_audit.md`与`local_gate/`。
