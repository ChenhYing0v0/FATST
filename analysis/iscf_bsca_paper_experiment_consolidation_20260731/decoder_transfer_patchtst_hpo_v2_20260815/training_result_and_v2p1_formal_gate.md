# PatchTST Decoder-HPO v2 Training Result and v2.1 Formal Gate

日期：2026-08-15  
Decision：`parent_v2_artifact_uniqueness_fail_validation_pass_v2p1_formal_candidate_frozen`

## 1. 完整性审计

- remote root：`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_decoder_transfer_patchtst_hpo_v2_20260815`；
- training/validation：50/50 runs complete，所有validation指标finite，formal test jobs=0；
- unique checkpoint hashes：40/50，未达到parent v2预注册的50/50硬门槛；
- duplicate不是路径复用：对应run directories、inodes、mtimes与effective configs均不同；重复pair在同一dataset上具有bitwise-identical checkpoint、epoch与validation metrics；
- 结构化重复共10对，来自以下两个profile pairs在五个datasets上的系统性collapse：
  - `p02_lr0p50` = `p08_lr0p50_wd1e4`；
  - `p05_wd1e5` = `p06_wd1e4`。

[Strong Evidence] 两组差异仅为极小的decoder-only AdamW weight decay；在当前FP32 update path上其数值影响不足以改变最终权重。没有证据支持checkpoint串写或copy corruption。无论原因如何，parent v2的literal `50 unique hashes` gate仍为FAIL，不作事后放宽。

Canonical audit artifacts：

- `training_audit/artifact_summary.json`；
- `training_audit/trial_scorecard.csv`；
- `training_audit/selection_result.json`；
- `training_audit/selected_profiles.csv`。

## 2. Validation-only selection

四horizon validation mean MSE selector得到：

| Dataset | Selected profile | Rank | Readout LR multiplier | Readout WD | Val MSE | Gain vs v1 ref. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Weather | `p04_lr4p00` | 116 | 4.00 | 0 | 0.520545 | +0.354% |
| ETTm1 | `p01_lr0p25` | 116 | 0.25 | 0 | 0.611411 | +2.443% |
| ETTm2 | `p01_lr0p25` | 106 | 0.25 | 0 | 0.184930 | +0.220% |
| ETTh1 | `p01_lr0p25` | 109 | 0.25 | 0 | 1.091247 | +0.400% |
| ETTh2 | `p10_rank1p50_lr0p50_wd1e4` | 174 | 0.50 | 1e-4 | 0.377104 | +0.228% |

- macro reference validation MSE：0.561611；
- macro selected validation MSE：0.557047；
- macro gain：+0.8126%，超过预注册的`>0.25%`；
- dataset gain `>0.1%`：5/5，超过预注册的`>=3/5`；
- 五个selected checkpoint hashes互异，且都不属于collapsed profile pairs。

因此validation performance gate通过，但parent artifact gate失败；两者不得合并描述为parent v2整体通过。

## 3. v2.1 protocol amendment

在用户于2026-08-15明确授权“继续formal test并整理结果”后，冻结`ISCF-BSCA-DECODER-TRANSFER-PATCHTST-v2.1`：

1. 复用上述五个selected BSCA checkpoints，不改checkpoint、不按horizon重选；
2. 对每个dataset from scratch训练一个matched `+ISCF` checkpoint；
3. matched pair共享PatchTST encoder、rank、base LR、batch、patience、readout LR multiplier、readout WD、seed与initialization class，唯一方法差异为`pcc_objective_mode`；
4. 10/10 checkpoints、10 unique hashes及5/5 matched initialization pairs形成immutable manifest后，才允许一次formal test；
5. formal test新增10 checkpoint jobs / 40 standard cells；旧DLinear三arms与旧PatchTST Original Decoder共80 cells直接复用v1 evidence，不重复访问test；
6. combined Decoder-Transfer table仍必须完整报告2 backbones × 3 arms × 5 datasets × 4 horizons=120 cells。

该修订不追认parent v2的50-hash gate，也不删除negative trials或v1 PatchTST负结果。v2.1是`test-informed`、validation-selected rescue candidate，不得声称untouched holdout。

## 4. Formal gates and rollback

PatchTST-style `+ISCF-BSCA`相对Original Decoder必须同时满足：

- macro MSE gain > 0；
- macro MAE gain > 0；
- dataset-mean MSE wins >=3/5。

同时单独报告BSCA vs matched ISCF。若PatchTST gate仍失败且无numeric/artifact pathology，则归因更新为`hypothesis_false_for_cross_backbone_portability_after_decoder_HPO`，保留v1总体负结论并回Step 4--6设计iTransformer-style carrier；不得继续在official test上扩展PatchTST search。

## 5. Authorization boundary

- matched remote training：authorized；
- manifest-gated formal test：authorized once for v2.1；
- checkpoint/seed/per-horizon/per-cell selection：forbidden；
- partial reporting：forbidden；
- table update：仅在120/120 combined cells与完整gate审计后执行。
