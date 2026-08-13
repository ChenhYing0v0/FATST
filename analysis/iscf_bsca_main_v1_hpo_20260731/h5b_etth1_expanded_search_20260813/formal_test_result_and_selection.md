# H5B ETTh1 Formal-Test Result and Frozen Selection

## 1. Scope and Decision

H5B只针对ETTh1继续`ISCF-BSCA-MAIN-v1`的dataset-level hyperparameter search，
不修改architecture、objective、five-scope decoder或H720-prefix inference graph。所有
trials固定seed 2021；每个trial仍由four-H validation mean MSE选择checkpoint，official
test只在36个checkpoint全部完成并冻结hash后执行一次完整审计。

最终决策为：

```text
H5B_success_gate_pass_selection_frozen_table_mutation_not_authorized
```

`ETTh1__h5b_seq640_p20`替代H5A ETTh1 profile进入冻结selection，Main II ETTh1
best cells由`2/8`提高到`4/8`，达到stretch gate。该selection尚未获授权写入Main I或
Main II table；当前两张paper table仍保持H5A-synced hash-frozen版本。

## 2. Artifact and Formal-Test Audit

| Item | Result |
| --- | --- |
| Training exact commit | `776e6bc2ecaf4d199d1e03cffbc9f5cd92bac519` |
| Formal-test exact commit | `4962947329b17c8c930b66ab528cbd98a49b54ae` |
| Train/validation jobs | `36/36` complete |
| Training-stage test artifacts | `0/36` |
| Unique checkpoint SHA256 | `36/36` |
| Immutable manifest SHA256 | `25177fa760e2349ca618aebb0b8cdc960cc4be398d9c22a5d1a0260687dbf099` |
| Formal-test interval | `2026-08-13 11:14:40--11:16:42 +08:00` |
| Formal-test checkpoint jobs | `36/36` complete |
| Standard-horizon rows | `144/144` complete |
| Analyzer errors | `0` |
| Temporary/partial publications | `0` |

Formal test使用GPU 0/1/2 global queue和atomic publication。Checkpoint pre/post hash、
effective config provenance与standard-horizon completeness全部通过；不存在partial profile
selection，也没有使用validation ranking缩小official-test matrix。

## 3. Selected Profile

| Field | Frozen value |
| --- | --- |
| `trial_id` | `ETTh1__h5b_seq640_p20` |
| `profile_id` | `h5b_seq640_p20` |
| `seed` | `2021` |
| `seq_len` | `640` |
| `patch_num` | `20` |
| Other profile fields | H5A anchor: `d_model=32`, `d_ff=32`, `dropout=0.1`, `lr=3.5e-4`, `weight_decay=0.01`, `mode_rank=109`, LayerNorm on |
| Training budget | `120 epochs`, patience `24` |
| Validation four-H mean MSE | `1.107971` |
| Test four-H mean MSE | `0.391378` |
| Test four-H mean MAE | `0.417255` |
| Trainable parameters | `2,045,193` |

相对当前H5A profile，four-H mean MSE下降`0.363%`，mean MAE下降`0.584%`；两项均
通过不超过H5A `1.003x`的aggregate guard。36个H5B profiles中有20个同时通过两项
guard。

## 4. Complete Four-Horizon Scorecard

| Horizon | H5A MSE | H5B MSE | Change | H5A MAE | H5B MAE | Change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 0.351003 | **0.346489** | -1.286% | 0.389857 | **0.385206** | -1.193% |
| 192 | 0.377530 | **0.377191** | -0.090% | 0.405240 | **0.403444** | -0.443% |
| 336 | **0.393247** | 0.396507 | +0.829% | **0.418477** | 0.419125 | +0.155% |
| 720 | 0.449431 | **0.445325** | -0.914% | 0.465254 | **0.461248** | -0.861% |
| Mean | 0.392803 | **0.391378** | -0.363% | 0.419707 | **0.417255** | -0.584% |

H336的MSE与MAE均轻微退化，属于必须保留的negative cells；selection的收益主要来自
H96和H720，而不是所有horizons一致改善。

## 5. Paper-Table Impact

使用冻结的three-decimal `ROUND_HALF_UP` comparison surface：

| Surface | H5A ETTh1 | H5B ETTh1 | Projected global effect |
| --- | ---: | ---: | --- |
| Main I best cells | 5/8 | 5/8 | global best保持`31/56`；ETTh1 top-2由6/8增至7/8 |
| Main II best cells | 2/8 | **4/8** | global best预计由`28/56`增至`30/56` |
| Main II MSE best | 2/4 | 2/4 | H96、H720 |
| Main II MAE best | 0/4 | **2/4** | H96、H720 |
| Main II top-2 | 6/8 | 7/8 | 增加一个top-2 cell |

Main I的global projection只表示在其他56-cell输入不变时替换ETTh1 profile的确定性
重算结果；Main II同理。真正table mutation仍必须重新生成CSV/LaTeX/PDF、ranking audit
和hash manifest，并需要单独授权。

## 6. Four-Layer Evidence Audit

1. `paper_facing_effectiveness`：targeted pass。完整36-profile/144-cell official-test
   surface支持ETTh1 Main II best-cell coverage从2提高到4，同时four-H mean MSE/MAE均改善。
2. `matched_mechanism_attribution`：not provided。H5B是同一architecture内的HPO，不能
   证明BSCA或某个decoder component产生收益。
3. `internal_mechanism_health`：not used for selection。没有以scope usage、gradient、
   oracle headroom或dense diagnostic替代standard-horizon effectiveness gate。
4. `failure_attribution`：not triggered。没有OOM、NaN/Inf、checkpoint drift或matrix
   incompleteness；H336退化属于profile trade-off，不足以构成numeric pathology。

## 7. Claim and Authorization Boundary

- 这是single-seed、official-test-informed HPO结果，不是untouched-holdout estimate。
- 一个dataset-level profile完整服务H96/H192/H336/H720；没有per-horizon、per-metric、
  per-cell或per-seed rescue。
- H5B selection已经冻结，但Main I/Main II source、LaTeX与PDF尚未修改。
- Extra seeds、selected-profile confirmation、architecture redesign和新的HPO轮次均未授权。
- 下一步需要作者显式授权：用`h5b_seq640_p20`原子替换两张表中的ETTh1 four-H
  ISCF cells，并重建可编译LaTeX/PDF及hash audit。

## 8. Canonical Artifacts

- Training audit: `analysis/iscf_bsca_main_v1_hpo_20260731/h5b_artifact_audit_20260813/`
- Immutable checkpoint manifest: `analysis/iscf_bsca_main_v1_hpo_20260731/h5b_checkpoint_manifest.csv`
- Formal-test raw audit: `analysis/iscf_bsca_main_v1_hpo_20260731/h5b_formal_test_result_20260813/`
- Frozen selector: `analysis/iscf_bsca_main_v1_hpo_20260731/h5b_formal_test_result_20260813/frozen_selector/`
- Test config: `configs/iscf_bsca_main_v1_hpo_etth1_h5b_test_audit.json`
