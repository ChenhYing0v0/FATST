# Introduction Evidence Visualization Pilot Step7A 与 Prelaunch

## 1. 当前节点

| Field | Content |
| --- | --- |
| `date` | `2026-07-29` |
| `protocol` | `SC-UVHF-INTRO-EVIDENCE-VIZ-v1` |
| `current_step` | Step7A implementation pass；Step7B prelaunch pass |
| `user_change` | 先减少runs；允许选择较明显但非极端案例 |
| `initial_dataset` | Weather |
| `seed` | 2021 |
| `new_runs` | 9 |
| `split` | validation only |
| `test_accessed` | false |
| `remote_training_authorized` | true，仅initial 9 runs |
| `fallback_authorized` | false |
| `formal_test_authorized` | false |
| `decision` | `step7b_prelaunch_pass_remote_launch_next` |

## 2. Scope reduction

原正式设计的180 baseline checkpoints与75 neutral runs均继续保留为未来
paper-facing confirmation design，但不作为本轮前置条件。

本轮只执行：

| Evidence | Matrix | Runs |
| --- | --- | ---: |
| Prefix disagreement | DLinear × Weather × H96/H192/H336/H720 × seed2021 | 4 |
| Sharing demand | Neutral × Weather × s1/s8/s32/s128/s720 × seed2021 | 5 |
| Total |  | 9 |

fallback顺序预先记录为`ETTm1 -> ETTh2`，但当前未授权。Weather只要得到清晰
visualization就停止扩展。

## 3. Example-selection amendment

用户取消“必须选择median sample”的限制。本轮固定：

- origin选择六个horizon pairs aggregate NCHPD的85% quantile nearest sample；
- channel也选择aggregate disagreement的85% quantile nearest channel；
- 禁止maximum或top-1%案例；
- summary公开真实percentile、score和index；
- figure caption必须说明这是purposefully selected illustrative example。

这允许提高图的可读性，但也把Figure 1明确降为`illustrative existence`，不能
单独支持prevalence或average materiality。

## 4. Split 与claim boundary

本轮只使用validation：

- validation选择checkpoint；
- validation导出prediction artifacts；
- validation选择85% quantile overlay；
- validation计算sharing-risk landscape与descriptive region oracle；
- DLinear `--skip-test`不构造test dataset；
- neutral trainer从代码路径上不构造test dataset。

允许：

- 展示一个非极端的prefix disagreement案例；
- 展示Weather/seed2021的exploratory risk crossover；
- 判断是否值得继续制作Introduction figure。

禁止：

- formal problem-existence pass；
- cross-dataset/seed prevalence；
- out-of-sample adaptive headroom；
- 用same-validation region oracle证明learnable allocation；
- untouched test或official-test claim。

## 5. Implementation

新增：

- `baselines/intro_evidence_neutral/model.py`；
- `baselines/intro_evidence_neutral/train.py`；
- `scripts/analyze_intro_prefix_disagreement.py`；
- `scripts/analyze_intro_sharing_demand.py`；
- `scripts/check_intro_evidence_visualization_pilot.py`；
- `scripts/remote/run_intro_evidence_visualization_pilot.sh`；
- `configs/intro_evidence_visualization_pilot_v1.json`。

修改：

- `baselines/dlinear/train.py`新增validation artifact export与`--skip-test`；
  默认paper-facing test behavior不变。

详细tensor path见
`docs/code-explanation/intro-evidence-visualization-pilot-step7a.md`。

## 6. Local verification

### 6.1 Static

- touched Python `py_compile`：pass；
- config JSON parse：pass；
- remote runner `bash -n`：pass；
- remote runner local `DRY_RUN=1`：pass，`jobs=9`。

### 6.2 Neutral model

| Check | Result |
| --- | --- |
| five-scale parameter counts | all `111312` |
| prediction shape | `[2,720,3]` |
| candidate/pooled shape | `[2,3,720,64]` |
| finite nonzero gradient parameter groups | `14/14` for all scales |
| maximum within-block pooled-state gap | `0` |
| same-weight s1-vs-s720 prediction max gap | `0.080029249` |

### 6.3 Analyzer

Synthetic artifacts验证：

- same-origin history/target guards通过；
- prefix pair metrics、85% sample selection、overlay与heatmap生成通过；
- sharing step/region risks、crossover detection、headroom与three-panel SVG生成通过；
- both analyzers report `test_accessed=false`。

### 6.4 DLinear validation-only path

使用本地Weather、H96、one epoch、CPU完成artifact smoke：

- `checkpoint.pt`；
- `metrics_val.json`；
- `predictions_val.npz`；
- `training_log.csv`；
- `effective_config.json`；
- `environment.json`。

没有产生`metrics.json`或`predictions_test.npz`，证明`--skip-test`生效。临时目录已
清理。

## 7. Failure attribution

- prefix overlay在85% quantile仍不可见：
  `illustration_weak`；不通过挑maximum修复；
- neutral出现one-scale severe degradation或不收敛：
  `optimization_or_numeric_pathology`，不能判hypothesis false；
- neutral risks无crossover且训练健康：
  只能标记`Weather_seed2021_visualization_not_supported`，不能方向级拒绝；
- Weather清晰：
  停止fallback扩展，保留其illustrative而非prevalence角色；
- Weather不清晰：
  先汇报结果，再请求是否授权fallback ETTm1；不得静默继续dataset search。

## 8. Remote gate

用户当前授权覆盖：

- commit/push；
- remote GPU preflight；
- one CUDA resource smoke；
- initial Weather 9-run validation-only matrix；
- training完成后的automatic validation analysis。

不覆盖：

- fallback datasets；
- official test；
- full 180/75 matrices；
- method modification或new loss/router。

## 9. Visualization-search extension

用户后续明确授权以“找到具有说服力的样本/数据集”为目标继续筛选。当前扩展严格
限制为：

- dataset=`ETTm1`；
- family=`sharing-only`；
- scales=`1,8,32,128,720`；
- seed=`2021`；
- validation only；
- 其余architecture、objective、optimizer、12×60 regions、0.5% crossover margin
  与0.5% headroom threshold保持不变；
- 不重复DLinear；
- ETTh2仍未授权。

该扩展是figure-candidate search，不是ISCF-BSCA architecture effectiveness gate，
不得据某一个入选dataset否定或确认完整论文架构。
