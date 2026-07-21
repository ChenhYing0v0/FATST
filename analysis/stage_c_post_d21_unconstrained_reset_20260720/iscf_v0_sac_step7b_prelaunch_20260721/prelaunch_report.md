# ISCF-v0 SAC Step 7B Prelaunch Report

## 1. Long-stage record

| Field | Record |
| --- | --- |
| `audit_date` | `2026-07-21` |
| `current_step` | Scope Attribution Confirmation Step7B：manifest、runner、analyzer与local prelaunch complete |
| `problem` | ISCF-v0收益是否来自有意义的future-output coupling scopes，而不是shared width或任意多分支capacity？ |
| `existence_evidence` | ISCF vs A6_FULL three-seed test MSE/MAE=`+1.3584%/+0.9144%`；D1.1 15/15 nonredundant responses |
| `idea` | 不改candidate；用Q1-WIDE与RANDOM-PARTITION分别隔离scope-specific maps和temporal partition structure |
| `theory_check` | fixed-past Bayes boundary不变；只检验finite-capacity output-sharing bias，不输入requested H或额外信息 |
| `design` | 25 new trainings + 35 hashed historical references = 60 effective runs；three seeds、five datasets、four horizons |
| `narrative_gate` | 已在Step4/5 conditional pass；SAC是promotion-blocking attribution，不是新method |
| `effectiveness_gate` | Q1 MSE margin `+0.5%`；RANDOM `+0.3%`；两者共享dataset/horizon/seed/MAE gates |
| `artifacts` | frozen config、18/18 checks、25-job manifest、35-reference audit、runner、analyzer |
| `decision` | `step7b_prelaunch_pass_waiting_remote_authorization`；remote training与formal test仍false |

## 2. Outcome

[Fact] SAC Step7B local prelaunch为`18/18`通过，具体覆盖：

1. `25/25` new jobs精确：Q1-WIDE seeds2022/2023 × five datasets=`10`；RANDOM-PARTITION three seeds × five datasets=`15`；
2. `35/35` historical references精确且protocol pass：ISCF-v0与A6_FULL three seeds各15，Q1-WIDE seed2021五个；
3. 四个effective arms × five datasets × three seeds=`60` runs，对应`240` standard-horizon metric rows；
4. five profiles上canonical/random的active parameters、readout initialization、Encoder initialization与model构造后的global RNG state一致；
5. scope1和scope720 group indices一致，48/144/360的group indices不同；同input输出shape均为`[2,720,7]`、finite且非恒等；
6. Q1-WIDE active-parameter signed gap在五datasets复核一致，最大绝对值=`0.4646379821%`；
7. runner syntax、25-job dry-run、analyzer synthetic smoke与`rg -> grep` remote log-scanner fallback通过；
8. 未授权normal launch被runner以exit code `3`硬拒绝。

[Decision] candidate implementation没有变化。SAC已经从Step6 design推进为可执行的Step7B protocol，但没有训练、validation或official-test结果，不能改变ISCF-v0的conditional candidate状态。

## 3. Exact matrix

### 3.1 Effective arms

| Arm | Seed role | Readout | Partition | Rank |
| --- | --- | --- | --- | --- |
| `iscf_v0` | historical 2021/2022/2023 | `siff-independent-scope-control` | canonical | dataset-matched 109/116/116/106/116 |
| `iscf_q1_wide` | historical 2021；new 2022/2023 | `siff-q1-wide-control` | canonical | dataset-matched 463/430/430/485/430 |
| `iscf_random_partition` | new 2021/2022/2023 | `siff-independent-scope-control` | random | exact ISCF rank |
| `a6_full` | historical 2021/2022/2023 | `learned-basis-forecast-operator` | control | 256 |

dataset顺序按remote workload冻结为Weather、ETTm1、ETTh1、ETTm2、ETTh2；它不影响汇总权重。所有新runs保持：

```text
seq_len/pred_len       = 720/720
target_horizons        = [720]
validation_horizons    = [96,192,336,720]
checkpoint             = best mean validation MSE over four horizons
pred_loss/objective    = full / equal_skill
policy                 = direct
epochs/patience        = 20/5
batch_size/lr          = 32/1e-4
test checkpoint choice = forbidden
```

### 3.2 Historical source roles

- seed2021的ISCF/Q1/A6来自`stage_c_siff_equal_attribution_v2`；
- seeds2022/2023的ISCF/A6来自`stage_c_siff_v2_fcc_v1`；
- 两份source config和run-audit均以SHA256冻结；
- historical checkpoint不重训、不重新选择，只复用既有official-test metrics和invariants。

这样做不把frozen replacement当架构效果证据：这里复用的是完整end-to-end trained arms，而不是替换某个frozen consumer。

## 4. Controls and tested claims

### 4.1 Q1-WIDE

Q1保留相同五个canonical scopes、shared synthesis、direct policy与equal-skill objective，但五个scopes只读取一个wide shared history-to-mode field。比较

$$
G_{Q1}=100\left(1-\frac{R(\mathrm{ISCF})}{R(\mathrm{Q1})}\right)
$$

检验“独立scope-specific maps是否超过near-matched shared width”。由于它不是exact parameter match，MSE macro threshold冻结为`+0.5%`，高于最大active-param gap `0.464638%`。

### 4.2 RANDOM-PARTITION

RANDOM与ISCF使用同一readout mode、rank、parameters、initialization、policy、objective和partition seed。仅48/144/360的future-coordinate grouping通过fixed permutation破坏contiguity/nesting；scale1/720保持不变。

比较

$$
G_R=100\left(1-\frac{R(\mathrm{ISCF})}{R(\mathrm{RANDOM})}\right)
$$

检验“canonical temporal scope structure是否超过任意independent grouping”。其MSE macro threshold冻结为`+0.3%`。

## 5. Statistics and split roles

对comparison $c$、metric $m$、dataset $d$、seed $s$、horizon $h$定义cell gain：

$$
g^{(c,m)}_{d,s,h}=100\left(1-\frac{m(\text{candidate})}{m(\text{reference})}\right).
$$

正值表示ISCF更好。analyzer输出：

- `macro_gain_percent`：60个dataset-seed-horizon cells的等权平均；
- `cell_wins`：$g>0$的cell数，只作完整描述，不是单独gate；
- `dataset_wins`：每dataset对3 seeds × 4 horizons平均后为正的dataset数；
- `horizon_wins`：每horizon对5 datasets × 3 seeds平均后为正的horizon数；
- `positive_seed_macros`：每seed对5 datasets × 4 horizons平均后为正的seed数；
- `q1_active_parameter_gap_percent`：$100(N_{ISCF}-N_{Q1})/N_{ISCF}$；
- `canonical_random_*_match`：对应initialization hash、parameter count和partition hash的contract checks。

两项primary comparisons各自必须同时满足：

1. MSE macro达到各自margin；
2. MSE dataset wins至少`3/5`；
3. MSE horizon wins至少`3/4`；
4. positive seed macros至少`2/3`；
5. MAE macro严格大于0；
6. 60 cells完整，negative cells不得删除。

validation只用于checkpoint selection、training health与split reversal解释。official test只在25/25 new training artifacts完成、获得显式授权后一次性执行；test不得选择epoch、rank、partition seed、dataset或horizon。

## 6. Analyzer decision map

| Evidence | Decision |
| --- | --- |
| Q1 pass + RANDOM pass + protocol health | `iscf_scope_architecture_supported_pending_modern_baselines` |
| Q1 fail；RANDOM pass | `capacity_or_shared_width_explains`；ISCF carrier-only |
| Q1 pass；RANDOM fail | `temporal_scope_structure_not_supported`；generic branches explain |
| both fail | `capacity_and_temporal_scope_attribution_both_not_supported` |
| protocol/init/numeric pathology | `diagnostic_invalid_for_direction_rejection_repair_exact_protocol` |

无论哪项失败，都不进行rank、partition-seed、loss、router或requested-H rescue。两项通过也只解除scope-attribution blocker，仍需modern-baseline/generalization gate；不会自动把ISCF标记为`passed_core_candidate`。

## 7. Failure-attribution boundary

[Self-critique] RANDOM是强exact control，但随机grouping可能比canonical grouping带来更难的optimization geometry；若它失败，支持的是“canonical temporal structure在本训练协议下有用”，而非抽象数学上的唯一最优partition。Q1则是near-matched，不是exact function-class equivalence；因此使用`+0.5%`margin并强制披露signed gaps。

如果出现NaN、missing cells、checkpoint mutation、initialization mismatch或validation/test protocol mismatch，只能修复exact protocol，不能据此拒绝ISCF方向。如果结果完整且Q1失败，则`capacity_control_explains`足以阻止independent-map claim；如果RANDOM失败，则temporal contiguity/nesting claim不成立。该归因不会被D1.1 internal response或A6 package gain覆盖。

## 8. Authorization

- local Step7B implementation/prelaunch：`true`；
- model change：`false`；
- remote training：`false`；
- formal test：`false`；
- modern baselines：`false`；
- router、second loss、requested-H conditioning：`false`。

下一步只能在用户明确授权后：commit-pinned remote pull，运行`nvidia-smi`，执行Weather-RANDOM与ETTm2-Q1 dual resource smoke；smoke finite/no-OOM后启动25-run training。25/25完成之前formal-test mode必须保持0。
