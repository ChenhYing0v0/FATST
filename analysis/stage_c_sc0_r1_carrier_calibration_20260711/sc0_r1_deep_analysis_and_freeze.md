# StageC SC0-R1：Multi-Seed Carrier Gate与冻结决策

> 2026-07-12 governance update：本文冻结的global P24 profile保留为更严格的历史control，但不再是active
> StageC要求。用户允许dataset有有限结构偏好；active mapping改为Weather=P12、ETTm1=P48、ETTh2=P24，
> 仍只使用本文三seedvalidation evidence选择。见
> `configs/stage_c_mechanism_control_dataset_aware.json`。

## 1. Decision

- `current_step`: SC0-R1 Step 9-10完成；StageC随后回到SC1/SC2 Step 1-3
- `runs`: 27/27
- `errors`: 0
- `test_metrics_used_for_selection`: false
- `calibration_profile_hash`: `3ebd07d647cdd4b0e8ea36a53eea9451d21f438a79164f74b8f4e8095426f31a`
- `selected_arm`: `sc0_p24_d64`
- `decision`: `global_profile_selected_and_frozen`

[Decision] `p24/d64`作为StageC standardized mechanism-control carrier冻结。它用于后续机制归因、消融与
small gate，不替代TimeAlign source-faithful reproduction，也不声称是每个dataset/horizon的最优preset。

## 2. Preregistered Gate

| Gate | Threshold | Observed | Result |
| --- | --- | --- | --- |
| run/config/numeric completeness | 27/27 | 27/27, all `ok` | pass |
| pooled vs median selector | same arm | both `p24/d64` | pass |
| seed winner direction | at least 2/3 | 2021=`p24`, 2022=`p12`, 2023=`p24` | pass |
| pooled max dataset regret | <=3% | 1.277% | pass |
| max seed-dataset regret | <=5% | 1.440% | pass |
| test blindness | no test use | false | pass |

该pass不是从单seed结果外推；所有三臂都在三个seeds上重新训练，因此global winner可真实复算。

## 3. Pooled Full-720 Evidence

| Arm | Macro regret | Max dataset regret | Weather MSE | ETTm1 MSE | ETTh2 MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `p12/d128` | 0.967% | 1.785% | **0.587234** | 0.972978 | 0.661798 |
| `p24/d64` | **0.695%** | **1.277%** | 0.594733 | 0.963647 | **0.654492** |
| `p48/d32` | 1.378% | 2.646% | 0.595970 | **0.955917** | 0.671807 |

`p24/d64`不是任何dataset都最优：Weather偏好`p12`，ETTm1偏好`p48`，ETTh2偏好`p24`。它通过的是
预注册的全局折中准则，而不是逐dataset winner count。这正是禁止dataset-specific presets后的预期选择。

## 4. Training-Policy Repair Evidence

27条trajectory全部由patience规则停止，realized epochs为6-16，best epochs为1-11。SC0中ETTh2训练到
epoch20产生的31.63%-44.95% degradation没有再次作为deployed checkpoint出现；每个run都恢复并评估
best-validation state。

这支持以下机制判断：SC0 selector reversal主要来自fixed-20 terminal checkpoint pathology，而不是
common token-MLP topology本身。统一超参数应冻结相同的validation-controlled rule，而不是强制所有
dataset在同一epoch停下。

## 5. Dense-Horizon Diagnostic Boundary

SC0-R1按预注册规则只用full-720 validation MSE选择profile；dense horizons不参与选择。将三seed MSE先
聚合后，与同dataset/horizon三臂oracle比较，`p24/d64`的diagnostic regret为：

| Dataset | Mean dense-horizon regret | Max regret | Max location |
| --- | ---: | ---: | --- |
| Weather | 0.914% | 1.277% | H720 |
| ETTm1 | 5.269% | 11.230% | H48 |
| ETTh2 | 1.646% | 6.428% | H48 |

[Boundary] 因此`p24/d64`是一个full-720-selected causal carrier，不是all-horizon tuned baseline。尤其在
ETTm1/ETTh2短horizon上，patch granularity仍影响absolute performance。后续SC1/SC2必须：

1. 用同一个frozen `p24/d64` base arm做matched mechanism attribution；
2. 将dense-horizon结果完整报告，不能只报H720；
3. 不把相对较弱短horizon operating point产生的恢复直接解释为新机制独有收益；
4. 在paper performance层继续单列source-faithful和native external baselines。

该diagnostic不会post-hoc推翻已通过的SC0-R1 gate，但它限制可声明的claim，并构成后续effectiveness gate的
必要审计项。

## 6. Frozen Contract

后续StageC mechanism experiments引用：

- calibration evidence：`configs/stage_c_mechanism_control_r1.json`及其SHA256；
- resolved profile：`configs/stage_c_mechanism_control_frozen.json`；
- selected carrier：`P=24`, patch length 30, `D=64`, `d_ff=536`, basis rank256；
- training：full-720 L1、AdamW、LR`1e-4`、cosine、effective batch32、max20、patience5、best-val；
- seeds：2021/2022/2023；dense validation horizons：48/96/144/192/288/336/512/720。

任何candidate若改变上述carrier或training字段，必须重新注册candidate family和matched control，不能继续
引用SC0-R1作为严格因果对照。

## 7. Next Cursor

SC0 blocker已关闭。StageC回到paper-level Step 1-3，并行推进：

- `SC1-PFO`：projective decoder/operator的prior-art boundary与problem diagnostic；
- `SC2-HML`：horizon-measure training的exposure/gradient causal diagnostic。

下一步先完成统一prior-art/problem matrix，再分别决定是否进入Step 4-6 narrative gate；当前仍不授权方法
implementation。
