# SIFF/MCCA 四标准 Horizon 回溯重评估

## 1. 当前记录

| Field | Value |
| --- | --- |
| `candidate` | `SC1-SIFF-v1 / SC2-MCCA-v1` |
| `evaluation_role` | retrospective development screen |
| `evaluation_split` | validation |
| `paper_facing_horizons` | 96, 192, 336, 720 |
| `datasets` | ETTh1, ETTh2, ETTm1, ETTm2, Weather |
| `seed` | 2021 |
| `checkpoint_rule_inherited` | best validation H720 MSE |
| `checkpoint_reselected` | false |
| `test_used` | false |
| `decision` | standard-horizon screen fail；exact SIFF-v1/MCCA-v1仍关闭 |

本报告落实新的项目评估协议：常规method development与paper-facing scorecard使用
$H\in\{96,192,336,720\}$；dense H1..720只作机制诊断和continuous unified-horizon补充证据。

## 2. 为什么需要重评估

旧Step 7B以validation dense-prefix MSE AUC为primary screening metric。该metric等权平均全部requested
prefix horizons，因此非常强调short-prefix error。它适合定位unified-horizon pathology，但与现有论文main table
通常展示的96/192/336/720不完全对齐。

新规则不是把test变成逐次反馈：日常开发在validation上使用相同四horizon；冻结candidate、checkpoint、seeds和
main/ablation matrix后，才执行一次完整official test audit。反复根据test修改mechanism会使test承担validation
职责；adaptive holdout研究已经证明这种反馈会引入selection overfitting：

- Dwork et al., *Generalization in Adaptive Data Analysis and Holdout Reuse*, NeurIPS 2015：
  <https://papers.nips.cc/paper_files/paper/2015/hash/bad5f33780c42f2588878a9d07405083-Abstract.html>

## 3. Artifact 与统计量

### 3.1 Artifact construction

本次没有重新训练。读取旧Step 7B保存的16个相关arms × 5 datasets完整H1..720 validation curves，只抽取
H96/H192/H336/H720，共形成320行raw MSE/MAE。

旧checkpoint由best validation H720 MSE选出。由于历史run没有保存每epoch四-horizon states，本次不能按新默认
checkpoint score重新选择epoch。因此结果是：

> 新paper-facing scorecard对旧checkpoint的retrospective evaluation，而不是新协议的完整rerun。

这个边界不改变当前exact candidate的关闭状态，但意味着checkpoint-selection归因仍需`SC-D16-CTD`。

### 3.2 Relative gain

每个dataset-horizon cell的gain为

$$
G(A,B;d,H)=100\left(1-\frac{L_A(d,H)}{L_B(d,H)}\right).
$$

正值表示candidate误差更低。`macro gain`对所有dataset-horizon cells等权平均；`dataset wins`先在每个dataset内
对四个horizon gain平均，再统计正值dataset数。raw MSE/MAE没有跨dataset直接平均。

`architecture_main_effect`在每个cell先平均EQUAL、PCC、MCCA三组SIFF-vs-PCSD paired gains；
`mcca_main_effect`先平均PCSD与SIFF carrier上的MCCA-vs-PCC gains。

本次沿用旧实验预注册门槛，避免事后修改：

- architecture：macro MSE gain $\ge+0.3\%$且dataset wins $\ge3/5$；
- MCCA：macro MSE gain $\ge+0.2\%$且dataset wins $\ge3/5$；
- joint：macro MSE gain $\ge+0.3\%$且dataset wins $\ge3/5$。

新报告的cell wins仅作透明度统计，不在回溯分析中临时升级为硬门槛。

## 4. Paper-facing primary result

| Effect | Macro MSE gain | Macro MAE gain | Cell wins | Dataset wins | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| SIFF architecture main effect | -2.3509% | -1.6146% | 8/20 | 2/5 | fail |
| MCCA main effect | -0.1357% | -0.2075% | 7/20 | 1/5 | fail |
| SIFF+MCCA vs A6 | -1.3325% | -0.9874% | 14/20 | 4/5 | fail |

[Fact] Joint candidate与A6的raw validation MSE如下；每格为`SIFF+MCCA / A6`：

| Dataset | H96 | H192 | H336 | H720 |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | 0.8100 / 0.8188 | 1.0521 / 1.0564 | 1.2636 / 1.2626 | 1.4212 / 1.4086 |
| ETTh2 | 0.2321 / 0.2547 | 0.3146 / 0.3292 | 0.4151 / 0.4267 | 0.6241 / 0.6463 |
| ETTm1 | 0.3853 / 0.4089 | 0.5085 / 0.5305 | 0.6603 / 0.6766 | 0.9474 / 0.9541 |
| ETTm2 | 0.1576 / 0.1123 | 0.1787 / 0.1521 | 0.2123 / 0.1988 | 0.2775 / 0.2812 |
| Weather | 0.3948 / 0.4014 | 0.4478 / 0.4544 | 0.5090 / 0.5130 | 0.5904 / 0.5898 |

完整raw MSE/MAE见`standard_horizon_metrics.csv`，这里不以跨dataset raw loss均值代替paired relative gain。

[Fact] 新scorecard没有推翻旧dense-AUC判定，负证据反而更强：

| Effect | 旧 dense MSE AUC | 新 four-horizon MSE | Decision |
| --- | ---: | ---: | --- |
| SIFF architecture | -1.5015% | -2.3509% | fail -> fail |
| MCCA | -0.0250% | -0.1357% | fail -> fail |
| Joint vs A6 | -0.5621% | -1.3325% | fail -> fail |

## 5. Horizon 与 dataset attribution

### 5.1 按 horizon

| Horizon | SIFF architecture | MCCA | Joint vs A6 |
| ---: | ---: | ---: | ---: |
| 96 | -6.3186% | -0.5210% | -4.6045% |
| 192 | -2.6027% | -0.0449% | -1.4181% |
| 336 | -1.0522% | +0.1357% | -0.1948% |
| 720 | +0.5698% | -0.1128% | +0.8875% |

[Strong Evidence] SIFF的失败具有清晰的horizon signature：越靠近short/mid horizon越差，H720反而正向。
因此旧H1 pathology不是dense-AUC制造的假象；即便只看论文标准horizon，H96也已经暴露严重退化。

### 5.2 按 dataset

| Dataset | SIFF architecture | MCCA | Joint vs A6 |
| --- | ---: | ---: | ---: |
| ETTh1 | +1.4797% | -0.1951% | +0.1278% |
| ETTh2 | +1.9191% | -0.0870% | +4.8664% |
| ETTm1 | -0.6621% | +0.0933% | +3.2563% |
| ETTm2 | -13.9594% | -0.4867% | -15.8593% |
| Weather | -0.5319% | -0.0033% | +0.9463% |

ETTm2仍是主要异质性来源，但属于冻结matrix，不能事后删除。Joint虽有4/5 dataset wins与14/20 cell wins，
ETTm2的系统性退化使macro为负，所以不能通过。

## 6. Attribution controls

| Comparison | Macro MSE gain | Cell wins | Dataset wins | Boundary |
| --- | ---: | ---: | ---: | --- |
| ordered vs constant | -1.7934% | 16/20 | 4/5 | ETTm2抵消其余dataset |
| ordered vs permuted | +1.1675% | 18/20 | 5/5 | order signal retained |
| ordered vs Q1-wide | -2.1082% | 11/20 | 3/5 | generic width解释macro |
| ordered vs independent | -3.1756% | 8/20 | 2/5 | shared field未超过independent scopes |
| ordered vs dense matched | +1.7580% | 16/20 | 4/5 | generic dense head不是全部解释 |
| transport vs pointwise | +0.5243% | 16/20 | 4/5 | transport ingredient retained |
| capability marginal vs uniform OT | +0.1361% | 20/20 | 5/5 | small consistent ingredient |

这些controls保留scope order、prefix transport与capability marginal的局部问题证据，但无法挽救exact SIFF/MCCA。
尤其MCCA main effect仍为负，不能把transport或marginal的小收益改写为完整MCCA method pass。

## 7. Failure attribution 与决策

1. `hypothesis_false`：
   exact MCCA competitive assignment在标准horizon下仍不超过same-mass PCC，关闭不变。
2. `readout_or_head_design_wrong / checkpoint_pathology`：
   SIFF在H96为负、H720为正，表明scale-field更像把capacity偏向long horizon。现有checkpoint由H720选择，
   无法判断trajectory中是否存在四-horizon更均衡的epoch。
3. `optimization_or_numeric_pathology`：
   原runs finite，未发现divergence；但ETTm2 short/mid degradation足够大，方向级拒绝仍需CTD。
4. `capacity_control_explains`：
   Q1-wide和independent controls在macro上优于ordered SIFF，exact continuous shared field不成立。

最终decision：

- exact `SC1-SIFF-v1`与`SC2-MCCA-v1`继续关闭；
- 不授权confirmation seed或test audit；
- `SC-D16-CTD`继续作为`diagnostic_only`，但主trajectory score改为validation
  H96/H192/H336/H720，dense/H1保留为failure attribution；
- 若best-standard checkpoint仍无法修复ETTm2并超过PCSD/constant/Q1 controls，回Step2关闭scale-field方向；
- 若修复，只允许five-dataset unchanged validation confirmation，不能直接升级paper contribution。

## 8. Artifact definitions

- `standard_horizon_metrics.csv`：每行来自一个dataset-arm-horizon，`mse/mae`直接读取原
  `metrics_by_target_horizon.csv`。
- `standard_horizon_effects.csv`：每行是一个paired或factorial-composite
  dataset-horizon gain；`factor_count`表示该cell平均了几个paired factors。
- `standard_horizon_breakdown.csv`：按dataset或horizon聚合每个effect；`cell_wins/cells`统计该组正MSE gain。
- `standard_horizon_summary.csv`：每个effect的全矩阵macro、cell wins、dataset wins及worst dataset。
- `standard_horizon_gate.json`：split、checkpoint边界、沿用门槛与最终gate。
