# SC-MNB Step1-3：Official Source 与 Native Protocol Audit

## 1. Decision

| Field | Content |
| --- | --- |
| `current_step` | SC-MNB Step1-3 source/native contract audit complete |
| `problem` | 四个P0 external baselines能否在不污染test governance的前提下形成可复现carrier-viability matrix？ |
| `existence_evidence` | official repositories、native configs/scripts与training loops |
| `idea` | source-faithful model/config与project test hygiene分层冻结 |
| `theory_check` | external native comparison不是matched causal attribution；不同weight/horizon contract必须分表 |
| `design` | `/tmp` shallow-clone read-only audit；未复制source、未安装依赖、未运行train/test |
| `narrative_gate` | not a method gate |
| `effectiveness_gate` | pending；没有result |
| `artifacts` | `configs/stage_c_modern_native_baseline_protocol.json` |
| `decision` | `source_set_frozen_protocol_repairs_required_before_prelaunch` |

[Fact] 四个官方source均可获得，且顶层repository均声明MIT license，但不能直接执行原始shell matrix。
SRSNet tree内一个继承自N-BEATS的`utils/losses.py`另带CC BY-NC 4.0 notice，必须确认它是否不在本次MSE
execution path并记录可分发source边界。CATS与TimePerceiver在每个training epoch读取test loader；CATS还有
一个ETTm2 H96 dataset identifier typo；SRSNet使用TFB normalized rolling metrics；ElasTST官方config包含
`limit_train_batches=10`与native H720 selector。上述问题必须在任何remote/test前解决。

## 2. Pinned official sources

检索与clone日期：`2026-07-20`。

| ID | Commit | Native role | Five-dataset coverage |
| --- | --- | --- | --- |
| ElasTST | `d49f7e41c2db7ac3208816225885b6e3f61c0fb3` | one checkpoint/dataset，multi-H test | 5/5 |
| CATS | `58854fc759d608ce400f378be83f4513960e505d` | per-H future-query model | 5/5 |
| TimePerceiver | `7e30cc07b51c709f408409fd60a34c81ae8990be` | per-H generalized target-position model | 5/5 |
| SRSNet | `6ee35d498f48eefecf84530b362b137de38e6592` | per-H selective-patch model | 5/5 |

所有source只在`/tmp/fatst_sc_mnb_*`审计，没有进入FATST tracked tree，也没有从`R_2026_FSA`导入任何内容。

## 3. Native weight and horizon contracts

### 3.1 ElasTST

- 每dataset训练一个checkpoint；
- native context length为96；
- train/validation prediction length均为720；
- 同一checkpoint原生评估`{24,96,192,336,720,1024}`；
- paper-facing只读取`{96,192,336,720}`；
- native selector是`val_weighted_ND`，不是FATST four-H mean MSE；
- config包含`limit_train_batches=10`，launch前必须以source-faithful smoke确认这不是未清理debug配置。

因此ElasTST是唯一P0 `single_weight_varied_horizon`外部baseline。不得为了匹配FATST selector而静默修改其
native checkpoint rule；若未来增加matched selector，必须作为第二种control角色另表。

### 3.2 CATS

- official 512-input scripts对五datasets分别训练四个H-specific checkpoints；
- seed固定2021，loss为MSE，early stopping只读取validation MSE；
- training loop仍每epoch计算并打印test loss；
- `scripts/ETTm2_512_input.sh`的H96 command中，`data_path=ETTm2.csv`与`model_id=ETTm2_512_96`，
  但`--data ETTm1`，属于明确script typo。

允许的最小protocol patch仅有：

1. 删除training loop中的`test_data/test_loader`与per-epoch `test_loss`计算/打印；
2. 将该H96 `--data`改为`ETTm2`；
3. 调整GPU、output path和非语义调度。

其余model/hyperparameter/validation逻辑保持source-faithful。

### 3.3 TimePerceiver

- official scripts覆盖五datasets与四H；
- 每个H独立训练，不能列入single-weight table；
- native context length为384，seed为2025；
- `generalized=1`、`standard=0`，使用MSE与validation early stopping；
- training loop同样每epoch读取test loss；
- paper README报告对input length `{96,384,768}`的平均结果；本matrix预先固定official script的384，
  禁止观察test后选择input length。

允许的source patch只移除training期test evaluation并改变调度/output path。

### 3.4 SRSNet

- official TFB scripts覆盖五datasets与四H，每个cell使用独立checkpoint；
- seed为2021，`deterministic=efficient`；
- 不同dataset/horizon使用原生不同seq length、patch length、head与dropout等hyperparameters；
- evaluation是TFB `rolling_forecast`，主要输出`mse_norm/mae_norm`；
- official scripts硬编码GPU2–5并混用background jobs。
- top-level repository为MIT，但`ts_benchmark/baselines/srsnet/utils/losses.py`保留N-BEATS
  CC BY-NC 4.0 notice；当前source grep未证明该文件进入MSE execution path，因此只登记为
  `file_level_license_trace` blocker，不能据此宣称整个tree无额外许可边界。

在metric-equivalence完成前，SRSNet只能作为native fixed-H baseline。允许调整GPU/output/scheduling，但不得把
horizon-specific profiles改成共享profile或按FATST test结果重选。

## 4. Test-governance repair

CATS/TimePerceiver的per-epoch test loss不参与optimizer、scheduler或early stopping，但仍违反本项目“test labels
不得参与ordinary development”的规则。故不能以“官方代码如此”为理由保留。

future local prelaunch应验证：

1. hygiene patch后training graph、optimizer、scheduler和validation-selected checkpoint路径未改变；
2. source与patched tree只存在预注册的最小diff；
3. train/validation smoke不创建test metrics；
4. full training结束后只对冻结checkpoint执行一次test；
5. raw predictions/targets、MSE、MAE、params与effective config均保存。

这类修改属于protocol hygiene，不是本地method adaptation；结果必须标记
`official-source model/config + FATST test-hygiene patch`，不能声称byte-identical official execution。

## 5. Planned matrix and roles

若未来通过prelaunch并获得独立授权：

- ElasTST：5 runs，20 paper-facing cells；
- CATS：20 runs，20 cells；
- TimePerceiver：20 runs，20 cells；
- SRSNet：20 runs，20 cells；
- external total：65 runs、80 cells；
- A6_FULL/A6_MEASURE复用已有frozen evidence，不因baseline comparison重新训练或调profile。

结果分表：

1. `single_weight_varied_horizon`：ElasTST、A6_FULL、A6_MEASURE；
2. `native_fixed_h_accuracy`：CATS、TimePerceiver、SRSNet，并列A6作为accuracy reference但明确weights contract不同；
3. foundation/pretrained baselines如未来需要，单独table。

## 6. Remaining blockers

在进入local implementation/prelaunch前仍需：

1. 证明五个外部source使用的raw CSV、channel order与split和FATST reference可映射；
2. 解决ElasTST `limit_train_batches=10`语义；
3. 为SRSNet完成executed-file license trace，确认CC BY-NC file是否在本次MSE path之外；
4. 为SRSNet证明`mse_norm/mae_norm`与FATST metric的可比边界，并冻结prediction artifact export；
5. 实现source-tree diff checker与80-cell completeness schema；
6. 在任何test access前冻结carrier-viability thresholds；
7. 另行获得remote training与official-test授权。

## 7. Failure attribution boundary

当前没有performance result，不能判断A6 pass/fail。source fault也不能归因为baseline model失败：

- official script typo属于`protocol_fault`；
- per-epoch test access属于`evaluation_governance_fault`；
- file-level license notice属于`distribution_boundary_unresolved`；
- metric/split不等价属于`comparison_unresolved`；
- 只有完整、修复后的native matrix才可判断`carrier_viable / carrier_not_viable / unresolved`。

## 8. Current authorization

`source_audit=true / local_protocol_patch=false / remote_training=false /
official_test=false / paper_method=false / D25=false`。
