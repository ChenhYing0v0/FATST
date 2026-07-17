# StageC 公平重评估代码说明

## 1. 变更范围

本次没有修改任何model forward或参数化。新增内容只负责：

1. 构造A6/PCSD/PCC/SIFF的matched训练矩阵；
2. 用四个validation horizons选择checkpoint；
3. 在official test统一计算paper-facing scorecard；
4. 审计checkpoint、initialization与完整矩阵；
5. 汇总逐dataset-horizon相对增益。

## 2. 训练与checkpoint数据流

`scripts/remote/run_stage_c_fair_reaudit_v1.sh`读取
`configs/stage_c_fair_reaudit_v1.json`与five-dataset profile contract。每个job均执行：

1. history输入`batch_x [B,720,C]`进入TimeAlign-derived Encoder；
2. A6或PCSD/SIFF readout生成完整`forecast [B,720,C]`；
3. training仍以full-T loss及对应objective更新全部Encoder–Decoder参数；
4. 每个epoch分别计算validation的H96/H192/H336/H720 prefix MSE；
5. `train_repo.validation_mean_mse`对四项取算术平均，保存最低score的`checkpoint.pt`。

因此checkpoint由multi-horizon validation task family选择，但任何test label都不进入训练或checkpoint路径。

## 3. Test evaluator

`scripts/evaluate_stage_c_pcsd_cf_checkpoint.py`现支持machine-readable的通用test audit authorization，同时保留
旧PCSD-CF audit兼容性。对选定checkpoint：

1. 从official test loader读取`batch_x [B,720,C]`与`target [B,720,C]`；
2. 只生成一次完整`forecast [B,720,C]`；
3. 得到逐time coordinate的squared/absolute error；
4. 用cumulative sum得到每个prefix H的MSE/MAE；
5. 写出`test_audit_metrics_by_target_horizon.csv`和checkpoint SHA256；
6. 验证test authorization、from-scratch contract、full-prefix identity与readout contract。

四个standard horizons与dense H1..720因此来自同一个完整prediction，不存在不同horizon forward造成的输出漂移。

## 4. Prelaunch checker

`scripts/check_stage_c_fair_reaudit_v1.py`生成70-job manifest，并通过真实CLI parser验证每个job。它对每个
unique dataset/readout/rank实例化模型，检查：

- `prefix(H96) == full[:96]`；
- model construction有限且无异常；
- 相同dataset与seed下各arms的Encoder initialization hash一致；
- profile hash、矩阵大小、comparison引用与test authorization一致。

这保证比较是joint E2E、from-scratch且paired initialization，而不是冻结替换实验。

## 5. Analyzer与统计定义

`scripts/analyze_stage_c_fair_reaudit_v1.py`读取每个run的：

- `effective_config.json`；
- `training_log.csv`；
- `initialization_contract.json`；
- `checkpoint.pt`；
- `test_audit_invariants.json`；
- `test_audit_metrics_by_target_horizon.csv`。

对每个dataset、arm和H96/H192/H336/H720生成long-form table。相对增益定义为
$100(1-\mathrm{MSE}_{candidate}/\mathrm{MSE}_{reference})$。`macro gain`对20个
dataset-horizon cells等权；`dataset win`先在一个dataset内平均四horizon gains；`horizon win`先跨五dataset
平均。任何缺失run、hash mismatch、非有限metric或protocol mismatch都会阻止summary生成。

## 6. Code-theory consistency

- Intended theory：统一multi-horizon模型必须由多horizon checkpoint selector选择，并在相同test task family上
  判断effectiveness。
- Code realization：四validation prefixes等权决定checkpoint；同一full forecast的四test prefixes形成primary
  scorecard；所有arms共享profile、seed、optimizer class与训练预算。
- Proxy boundary：seed2021仍是Phase A screen，不是最终统计稳定性证据；test已被历史工作访问，因此结果是
  `test_informed` benchmark evidence。
- Falsification：任一Encoder init不匹配、checkpoint被test阶段修改、280 cells不完整、或candidate只在未预注册
  metric上正向，都不能通过本次gate。

## 7. Step9 attribution analyzer

`scripts/analyze_stage_c_fair_reaudit_step9.py`同时读取聚合test scorecard和各run的validation
`metrics_by_target_horizon.csv`。它输出：

- `comparison_scorecard.csv`：每个comparison在validation/test、MSE/MAE上的macro gain、cell wins、
  dataset wins、horizon wins与pre-registered gate；
- `comparison_test_cells.csv`：每个comparison的20个test MSE cells；
- `all_arms_vs_a6.csv`：所有非A6 arms相对A6的exploratory MSE/MAE scorecard；
- `checkpoint_epochs.csv`：每个run实际训练epoch、selected best epoch与early-stopping状态；
- `step9_attribution.json`：区分performance pass、mechanism specificity、missing controls和rollback。

其中validation/test相对增益使用各自split的同dataset、同horizon MSE或MAE；不会跨dataset直接平均raw loss。
`all_arms_vs_a6.csv`属于结果解释，不是事后新增的method gate。
