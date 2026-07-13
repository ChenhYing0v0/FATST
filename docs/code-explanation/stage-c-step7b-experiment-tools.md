# StageC Step 7B Experiment Tools

## 1. Experiment Contract

Step 7B只测试decoder architecture，不改变frozen natural carrier的training semantics：

```text
history: [B,720,C]
-> frozen dataset profile Encoder
-> hidden: [B,C,768]
-> one of five readouts
-> normalized forecast: [B,720,C]
-> full-H720 pointwise L1 training
```

checkpoint由H720 validation MSE选择。test时每个sample只生成一次H720 forecast，再由同一prediction crop出
H1..H720 metrics。这里的“full-crop”是evaluation aggregation，不改变任一decoder的native prefix forward；
trained checkpoint另做native-prefix versus full-crop invariant audit。

## 2. Training Adapter Changes

### 2.1 Dense prefix metrics

`train_repo.metric_rows`先从`pred/true: [N,720,C]`计算逐step squared/absolute error，再沿time做
`cumsum`。H步MSE为`cumulative_squared[H-1] / H`，与直接对`[:, :H, :]`求MSE数学等价，但把
H1..H720从quadratic repeated slicing降为一次error construction和一次cumulative aggregation。

### 2.2 Evaluation routing

- `--evaluation-prefix-mode native`：保留原有逐H native forward，作为默认兼容路径；
- `--evaluation-prefix-mode full-crop`：单次H720 forward后聚合所有prefix metrics，供Step 7B使用；
- `--segment-horizons`：只为`48/96/192/336/720`生成segment metrics，避免对720个H重复生成segment表；
- `--no-save-predictions`：screening不保存大型test NPZ，checkpoint仍保留，可在通过后重新导出；
- `protocol_class=method_screening`：要求profile name/hash，但允许冻结协议后的test evaluation。

默认值保持原行为，因此历史runner不会被静默改成full-crop或停止保存predictions。

## 3. Remote Runner

`scripts/remote/run_stage_c_step7b_pmfo_rct.sh`从frozen contract读取三个dataset profiles，构造15 jobs：

```text
datasets = Weather, ETTm1, ETTh2
arms = PMFO-RCT, no-conservation, no-transition, dense matched, A6
seed = 2021
```

jobs按slow-dataset-first和slow-arm-first排序，再由固定GPU workers round-robin领取；每个worker始终绑定同一
GPU并串行执行自己的queue，避免动态`wait -n`后把新job错误放到仍占用的GPU。runner记录commit、contract
hash、GPU memory、command contract、jobs TSV与逐run log。`DRY_RUN=1`执行五arms synthetic invariant
smoke；`STATUS_ONLY=1`报告metrics/invariant完成数。

## 4. Trained Checkpoint Invariants

`check_stage_c_step7b_checkpoint_invariants.py`从每个run的`effective_config.json + checkpoint.pt`重建模型：

- 所有arms检查H=`1/48/96/192/336/720`的native prefix与H720 crop最大差；
- `pmfo-rct`额外检查parent/detail recovery、detail conservation和support locality；
- `trained_invariants.json.pass`要求所有适用误差finite且不超过`1e-6`。

该检查证明checkpoint没有破坏结构contract；它不证明forecast effectiveness。

## 5. Analyzer Statistics

`analyze_stage_c_step7b_pmfo_rct.py`定义：

- `dense_mse_auc`：H1..H720共720个prefix MSE的算术平均，即uniform horizon measure下的risk；
- `dense_mae_auc`：对应prefix MAE平均；
- `pmfo_vs_a6_improvement_pct = (1 - PMFO_AUC / A6_AUC) * 100`，正值表示PMFO更好；
- `best_control`：每个dataset在dense/no-transition/no-conservation中AUC最低者；
- macro improvement：先计算每个dataset的relative improvement，再对三个dataset等权平均，避免某个dataset
  的metric scale支配结论。

`run_summary.csv`记录逐dataset-arm AUC、标准horizon metrics、epoch、validation与invariant；
`comparisons.csv`记录相对A6结果；`step7b_gate.json`保存预注册gate与failure attribution；报告只允许
`partial_pass`或明确rollback，单seed不能写成effectiveness pass。

## 6. Verification Boundary

本地`check_stage_c_step7b_local.py`验证cumulative metric与直接MSE/MAE等价，并对五arms比较native与
full-crop metrics。remote dry-run只验证command matrix和synthetic structure，不声称training成功。
