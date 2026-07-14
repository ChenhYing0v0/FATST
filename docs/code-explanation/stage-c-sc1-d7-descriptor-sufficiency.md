# SC1-D7 Descriptor Sufficiency Code Explanation

## Scope

D7是`diagnostic_only` frozen-memory probe，不是forecast model implementation。它读取冻结A6 encoder memory，
训练七个head-only arms，不更新encoder、不读取test，也不使用official validation做early stopping。

## Forward Tensor Flow

冻结A6路径：

```text
history [B,L,C]
  -> frozen A6 encoder
  -> memory [B,C,P,D]
  -> flatten/standardize h [B*C,R]
```

`free_m0` control：

```text
h [N,R] -> Linear(R,256) -> Linear(256,720)
  -> free RGNB coefficients alpha [N,720]
  -> alpha @ Q^T [720,720] -> prediction [N,720]
```

PAF arms：

```text
h [N,R] -> branch Linear(R,256) -> z [N,256]
descriptor d [720,8]
  -> Linear(8,W) -> tanh -> Linear(W,256)
  -> psi [720,256]
alpha = z @ psi^T + coefficient_bias -> [N,720]
prediction = alpha @ Q^T -> [N,720]
```

其中$W=256$为compact arm，$W=694$为near-A6-budget arm。GEO、PERM、RANDOM在同一width下拥有完全相同
的architecture、parameter count、initialization seed和optimizer；唯一差异是descriptor-to-atom assignment。

## Descriptor Controls

- `GEO`：canonical RGNB type/support/length/depth/order descriptor；
- `PERM`：固定seed7101打乱canonical descriptor rows，保持descriptor distribution；
- `RANDOM`：固定seed7102生成random rows，再逐列匹配GEO mean/std；
- 所有structure seeds跨dataset/checkpoint冻结，不能由结果选择。

## Data And Optimization Boundary

- train batches0-15只用于构造fit/inner-holdout；
- official validation batches16-23只用于最终evaluation；D4使用0-7，D6使用8-15；
- early stopping只读取train-derived inner holdout；
- objective固定为H720 evaluation-space MSE；报告八个cumulative horizons；
- test split、forecast checkpoint更新、dataset-specific arm tuning均禁止。

## Artifact Definitions

| Artifact | Definition |
| --- | --- |
| `d7_probe_metrics.csv` | 每dataset/checkpoint/arm的parameter、best epoch、fit/holdout H720与八个validation horizon metrics |
| `d7_training_history.csv` | 每epoch fit MSE与inner-holdout MSE |
| `d7_metadata.json` | checkpoint/profile/hash、memory shape、split rows、descriptor hashes、RGNB/projectivity invariants与环境 |
| `d7_horizon_comparisons.csv` | GEO相对median(PERM,RANDOM)的逐horizon MSE/MAE gain及相对free-M0 gap |
| `d7_checkpoint_summary.csv` | 15个dataset-checkpoint units × 两width的macro gain和fit-holdout gap |
| `d7_dataset_summary.csv` | 每dataset/width的三checkpoint方向与macro effect |
| `d7_summary.json` | completeness、numeric、freeze、parameter、descriptor、projectivity与hard-gate decision |

## Code-Theory Consistency

- intended theory：canonical RGNB geometry应对coefficient-row parameterization提供forecast-relevant inductive
  bias；
- code realization：GEO/PERM/RANDOM只改变固定descriptor rows，branch、trunk、basis、optimizer与seed完全匹配；
- projectivity：PAF逐atom独立计算，active descriptor subset必须复现full coefficients及prediction prefix；
- proxy boundary：head-only gain证明descriptor sufficiency，不等于end-to-end method effectiveness；
- falsification：两width任一无法超过PERM/RANDOM、跨dataset方向不足、MAE为负、matched arm远离free-M0或
  fit-only advantage过大，均阻断PAF进入Step 7。

## Reproduction

```bash
/opt/anaconda3/envs/r2026-fsa/bin/python \
  scripts/run_stage_c_sc1_d7_descriptor_sufficiency.py --synthetic-smoke
/opt/anaconda3/envs/r2026-fsa/bin/python \
  scripts/analyze_stage_c_sc1_d7_descriptor_sufficiency.py --synthetic-smoke
```
