# StageC D15-A Native PCSD Direct-Control Protocol

## Status

| Field | Value |
| --- | --- |
| `stage` | `StageC-UVHF` |
| `current_step` | `SC1-PCSD-CF-v1` Step7B prelaunch passed；seed2021 remote screen authorized |
| `role` | architecture candidate + Contribution-2 problem diagnostic |
| `narrative_gate` | conditional pass |
| `local_implementation` | passed（9/9 gate categories） |
| `remote_training` | seed2021 validation-only 60-run screen authorized；not yet launched |
| `test_access` | false |
| `config` | `configs/stage_c_pcsd_cf_native_direct.json` |
| `analysis` | `analysis/stage_c_pcsd_native_reset_20260716/pcsd_cf_step46_source_theory_design_audit.md` |
| `step7a_artifact` | `analysis/stage_c_pcsd_cf_step7a_local_20260716/step7a_local_gate_report.md` |
| `step7b_artifact` | `analysis/stage_c_pcsd_cf_step7b_prelaunch_20260716/prelaunch_gate_report.md` |

## What We Plan To Test

D14-A已证明独立trained coupling scopes有稳定互补性。D15-A不再训练五个完整models后再学习router，而测试：

> 一个共享history-to-future parameter field，能否仅通过future-coordinate scope pooling，在同一decoder中形成
> skilled point/block/global arms、精确包含A6，并由ordinary direct task loss学习history × target allocation？

该问题必须先于任何新training contribution。若direct policy自然工作，SC2 credit mechanism没有存在性；若失败，
也只有在same-run arms有skill/headroom且排除architecture、capacity与optimization解释后，才支持SC2问题。

## Tensor Contract

```text
memory [B,C,P,De]
  -> z [B,C,R]
  -> shared mode field Z [B,C,Dq=4,K=256]
  -> pooled group states A_s [B,C,G_s,K], s in {1,48,144,360,720}
  -> shared identity+GELU row synthesis
  -> arm forecasts [B,C,5,T=720]
  -> history x target policy [B,C,T,5]
  -> fused full forecast [B,C,T]
  -> prefix crop [B,H,C]
```

requested $H$只用于最后crop。五个scopes共享mode maps与target synthesis rows，不保存五个完整decoders。

## Step7A Local Gate

1. five natural profiles shape contract；
2. all dense horizons与arbitrary prefixes exact equality；
3. float64/float32 arbitrary-A6 containment；
4. scope Jacobian-sharing topology与random-parameter arm separation；
5. canonical/random partition parameter equality；
6. finite forward/backward、equal-logit initialization；
7. parameter/DoF/FLOP/activation accounting；
8. no requested-H feature、warm-start、frozen replacement或test access。

任一项失败只返回Step5/6修复，不启动remote。

### Returned local result

2026-07-16返回`overall_pass=true`：5 profiles × 13 horizons共65个direct prefix cases及5个真实A6-natural
integration cases均为exact crop（max gap `0`）；arbitrary-A6 containment的float32最大output/arm gap分别为
`3.815e-6/2.384e-6`，float64分别为`3.109e-15/5.329e-15`。point/block/global Jacobian-sharing classes为
`720/15/5/2/1`；修正fan-in初始化后canonical/random arm最小pairwise normalized RMSE分别为
`0.131493/0.023079`，初始policy
uniform gap为`0`。五profile module与ETTh2真实Encoder-PCSD E2E two-step gradients均finite/active。

PCSD coupling-field core相对A6 decoder参数为`3.0291-3.6184x`，含policy总计`3.1006-3.7224x`；static
FLOP估算为A6的`7.97-13.93x`。因此local contract成立，但Step7B的dense nonlinear capacity control与remote
memory/runtime smoke为mandatory，不能将local pass解释为性能优势。

## Step7B Frozen Arms

- `A6_LBF_E2E`；
- `PCSD_CF_M0` exact morphism control；
- `PCSD_CF_FIXED_{1,48,144,360,720}`；
- `PCSD_CF_EQUAL`；
- `PCSD_CF_STATIC_TARGET`；
- `PCSD_CF_DIRECT` primary；
- `PCSD_CF_RANDOM_PARTITION`；
- `DENSE_NONLINEAR_MATCHED`。

所有method/control使用相同A6-natural dataset profile、full-H720 pointwise L1、from-scratch E2E、best-validation
H720 checkpoint。seed2021 five-dataset screen需在Step7A通过后另行授权。

### Returned prelaunch result

Step7B local prelaunch返回4/4 categories pass、60/60 dataset-arm contracts pass：

- `A6_LBF_E2E`与`PCSD_CF_M0`按相同seed构造时operator initialization hash与初始输出严格相同，五profile
  maximum gap均为`0`；
- 五profile内所有12 arms的Encoder initialization hash一致，所有full PCSD arms的trainable parameter values、
  shapes与hash一致，policy/partition只改变active path或fixed buffers；
- 修正了Step7A实现中`mode_weight` Kaiming方向错误：现在按history width $R$使用
  $[-R^{-1/2},R^{-1/2}]$ uniform initialization，五profile empirical std与理论值误差均低于`0.06%`；
- `DENSE_NONLINEAR_MATCHED`按dataset state width自动选hidden width，decoder parameter gap约`0.01-0.04%`，
  低于冻结的`0.1%`；
- primary metric冻结为validation dense-H1..720 MSE AUC；secondary为H720 MSE与dense MAE AUC；所有run
  full-H720 L1、best-val-H720 checkpoint、full-crop validation，test=false。

runner固定12 arms × 5 datasets = 60 jobs，并按arm-major、slow-dataset-spread顺序把Weather/ETTm1/ETTh1分散到
不同GPU。正式矩阵前必须先通过Weather-direct、batch32、one-batch GPU resource smoke。

## Gates And Failure Attribution

PCSD-CF method gate要求`DIRECT`至少3/5 datasets超过A6且macro `>=0.3%`，并至少3/5超过equal/static且macro
各`>=0.2%`；dense/random controls不得解释收益，arms必须有skill/separation且policy不得collapse。

- M0 invalid：implementation/theory mismatch；
- M0 valid但fixed arms弱：`readout_or_head_design_wrong`；
- dense/random复制收益：`capacity_control_explains`；
- arms skilled、有same-run headroom但direct policy misallocates：只支持SC2 problem，不能自动通过新方法；
- numeric/collapse/reversal：`optimization_or_numeric_pathology`，不得方向级拒绝。

## Decision

`step7b_prelaunch_pass_remote_seed2021_authorized`。用户已明确授权在3090启动validation-only seed2021 screen；
先执行GPU audit与resource smoke，再启动60-run matrix。paper effectiveness claim、test、seeds2022/2023与SC2
implementation仍未授权。
