# StageC D15-A Native PCSD Direct-Control Protocol

## Status

| Field | Value |
| --- | --- |
| `stage` | `StageC-UVHF` |
| `current_step` | `SC1-PCSD-CF-v1` Step7A local implementation next |
| `role` | architecture candidate + Contribution-2 problem diagnostic |
| `narrative_gate` | conditional pass |
| `local_implementation` | authorized |
| `remote_training` | false |
| `test_access` | false |
| `config` | `configs/stage_c_pcsd_cf_native_direct.json` |
| `analysis` | `analysis/stage_c_pcsd_native_reset_20260716/pcsd_cf_step46_source_theory_design_audit.md` |

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

## Gates And Failure Attribution

PCSD-CF method gate要求`DIRECT`至少3/5 datasets超过A6且macro `>=0.3%`，并至少3/5超过equal/static且macro
各`>=0.2%`；dense/random controls不得解释收益，arms必须有skill/separation且policy不得collapse。

- M0 invalid：implementation/theory mismatch；
- M0 valid但fixed arms弱：`readout_or_head_design_wrong`；
- dense/random复制收益：`capacity_control_explains`；
- arms skilled、有same-run headroom但direct policy misallocates：只支持SC2 problem，不能自动通过新方法；
- numeric/collapse/reversal：`optimization_or_numeric_pathology`，不得方向级拒绝。

## Decision

`authorize_step7a_local_only`。D14-B1/CCRL不再active。remote、test、effectiveness claim与SC2 implementation均
保持false。
