# B14 Prerequisite Hierarchical Patch Memory Gate Report

> Superseded interface note 2026-07-10：本报告验证的是 initial right-padded 30-patch interface。A6 exact
> preservation结论仍有效；B14 Step 3为避免复制末端 evidence，使用 later valid 29-patch interface，并逐 batch
> 验证 manual slice、overlap-add reconstruction与 forecast equivalence。

## Conclusion

[Decision] `hierarchical_patch_memory_ready`。full contextual encoder replacement失败后，Step 5/6 repair在
ETTh2、ETTm1、Weather 全部通过 strict functional-equivalence gate。A6 forecast carrier性能和计算图保持
不变，同时得到跨 datasets一致的 canonical local memory `[B,C,30,48]`。

该结果解除 B14 Step 3 diagnostic 的 carrier blocker，但不验证任何 trainable retrieval mechanism。

## Gate Results

| Dataset | State keys | Parameters legacy/HPM | First-batch output max diff | Full metric max diff | Retrieval memory |
| --- | --- | ---: | ---: | ---: | --- |
| ETTh2 | equal | `1,690,016 / 1,690,016` | `0.0` | `0.0` | `[32,7,30,48]` |
| ETTm1 | equal | `884,640 / 884,640` | `0.0` | `0.0` | `[32,7,30,48]` |
| Weather | equal | `6,250,656 / 6,250,656` | `0.0` | `0.0` | `[32,21,30,48]` |

验证范围：

- clean A6 seed-2021 checkpoints；
- prefixes `{96,192,336,720}`；
- first test batch逐元素比较；
- full test MSE/MAE比较；
- `strict=True` state-dict load；
- parameter count与 state keys比较。

## Final Encoder Contract

```text
x [B,720,C]
  -> Normalize
  ├─ accepted A6 carrier encoder
  │    -> carrier_memory [B,C,P_legacy,D_legacy]
  │    -> flatten -> coeff [B,C,256]
  │    -> basis[:H] -> prediction [B,H,C]
  └─ CanonicalPatchMemory(K=48,S=24)
       -> local_memory [B,C,30,48]
```

`local_memory` 无 parameters、无随机 projection、无 auxiliary loss，也不进入 current forecast output。因此：

- ETTm1 不再因 legacy `P=1` 而缺少 B14 patch axis；
- ETTh2/Weather 与 ETTm1 使用同一 local-memory tensor semantics；
- current A6 evidence不需要重跑或降级；
- trainable patch projection/retrieval只允许在 B14 problem gate通过后设计。

## Why The Repair Is Not A Residual Route

HPM 不计算 `prediction + correction`，也不修改 A6 output。它只把 history representation contract分成：

1. 已验证的 global/carrier state；
2. 为 future-unit retrieval保留的 local evidence memory。

后续 paper-core candidate若成立，必须让 future units在 primary generation path中读取 local memory，而不是
在 A6 prediction之后做 residual repair。

## Failure Attribution Closure

full CPE replacement的失败归因为 `readout_or_encoder_design_wrong`，不是 `hypothesis_false`：

- standard contextual backbone整体 `+4.14%/+4.80%`，不能替换 A6；
- exact HPM repair证明保留 carrier与建立 canonical patch axis可以同时做到；
- 但 local memory尚未被预测使用，因此这个 exact pass不是 retrieval effectiveness证据。

## Decision And Next Step

- close：`contextual-patch-transformer` 作为 active carrier replacement；
- retain：仅用于 negative control/traceability；
- accept prerequisite：`hierarchical-patch-memory` interface；
- active step：B14-FURD Step 3 problem diagnostic；
- gate：比较 `U180/U240` future-unit error demand 与 accepted A6 sensitivity，并按同一 30 个 overlapping
  patch supports聚合；
- prohibition：B14-A通过前不得实现 cross-attention/retrieval model。

## Artifacts

- `hierarchical_patch_memory_gate/ETTh2/seed2021/equivalence_summary.json`；
- `hierarchical_patch_memory_gate/ETTm1/seed2021/equivalence_summary.json`；
- `hierarchical_patch_memory_gate/Weather/seed2021/equivalence_summary.json`；
- `scripts/evaluate_phase5_stage_b_b14_hierarchical_patch_memory.py`；
- `scripts/remote/run_phase5_stage_b_b14_hierarchical_patch_memory_gate.sh`。
