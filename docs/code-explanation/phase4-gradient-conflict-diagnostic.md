# Phase4 Gradient Conflict Diagnostic

## Purpose

本脚本服务于 Phase4 的主线升级：从 loss-only HSS 转向
gradient-routing / representation-aware HSS。它不直接证明新方法有效，而是检验
新方法的前提：

> 不同 future supervision units 是否会在 shared representation parameters 上产生
> 冲突梯度。

如果冲突成立，后续 `adapter-isolated` 或 `detach-shielded` supervision strategy
才有理论与经验动机；如果不成立，应回退到 predictability proxy，而不是做架构升级。

## Script

- entry: `scripts/analyze_phase4_gradient_conflict_diagnostic.py`
- remote runner: `scripts/remote/run_phase4_gradient_conflict_diagnostic.sh`
- default remote output:
  `/home/yingch/exp_outputs/r-2026-fatst/phase4_gradient_conflict_diagnostic`

## Computation Flow

1. 构建 `PatchEncoderTargetSetDecoder`。
2. 对每个 dataset 使用 train split、`pred_len=720`。
3. 若未提供 checkpoint，先用 full-time MSE 做短 warmup。
4. 对同一个 batch forward 一次，得到 `pred`。
5. 构造多个 supervision group loss：
   - `early_1_96`
   - `middle_97_192`
   - `middle_193_336`
   - `late_337_720`
   - `learnable_hard`
   - `noisy_hard`
   - `predictable_easy`
   - `full_1_720`
6. 对每个 loss 分别 backward，收集参数梯度。
7. 按参数组计算 pairwise gradient cosine：
   - `encoder`
   - `target_path`
   - `readout_head`
   - `all_shared`

## Bucket Definition

`learnable_hard` / `noisy_hard` 使用与 Phase4-S2 一致的 train-side block proxy：

- `label_novelty`: block target 相对 history last value 的 MSE；
- `local_variation`: block 内一阶差分能量；
- `noisy_hard`: `top_novelty ∩ top_variation`；
- `learnable_hard`: `top_novelty - top_variation`；
- `predictable_easy`: 非 `top_novelty` blocks。

这不是最终 predictability 定义，只是为了诊断 S1/S2 已经使用过的 proxy 是否真的造成
shared-gradient conflict。

## Output Files

- `phase4_gradient_conflict_pairs.csv`:
  batch-level pairwise cosine。
- `phase4_gradient_conflict_summary.csv`:
  按 dataset、loss pair、parameter group 聚合后的 mean/min cosine、
  negative share 和 low-cosine share。
- `phase4_gradient_conflict_block_trace.csv`:
  每个 batch 的 learnable/noisy/predictable block 数量。
- `phase4_gradient_conflict_report.md`:
  面向研究决策的 11-step 记录与重点 pair 摘要。

## Code-Theory Consistency

[Theory] SRP/SRP++ 的启发不是直接复制 step-specific LoRA，而是指出 multi-step
forecasting 可能存在 step/segment representation interference。

[Code] 本脚本通过同 batch、同模型、不同 supervision group 的 backward 来估计这种
interference 是否在当前 carrier 中可观测。

[Proxy] Gradient cosine 只能反映局部优化方向，不等价于最终泛化性能；warmup 步数也会影响
冲突强度。

[Falsification] 如果 Weather 的 `noisy_hard` vs `early_1_96` /
`predictable_easy` 在 `encoder`、`target_path`、`readout_head` 上没有明显更低的
cosine 或 negative share，则不能把 S2 失败解释为 representation conflict，下一步应回到
predictability proxy 或 residual stability 诊断。
