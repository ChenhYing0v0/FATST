# Phase5-B0 Future Supervision Pressure Audit

本文档是 Stage B problem redefinition 后的 source/code audit。B0 仍是 diagnostic-only；本文件不启动
remote，也不提出 method。

## 11-Step State

| Field | Content |
| --- | --- |
| `current_step` | Step 8：B0 diagnostic remote gate running |
| `problem` | 需要判断 TimeAlign future branch 的 reconstruction/alignment pressure 是否存在 useful/harmful structure |
| `existence_evidence` | Stage A head route 已暂停；A7DG 显示 official-last stability 有条件性；A4/A4R/A4S 显示 existing-path reliability signal 不足 |
| `idea` | 先审计 `recon_loss` 与 `align_loss` 的 tensor/gradient path，再决定是否做 pressure ablation diagnostic |
| `theory_check` | 如果 future branch pressure 是关键变量，应能通过 `w_recon/w_align` ablation 或新增 logging 观察到与 prediction/drift 的关系 |
| `design` | B0 分两步：source/code audit -> diagnostic-only local/remote design；当前已完成最小实现与 CPU smoke |
| `narrative_gate` | diagnostic-only pass：问题值得审计，但还不能作为 method candidate |
| `effectiveness_gate` | pending；需要后续 diagnostic artifacts |
| `artifacts` | `TimeAlign.py`、`train_repo.py` code path audit；`train_repo.py` B0 controls；remote wrapper；CPU smoke `/tmp/fatst-b0-smoke`；remote root `/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b0_future_supervision_pressure_audit` |
| `decision` | B0 remote gate 已启动；等待 artifacts 后做 problem-existence analysis |

## Source Path

### Future branch construction

[Fact] `baselines/timealign_official/models/TimeAlign.py` 中 future branch 包括：

- `patch_emb_y`：future label sequence 的 patch embedding；
- `autoencoder`：future-side autoencoding path；
- `proj_y`：future reconstruction projection；
- `align`：history/future representation alignment loss。

Relevant code:

- `patch_emb_y` / `autoencoder` / `proj_y`: `TimeAlign.py:41-79`;
- future forward and alignment: `TimeAlign.py:511-528`;
- reconstructed future output: `TimeAlign.py:574-579`.

### Gradient path

[Fact] 在 forward 中：

```text
x_ = self.ffn[i](x)
y = y + self.autoencoder[i](y)
align_loss += self.align(x_, y.detach())
```

`y.detach()` 表示 alignment loss 不反向更新 future branch `y`；它把 future representation 作为 target，
主要约束 history-side encoder/ffn path。future branch自身主要由 reconstruction loss 约束。

[Fact] `train_repo.py` 中 final training loss 为：

```text
loss = pred_loss
     + teacher/self-teacher terms
     + official_args.w_recon * recon_loss
     + official_args.w_align * alignment_loss
     + optional stability regularizers
```

Relevant code: `train_repo.py:1034-1043`。

## Current Controls And Logging

| Item | Current State | B0 Implication |
| --- | --- | --- |
| `w_recon` | CLI `--w-recon` exists, default `1.0` | 可以直接做 reconstruction ablation |
| `w_align` | 来自 dataset preset `preset.w_align`，无 CLI override | B0 需要新增 `--w-align-override` 或等价控制 |
| train logging | 记录 `train_reconstruction_l1` 与 `train_alignment_loss` | 只有 epoch average，不足以解释 future-unit reliability |
| validation logging | 只记录 prediction `val_mean_mse` | 缺 validation recon/align 与 prefix/future-unit pressure |
| prefix direct modes | 每个 selected prefix forward 会得到 recon/align；当前 `prefix_samples=1` | 如果未来改多 prefix，需要避免重复计算 pressure 造成 confounder |

## Diagnostic Question

B0 应回答三个问题：

1. future reconstruction pressure 是否帮助或伤害 prediction？
2. future alignment pressure 是否是 official-last drift / dataset split 的来源？
3. harmful/useful pressure 是否能被 train/validation signals 观察到，而不是只能事后看 test horizon？

## Minimal Diagnostic Design

[Design] 若进入实现，先做 ETTh2/ETTm1/Weather × A6-LBF-r256 carrier 的 diagnostic-only ablation：

| Arm | `w_recon` | `w_align` | Role |
| --- | ---: | ---: | --- |
| `b0_lbf_base` | 1.0 | preset | existing A6-LBF-r256 reference，不必重跑若可复用 |
| `b0_no_recon` | 0.0 | preset | 检查 future reconstruction pressure |
| `b0_no_align` | 1.0 | 0.0 | 检查 history-future alignment pressure |
| `b0_no_future_pressure` | 0.0 | 0.0 | 检查 future branch pressure 总体作用 |

[Rule] 这些 arms 是 diagnostic/control，不是 method candidate。若某个 ablation 变好，也只能说明 problem
exists，不能直接写成 paper method。

## Required Code Before Remote

1. 新增 `--w-align-override`，默认 `None`，不改变历史行为；
2. `effective_config.json` 明确记录 effective `w_align` 和 `w_recon`；
3. training log 增加 weighted quantities：
   - `train_weighted_reconstruction_l1 = w_recon * recon_loss`;
   - `train_weighted_alignment_loss = w_align * alignment_loss`;
4. 若成本可控，新增 validation-side prefix diagnostics：
   - prediction MSE by prefix；
   - optional reconstruction/alignment proxy by validation batch；
5. analyzer 必须区分 `loss-only ablation` 和未来可能的 `gradient-path routing`。

## Implementation And Local Verification

[Fact] 已完成最小代码更新：

- `baselines/timealign_official/train_repo.py` 新增 `--w-align-override`，默认 `None`，不改变历史 preset；
- `build_official_args` 将 effective `w_align` 写入 `official_args`，因此 `effective_config.json` 可直接审计；
- `training_log.csv` 新增：
  - `train_weighted_reconstruction_l1 = w_recon * train_reconstruction_l1`;
  - `train_weighted_alignment_loss = w_align * train_alignment_loss`;
- `scripts/remote/run_phase5_stage_b0_future_supervision_pressure_audit.sh` 已提供 ETTh2/ETTm1/Weather
  × `b0_no_recon/b0_no_align/b0_no_future_pressure` 的 diagnostic remote wrapper。

[Verification] 本地验证已通过：

- `python -m py_compile baselines/timealign_official/train_repo.py`;
- `bash -n scripts/remote/run_phase5_stage_b0_future_supervision_pressure_audit.sh`;
- CPU smoke：`ETTh2`、`A6-LBF` carrier、`--w-recon 1.0 --w-align-override 0.0`、
  `max_train_batches=1`、`max_eval_batches=1`。

[Fact] CPU smoke 的 `effective_config.json` 显示 `official_args.w_align = 0.0`，`training_log.csv`
显示 `train_weighted_alignment_loss = 0.0`。这证明 B0 可以在不改默认 TimeAlign preset 的前提下隔离
alignment pressure。

## Remote Launch

[Fact] B0 diagnostic remote gate 已在 529_Lab-3090 启动：

| Item | Value |
| --- | --- |
| commit | `8d802f5` |
| remote PID | launcher shell `2561335`; wrapper `2561337` |
| output root | `/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b0_future_supervision_pressure_audit` |
| launcher log | `/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b0_future_supervision_pressure_audit/_launcher/b0_future_supervision_pressure_launcher.log` |
| GPU preflight | GPU 0/1/2 all free：`18 MiB used`、`24107 MiB free`、`0% util` |
| after launch | Weather three arms running on GPU 0/1/2；about `4431/4432/4432 MiB` used |

[Run Matrix] ETTh2/ETTm1/Weather × `b0_no_recon/b0_no_align/b0_no_future_pressure`，carrier 为
A6-LBF-r256，checkpoint policy 为 `official-last`。

## Narrative Gate

| Gate Item | Assessment |
| --- | --- |
| problem motivation | pass for diagnostic：Stage A 已证明 head-only 不足，future branch pressure 是下一自然变量 |
| tensor/gradient path | pass：alignment 主要约束 history path，reconstruction 约束 future branch |
| novelty | pending：B0 只是问题验证；method 仍需后续 narrative gate |
| risk | high：若只发现 `w_align=0` 更好，容易退化成 loss ablation paper |

## Decision

[Decision] B0 source/code audit 与 local verification 通过，remote gate 已启动。远程结果只用于判断
future supervision pressure problem 是否真实存在；任何正向 arm 都不能直接升级为 paper-core method，必须重新进入
Step 4-6 narrative gate。
