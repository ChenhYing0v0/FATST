# StageC SC0 Mechanism-Control Code Explanation

## Scope

本次代码只实现 `SC0-MCP` standardized carrier calibration，不实现 `SC1-PFO` decoder或 `SC2-HML`
training method。SC0 是 validation-only control，用于选择后续统一冻结的 research carrier。

## Functional Modules

### 1. Configuration source

`configs/stage_c_mechanism_control.json` 是 SC0 的唯一 profile source，定义：

- datasets与 training/validation/evaluation horizons；
- neutral full-720 objective；
- optimizer、LR、effective batch、epoch与checkpoint policy；
- 三个 `P*D=1536` capacity-matched arms；
- global validation regret gate。

runner计算 config bytes 的 SHA256，并将 `protocol_class/profile/hash` 写入每个 run 的
`effective_config.json`。后续 analyzer拒绝读取 hash或 protocol class不一致的 artifacts。

### 2. Training and validation separation

`baselines/timealign_official/train_repo.py` 新增三个 horizon fields：

```text
target_horizons      -> construct the training objective
validation_horizons  -> select best checkpoint
evaluation_horizons  -> export diagnostic metrics after training
```

SC0 的实际 contract 为：

```text
target_horizons=[720], pred_loss_mode=full
    -> one L1 loss on prediction[:, :720, :]

validation_horizons=[720]
    -> one full-720 validation MSE per epoch

evaluation_horizons=[48,96,144,192,288,336,512,720]
    -> dense validation-prefix metrics for last/best states
```

因此 config selection不含 `{96,192,336,720}` nested-prefix exposure，也不读取 test split。

### 3. Standardized Encoder overrides

SC0 沿用 clean TimeAlign token-MLP computation：

```text
x [B,720,C]
  -> Normalize
  -> per-channel non-overlap patches [B,C,P,720/P]
  -> Linear patch projection [B,C,P,D]
  -> two shared residual token MLP layers
  -> hidden flatten [B,C,P*D] = [B,C,1536]
  -> learned_basis_coeff [B,C,256]
  -> learned_temporal_basis[:H] @ coeff
  -> prediction [B,H,C]
```

新增 `legacy_layer_norm` override，保证 Weather不再继承 upstream `layer_norm=0`。三 dataset在 SC0 下都
使用 LayerNorm、dropout `0.1`、LR `1e-4`与相同 architecture budget。

### 4. Effective batch semantics

新增 `gradient_accumulation_steps`。每个 micro-batch loss先除以 accumulation steps再 backward；到达完整
accumulation window或 epoch最后一个有效 batch时才执行 optimizer step。日志同时写出：

- `gradient_accumulation_steps`；
- `effective_batch_size=batch_size*gradient_accumulation_steps`；
- `validation_horizons`。

SC0 当前 profile使用 `32*1=32`，但该实现允许因GPU memory改变 micro-batch而不改变 effective batch。

### 5. Validation-only artifact boundary

`final_evaluation_split=val` 时，训练后只构造 validation loader，输出：

```text
metrics_last_by_target_horizon.csv
metrics_best_val_by_target_horizon.csv
metrics_last_by_segment.csv
metrics_best_val_by_segment.csv
predictions_val.npz
```

metric rows新增：

- `evaluation_split`：artifact来自哪个 split；
- `protocol_class`：`source/mechanism_control/native_external`；
- `protocol_profile`：profile stable name；
- `profile_hash`：config SHA256。

当 `protocol_class=mechanism_control` 时，CLI禁止 `final_evaluation_split=test`，从执行路径阻止 SC0 用
test set选 config。

### 6. Local checker

`scripts/check_stage_c_sc0_carrier_local.py` 分两层验证：

1. structural gate：patch boundary、memory shape、state width、active/unused params、prefix consistency、strict reload；
2. end-to-end gate：三个 datasets乘三个 arms各跑 one-batch CPU smoke，验证 validation-only metrics、
   dense horizons、dual checkpoints、profile hash与artifact filenames。

### 7. Remote runner

`scripts/remote/run_stage_c_sc0_carrier_calibration.sh` 从 JSON读取全部 common settings与arms，默认输出到：

```text
/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc0_carrier_calibration
```

runner按 `Weather -> ETTm1 -> ETTh2` dataset-major order提交任务，使用 `wait -n` 填充 GPU slots。启动时记录
git commit/status、GPU memory、profile hash、effective batch、dataset与output root。全部九 runs完成后调用
analyzer。

### 8. Analyzer statistics

`scripts/analyze_stage_c_sc0_carrier_calibration.py` 只读取 validation artifacts。对 dataset $d$ 和 arm $g$：

$$
r_{d,g}=\frac{\operatorname{MSE}^{val}_{d,g}}
{\min_{g'}\operatorname{MSE}^{val}_{d,g'}}-1,
\qquad
S_g=\frac{1}{|D|}\sum_d r_{d,g}.
$$

输出字段含义：

- `macro_regret`：$S_g$，跨dataset平均归一化regret；
- `max_dataset_regret`：最差dataset的$r_{d,g}$；
- `per_dataset_gate_pass`：所有dataset regret是否不超过3%；
- `selected`：按macro score与预注册tie-break选出的单一global arm；
- `mean_epoch_seconds`：仅在macro score差异小于0.5%时作为efficiency tie-break；
- `selector_stable`：last与best-val是否选择同一global arm。

seed2021通过只能得到 `preliminary_global_profile_selected_needs_seed_confirmation`，不能直接冻结profile。

## Code-Theory Consistency

### Intended theory

机制归因需要一个跨dataset相同、非 `patch_num=1`、capacity-matched且不受test selection影响的carrier。

### Code realization

- 三 arms的 `P*D`相同；active参数差异低于0.1%；
- dataset-specific TimeAlign width/dropout/LR/LayerNorm被显式override；
- training、checkpoint selection和dense diagnostic horizons分离；
- mechanism-control路径禁止test evaluation；
- config hash贯穿runner、artifacts和analyzer。

### Remaining proxies

- 相同active parameter count不等于相同optimization geometry或effective capacity；
- token-MLP family本身可能无法形成跨dataset稳定carrier；
- validation regret gate只选择research instrument，不证明该配置是SOTA或paper method。

### Falsification evidence

若没有global arm在所有datasets满足3% regret，或last/best选择不同winner，则SC0失败并回StageC Step 2/3。
不得以逐dataset preset、test-based tuning或新增第四个arm来绕过该gate。
