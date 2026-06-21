# Phase0 Segment-wise Checkpoint Oracle

## 目的

该分析回答：在统一 `0-720` forecast interval 内，每个小段最适合使用哪个
`pred_len` checkpoint。短 checkpoint 无法直接覆盖长区间时，用 rolling autoregression 扩展到
720。

## Forward / Evaluation Flow

For each dataset and checkpoint horizon $H_c \in \{96,192,336,720\}$:

1. Load `PatchEncoderFixedHead` checkpoint trained with `pred_len=H_c`.
2. Build the `test` dataset with `pred_len=720`, so all checkpoints are evaluated on the same
   target windows.
3. Run rolling forecast:

```text
context_0 = x[:, -336:, :]
pred_1 = model_Hc(context_0)
context_1 = concat(context_0, pred_1)[:, -336:, :]
pred_2 = model_Hc(context_1)
...
pred_720 = concat(pred_1, pred_2, ... )[:, :720, :]
```

4. Split the 720-step forecast into 48-step segments:

```text
0-48, 48-96, ..., 672-720
```

5. Compute MSE/MAE for each segment and select the checkpoint with lowest segment MSE.

## Metrics

`phase0_segment_oracle_metrics.csv` stores every candidate:

- `dataset`
- `segment_start`
- `segment_end`
- `checkpoint_pred_len`
- `roll_steps`
- `mse`
- `mae`

`phase0_segment_oracle_winners.csv` stores segment winners:

- `best_pred_len`
- `best_mse`
- `second_pred_len`
- `second_mse`
- `relative_margin_to_second`

## Interpretation Boundary

[Fact] This is an oracle diagnostic over already trained horizon-specific checkpoints. It is not an
inference policy that should be used directly in the final model.

[Strong Evidence] If early segments prefer short checkpoints while later segments prefer long
checkpoints, then a one-size fixed head is not naturally aligned with variable-horizon behavior.

[Speculative] If rolling short checkpoints dominate many late segments, the model may be using local
autoregressive correction better than direct long-horizon projection. That would motivate a Phase1
decoder that can adapt computation by requested horizon/segment.
