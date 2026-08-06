# QDF official baseline adapter

This directory vendors the official implementation of *Quadratic Direct
Forecast for Training Multi-Step Time-Series Forecast Models* at the immutable
commit recorded in `UPSTREAM_SOURCE.md`.

## Result roles

- ETTh1, ETTh2, ETTm1, ETTm2, ECL, and Weather use the per-horizon QDF values
  reported in the paper's Table 6.
- Solar is reproduced locally with `scripts/Solar.sh` because neither the paper
  nor the official repository reports it.
- QDF is a fixed-horizon native external baseline. It provides accuracy context
  for Main I and is not a matched unified or mechanism-attribution control.

## Solar contract

- model/objective: upstream `TQNet` + QDF;
- lookback: 96;
- horizons: 96, 192, 336, 720;
- seed: 2023, matching the upstream scripts;
- loader: upstream `Dataset_Solar`, 70/10/20 chronological split and train-only
  standardization;
- source data: `Solar/solar_AL.txt`, 52,560 rows, 137 variables;
- hyperparameters: ECL per-horizon settings transferred without tuning;
- Solar-only semantic changes: `enc_in=dec_in=c_out=137`, `cycle=144`;
- checkpoint: validation-selected early stopping as implemented upstream;
- test: one evaluation after training for each frozen horizon run.

The upstream paper states that drop-last is disabled and patience is 3, whereas
the released ECL script/data factory implement `patience=5` and `drop_last=True`.
The Solar extension follows the released ECL script/code to preserve source-code
comparability, and this discrepancy must remain disclosed in the paper audit.
