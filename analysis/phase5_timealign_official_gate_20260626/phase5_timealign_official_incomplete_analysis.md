# Phase5 TimeAlign Official Gate Incomplete Analysis

## Status

[Decision] `incomplete_due_to_quota`.

The official-source run did not complete the planned matrix. Valid metrics exist
only for Weather fixed-horizon runs under `checkpoint_policy=official-last`.
The run stopped after writing Weather fixed metrics because saving
`predictions_test.npz` triggered user quota overflow on the remote server.

## Valid Result Table

| Dataset | Setting | Horizon | MSE | MAE |
| --- | --- | ---: | ---: | ---: |
| Weather | fixed official-last | 96 | 0.140427 | 0.178288 |
| Weather | fixed official-last | 192 | 0.182234 | 0.220079 |
| Weather | fixed official-last | 336 | 0.232655 | 0.262207 |
| Weather | fixed official-last | 720 | 0.306776 | 0.316634 |

## Comparison Against Local TimeAlign Carrier

| Dataset | Horizon | Local MSE | Official-source MSE | Relative MSE | Local MAE | Official-source MAE | Relative MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Weather | 96 | 0.149646 | 0.140427 | -6.16% | 0.190094 | 0.178288 | -6.21% |
| Weather | 192 | 0.198690 | 0.182234 | -8.28% | 0.238119 | 0.220079 | -7.58% |
| Weather | 336 | 0.244323 | 0.232655 | -4.78% | 0.272599 | 0.262207 | -3.81% |
| Weather | 720 | 0.316670 | 0.306776 | -3.12% | 0.326938 | 0.316634 | -3.15% |

## Checkpoint Diagnostics

| Dataset | Horizon | Epochs | Best Epoch | Best Val MSE | Last Val MSE | Last-vs-Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Weather | 96 | 10 | 2 | 0.378200 | 0.381625 | +0.91% |
| Weather | 192 | 10 | 9 | 0.438271 | 0.438938 | +0.15% |
| Weather | 336 | 10 | 3 | 0.500492 | 0.503928 | +0.69% |
| Weather | 720 | 10 | 3 | 0.592020 | 0.596658 | +0.78% |

## Reading

[Fact] Switching from the repo-local TimeAlign-style implementation to the
official-source model/dataloader/preset improves Weather fixed-horizon MSE by
`3.12%` to `8.28%`.

[Strong Evidence] The earlier Weather fixed-horizon gap was at least partly
caused by implementation/preset mismatch. The official per-horizon settings,
especially Weather h720 `d_ff=128/dropout=0.5`, matter.

[Fact] The last-epoch protocol is not a large source of Weather fixed-horizon
damage in this run: last validation MSE is only `0.15%` to `0.91%` worse than
best validation MSE.

[Decision] This incomplete run cannot support any unified-vs-fixed or HSS
necessity claim. The next run should disable prediction NPZ saving, clean remote
quota-heavy temporary outputs, and rerun the full official-last matrix.
