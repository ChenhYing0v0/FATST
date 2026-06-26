# TimeAlign Official Baseline Adapter

This directory vendors the official TimeAlign source code and adds only a thin
FATST adapter in `train_repo.py`.

## Source Boundary

- `models/`, `layers/`, `data_provider/`, `exp/`, `utils/`, `run.py`, and
  `scripts/` are copied from the official TimeAlign repository.
- `train_repo.py` is the repo adapter for dataset roots, unified/fixed batch
  execution, seed control, and CSV artifact export.
- The only compatibility changes inside official files are:
  - `sktime` import is optional because ETT/Weather loaders do not use it.
  - `DataFrame.drop(['date'], 1)` is changed to
    `DataFrame.drop(columns=['date'])` for modern pandas.

## Checkpoint Protocols

The official `EarlyStopping` implementation saves every epoch and has the
actual early-stop comparison logic commented out. The official `test()` path
also does not reload the saved checkpoint. The author clarified in GitHub issue
#2 that the paper uses the last-epoch checkpoint because validation/test
distribution shift can make validation-best selection undertrain the model.
Therefore this is treated as an author-intended training/checkpoint policy, not
as a source bug. To keep sensitivity analysis explicit, the adapter exposes two
policies:

- `official-last`: keep the effective official behavior and evaluate the last
  epoch model. This is the primary paper-faithful reproduction protocol.
- `best-val`: evaluate the model state with the best validation MSE. This is a
  validation-selector diagnostic, not a correction of the source-faithful
  reproduction.
