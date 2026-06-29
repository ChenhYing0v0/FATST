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

## D0 Head / Interface Diagnostic

The adapter exposes `--pred-loss-mode` for Phase5 TimeAlign-HSS D0:

- `full`: official prediction loss on the full `pred_len` output. This is the
  default and preserves previous official-source runs.
- `multi-prefix`: average prediction loss over the requested target prefixes
  such as `96,192,336,720`, while keeping the official TimeAlign forward,
  reconstruction loss, and alignment loss unchanged.
- `balanced-step`: average prediction loss over non-overlapping regions split
  by the requested prefixes, such as `1:96`, `97:192`, `193:336`, `337:720`.
- `stochastic-prefix`: sample prefix lengths from the requested target prefixes
  during training.
- `continuous-prefix`: sample prefix lengths from a denser prefix pool such as
  `32,64,...,720` during training.

This is a diagnostic for the unified-head/interface confounder. Official
TimeAlign uses a fixed `Linear(d_model * patch_num, pred_len)` output projection;
the unified-720 setting evaluates shorter horizons by cropping prefixes. The
diagnostic tests whether the observed unified decrease is mainly caused by short
prefixes receiving insufficient direct prediction supervision before introducing
HSS reliability scheduling. `balanced-step` is a mechanism control, while
`stochastic-prefix` and `continuous-prefix` are candidate scheduling protocols.
