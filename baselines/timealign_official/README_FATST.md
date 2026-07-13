# TimeAlign Official Baseline Adapter

This directory vendors the official TimeAlign source and preserves its baseline
path. FATST also maintains explicitly named local A6 encoder/readout paths in
`models/TimeAlign.py` and the training adapter in `train_repo.py`.

## Source Boundary

- `layers/`, `data_provider/`, `exp/`, `utils/`, `run.py`, and `scripts/` are
  copied from the official TimeAlign repository, apart from the compatibility
  changes listed below.
- `models/TimeAlign.py` preserves the official baseline path and contains the
  explicitly named local A6 encoder/readout candidates.
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

## A6-LBF Clean Carrier

For `--readout-mode learned-basis-forecast-operator`, the active FATST carrier is no longer the inherited
TimeAlign future-reconstruction architecture. The A6-LBF path:

- keeps the official history patch encoder;
- replaces the prediction head with learned basis coefficients and a prefix-native temporal basis;
- removes the future reconstruction/alignment branch from the model instance;
- forces `w_recon=0.0` and `w_align=0.0` in `train_repo.py`.

The official `--readout-mode official` path still keeps reconstruction and alignment for baseline reproduction.

## StageC PMFO-RCT Step 7

The active StageC decoder gate adds four prefix-native readouts after the same frozen A6 history memory:

- `pmfo-rct`: shared parent-to-child states plus fixed conservative synthesis;
- `pmfo-rct-no-transition`: direct history-to-scale coefficient control;
- `pmfo-rct-no-conservation`: the same state tree with unconstrained child updates;
- `dense-mlp-matched`: nonlinear dense control matched to the PMFO decoder parameter budget.

All four require unified `pred_len=720`, disable the future reconstruction/alignment branch, and keep requested
horizon out of learned module inputs. Their active protocol is
`docs/experiments/stage-c-pmfo-rct-step7-protocol.md`; the older StageB sections below are retained for code
traceability and do not authorize those archived methods.

## B14 Prerequisite Contextual Patch Encoder

`--encoder-mode contextual-patch-transformer` replaces the inherited dataset-specific patch/token MLP with a
PatchTST-derived history encoder:

- channel-independent overlapping patches with end replication padding;
- learnable positional embeddings;
- residual cross-patch self-attention;
- post-BatchNorm residual blocks;
- a public `[B,C,P,D]` history-memory interface through `Model.encode_history()`.

This encoder is currently restricted to `--readout-mode learned-basis-forecast-operator`, unified `pred_len=720`.
It does not instantiate or restore the TimeAlign future reconstruction/alignment branch. The full contextual
replacement failed its 3-dataset effectiveness gate and is retained only for traceability.

The Step 5/6 repair is `--encoder-mode hierarchical-patch-memory`: it preserves the accepted A6 forecast path
exactly and exposes parameter-free normalized `P48-S24` local memory through
`Model.encode_retrieval_memory()`. This mode has identical state-dict keys and parameter count to legacy A6 and is
the intended B14 prerequisite interface.

## B9-FSN-SCF Stage-Native Coefficient Field

StageB adds two prefix-native learned-basis readout modes:

- `stage-native-coefficient-field`: B9-FSN-SCF candidate;
- `stage-native-coefficient-field-no-stage`: no-stage capacity control.

Both modes keep the A6 learned temporal basis, but replace the single coefficient vector
`coeff: [B, C, K]` with a stage-indexed coefficient field `coeff_field: [B, C, S, K]`
before basis projection. They do not restore the TimeAlign future reconstruction/alignment branch and still force
`w_recon=0.0` and `w_align=0.0`.

The final stage modulation layer is zero-initialized, so the initial forward path is function-preserving with respect
to `learned-basis-forecast-operator`. The no-stage control keeps the same module shape but shares the averaged stage
token/gate across all stages.

## B11-BCF Continuous Basis-Conditioned Field

StageB B11 adds four prefix-native learned-basis readout modes:

- `basis-conditioned-coefficient-field`: B11-BCF candidate;
- `basis-conditioned-coefficient-field-no-basis`: learned slot control without basis descriptors;
- `basis-conditioned-coefficient-field-shuffled-basis`: reversed basis-order control;
- `basis-conditioned-coefficient-field-constant-slot`: constant row-mixture control.

These modes keep the A6 temporal basis projection but replace the single coefficient vector with a soft coefficient
field. The field is generated from overlapping basis-window descriptors, then mixed for each future row before the
same `learned_temporal_basis` projection. They do not restore TimeAlign future reconstruction/alignment and still force
`w_recon=0.0` and `w_align=0.0`.

The field delta layer is zero-initialized. At initialization all slots equal the A6 `coeff_base`, so the output is
function-preserving up to numerical tolerance.

## B12-STBO Subspace-Tiled Basis Operator

StageB B12 adds four prefix-native STBO readout modes:

- `subspace-tiled-basis-operator-shared`: shared learned local basis across future tiles;
- `subspace-tiled-basis-operator-bank`: learned local basis bank with tile-wise soft mixture;
- `subspace-tiled-basis-operator-dct`: fixed local DCT basis control;
- `subspace-tiled-basis-operator-independent`: independent learned local basis per tile.

These modes replace the A6 full-720 temporal basis projection with a tiled primary operator:

```text
hidden [B,C,R] -> tile_coeff [B,C,M,Rb]
local_basis [M,L,Rb] -> prediction [B,H,C]
```

They do not instantiate the TimeAlign future reconstruction/alignment branch. They also do not use
`learned_temporal_basis` or `learned_basis_coeff`; this is an operator replacement, not `A6 + residual` and not
`coeff + delta`.

STBO-specific arguments:

- `--stbo-tile-len`, default `48`;
- `--stbo-rank`, default `16`;
- `--stbo-bank-count`, default `4`;
- `--stbo-basis-init-std`, default `stbo_rank ** -0.5` when set to `0`.
