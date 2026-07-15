# R_2026_FATST

StageC research repository for unified varied-horizon time series forecasting.

## Active Research

- carrier: frozen `A6-LBF-natural-baseline` with five dataset-aware natural profiles;
- decoder candidate: provisional Compression-Aware Dual-Memory Operator (CADMO);
- training candidate: provisional Conditional Predictive-Gain Accounting (CPGA);
- current step: D14 Step 2-3 conditional patch-memory headroom audit;
- method training, remote training and test access are not authorized yet.

Start from:

1. `docs/paper-mainline.md`
2. `docs/stage-ledgers/stage-c-unified-forecasting-redesign.md`
3. `docs/research-roadmap.md`
4. `docs/experiments/stage-c-d14-conditional-patch-memory-headroom.md`

The forecast-revision surface idea is intentionally separated as a future-paper concept in `New-idea.md`.

Historical scripts, local candidates, configs and documents are retained under explicit `archive/` directories.
Detailed experiment evidence remains under `analysis/`.

## Repository Layout

```text
analysis/              Detailed experiment evidence and reports.
baselines/             Active carriers and external controls; old candidates are archived.
configs/               Frozen active contract and its provenance.
docs/                  Paper mainline, roadmap, active ledger and protocols.
scripts/               Active runners/analyzers only; old scripts are archived.
src/fatst/             Local package skeleton.
tests/                 Targeted verification.
```

## First Checks

```bash
python scripts/check_project_structure.py
scripts/remote/check_529lab_3090_gpus.sh
```
