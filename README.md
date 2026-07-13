# R_2026_FATST

StageC research repository for unified varied-horizon time series forecasting.

## Active Research

- carrier: frozen `A6-LBF-natural-baseline` on Weather, ETTm1, ETTh2;
- decoder candidate: Projective Multiresolution Forecast Operator (PMFO);
- training candidate: Projective Increment Risk (PIR);
- current step: Step 2-3 problem-existence diagnostics; method training is not authorized yet.

Start from:

1. `docs/paper-mainline.md`
2. `docs/stage-ledgers/stage-c-unified-forecasting-redesign.md`
3. `docs/research-roadmap.md`
4. `docs/experiments/stage-c-pmfo-pir-problem-diagnostic.md`

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
