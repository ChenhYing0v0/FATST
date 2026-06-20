# R_2026_FATST

Clean research repository for the next stage after `R_2026_FSA`.

## Scope

This repository starts from a clean implementation and documentation base for a
high-level SCI journal paper in time series forecasting.

Confirmed research directions:

- one model for multi-horizon forecasting
- future-aware architecture
- MoE-style conditional computation

Confirmed evidence boundary:

- Zotero is the source of truth for literature discovery and metadata.
- The Zotero `FSA` subset is the seed paper set for this project.
- SRSNet is the key comparison baseline.
- Old `R_2026_FSA` code, configs, artifacts, and memory are not imported unless
  the user approves a specific source and purpose.

## Directory Layout

```text
analysis/              Analysis notebooks, reports, and post-processing notes.
artifacts/             Generated outputs; large or reproducible outputs stay untracked.
baselines/             Baseline reproduction notes and local wrappers.
configs/               Experiment and model configs.
data/                  Local data entry point; raw data should not be committed.
docs/                  Project documentation.
docs/code-explanation/ Code-facing explanations for implementation changes.
docs/experiments/      Experiment plans, run logs, and result reports.
docs/remote/           Remote execution and server usage notes.
Papers/                Canonical Chinese paper notes from Zotero sources.
scripts/               Utility scripts.
src/fatst/             Local package for project code.
tests/                 Targeted tests and verification scripts.
```

## First Checks

```bash
python scripts/check_project_structure.py
scripts/remote/check_529lab_3090_gpus.sh
```

The remote GPU check assumes `529_Lab-3090` is a valid SSH alias. Pass a host
explicitly if needed:

```bash
scripts/remote/check_529lab_3090_gpus.sh user@host
```
