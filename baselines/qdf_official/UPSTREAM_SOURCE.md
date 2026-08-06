# Upstream source

- Repository: `https://github.com/Master-PLC/QDF`
- Upstream commit: `eb0693a962928e229417fd80b401c37b0dac6a67`
- Pulled: `2026-08-06`
- License: MIT (`LICENSE` retained verbatim)
- Vendoring rule: source files are tracked under `baselines/qdf_official/`;
  the upstream `dataset/` directory and `.git/` metadata are excluded.

## FATST-local changes

The forecasting architecture, QDF loss, meta-train/meta-test algorithm, dataset
split, scaler, and metric formulas are unchanged. Local changes are limited to:

1. `scripts/Solar.sh`, adapted from the upstream `scripts/ECL.sh` settings;
2. optional `cupy` import because the selected remote environment does not use
   cuML/CuPy;
3. explicit `weights_only` arguments for PyTorch >=2.6 checkpoint loading;
4. `np.Inf` to `np.inf` for NumPy >=2.0;
5. bounded train/eval batch caps and a `final_evaluation_split=none` option used
   only by resource smoke. Their default values preserve the upstream run path.

Solar is a source-informed extension, not an author-provided official QDF
configuration. The model uses the upstream `Dataset_Solar` loader and transfers
the ECL per-horizon QDF settings, with `enc_in=137` and the 10-minute daily
cycle changed from ECL's 168 to Solar's 144.
