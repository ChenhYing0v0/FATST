# ISCF-BSCA Method Figure Code Explanation

## Purpose

`scripts/plot_iscf_bsca_method_overview.py` renders Figure 4 as a deterministic architecture schematic. It does not read datasets, checkpoints, predictions or evaluation labels, and it does not execute the model.

## Functional modules

### Shared drawing primitives

`rounded_box`, `arrow`, `panel_frame`, `draw_waveform`, `draw_scope_blocks` and `draw_matrix` provide the consistent visual grammar. All coordinates are expressed in axis-relative units, so the four panels retain their alignment at the fixed 183 × 112 mm canvas size.

### Panel a: single-scope comparison

`plot_single_scope` draws one history state, one history projection and one repeated block pattern. Its role is to show that a single-scope decoder applies the same latent-state sharing extent across all future steps.

### Panel b: scope-field construction

`plot_scope_field` draws a shared history state followed by five independent scope-specific projections. The block groups represent sharing extents $\{1,48,144,360,720\}$, and the shared synthesis column maps every row into one `scope_field:[B,C,T,S]`. The row colors identify scope-conditioned slices, not independent forecasting models.

### Panel c: allocation and prefix output

`plot_allocation` places a schematic field beside a schematic target-conditioned allocation. Their elementwise weighting and scope-axis contraction produce one trajectory with three nested request endpoints. The matrices are deterministic visual encodings and are not learned weights or empirical measurements.

### Panel d: training/inference boundary

`plot_bsca` keeps the field-allocation-contraction path solid and places the three objective terms in a separate dashed region. This matches the frozen contract that BSCA changes the training objective but adds no inference parameter or operation.

### Export and manifest

`save_bundle` writes SVG, PDF, 300-dpi PNG and 600-dpi LZW TIFF. `sync_manuscript_assets` copies the same files to `paper-figures/`. `main` also writes a machine-readable manifest that records the panel roles, non-empirical data status and claim boundary.

## Code-theory consistency

- Intended theory: multiple latent-state sharing scopes form one scope-indexed output field, target-conditioned allocation contracts the scope axis, and BSCA acts only during training.
- Code realization: panel b shares the history state and synthesis path across all scope rows; panel c performs one visual contraction; panel d separates dashed train-only paths from the solid inference path.
- Proxy boundary: block widths, field intensities, allocation intensities and the output trajectory are schematic. They explain tensor semantics but do not reproduce a model forward pass or report learned behavior.
- Falsification: the figure is invalid if it implies one model per scope, equates scope with requested horizon, conditions allocation on future labels or places BSCA in the inference path.
