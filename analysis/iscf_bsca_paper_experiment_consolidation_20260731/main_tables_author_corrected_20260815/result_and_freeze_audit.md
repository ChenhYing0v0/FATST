# Main I / Main II Author-Corrected Freeze Audit (2026-08-15)

## 1. Correction scope

This freeze supersedes the prior canonical Main I and Main II display artifacts while
preserving those artifacts as historical snapshots. It transcribes the author's
corrected rerun values at the supplied three-decimal precision.

- Main I: ISCF-BSCA and TimeAlign on all seven datasets; SimpleTM on Solar; TVNet on
  **ETTh2**.
- Main II: ISCF-BSCA and TimeAlign on all seven datasets; SimpleTM on Solar; PatchTST
  on ETTh2.
- Every unlisted baseline cell is retained from the preceding frozen table.

Each table therefore replaces 64 standard dataset-horizon rows. Dataset averages are
recomputed from the four displayed horizons; best and second-best styles are recomputed
over the complete table using common three-decimal half-up rounding with ties allowed.

## 2. Result summary

| Table | Matrix | ISCF best metric cells | ISCF second metric cells | ISCF macro MSE / MAE |
| --- | ---: | ---: | ---: | ---: |
| Main I | 14 systems x 7 datasets x 4 H | 44/56 | 9/56 | 0.260714 / 0.306107 |
| Main II | 8 systems x 7 datasets x 4 H | 50/56 | 6/56 | 0.260714 / 0.306107 |

The ranking counts are descriptive consequences of the corrected complete tables; no
dataset, horizon, seed, or metric cell was removed from the reporting surface.

## 3. Provenance and claim boundary

The corrected values were supplied as author screenshots and are available only at
three-decimal precision. The tables therefore store those displayed values directly as
the corrected numeric evidence. They do not infer higher-precision values, checkpoint
identities, or checkpoint hashes. For Main II, superseded hashes are intentionally
blanked on corrected aggregate rows rather than being incorrectly carried forward.

Unchanged Main I rows retain their previous official-local or published-context roles.
Unchanged Main II rows retain the complete 2026-08-13 horizon-loader reaudit role.
Accordingly, Main I remains a system-level horizon-specific accuracy comparison and
Main II remains a source-native one-model benchmark; neither table provides matched
mechanism attribution.

## 4. Verification and freeze decision

- Both CSV matrices are complete and key-unique.
- Avg. rows were regenerated rather than transcribed.
- Main II's former banker's-rounding edge case was replaced with explicit decimal
  half-up rounding, aligning ETTm2 and Weather averages with the shared presentation
  contract.
- Both standalone LaTeX sources compile without errors to one-page A3 landscape PDFs.
- Both PDFs were rendered to PNG and visually checked for clipping, overlap, broken
  rules, unreadable text, or caption overflow; no defect was found.
- The exact canonical paths and SHA256 values are frozen in `freeze_manifest.json`.

Decision=`main_i_main_ii_author_corrected_20260815_complete_hash_frozen`.
