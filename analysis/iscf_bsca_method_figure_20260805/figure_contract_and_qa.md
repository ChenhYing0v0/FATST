# ISCF-BSCA Method Figure Contract and QA

## Status

| Field | Content |
| --- | --- |
| `figure_id` | `figure_iscf_bsca_method_overview` |
| `figure_number` | Figure 4 |
| `status` | `initial_draft_for_author_review` |
| `manuscript_role` | Explain the frozen ISCF-BSCA architecture and its training/inference boundary |
| `backend` | Python/matplotlib only |
| `archetype` | asymmetric schematic-led composite |
| `data_role` | architecture schematic; no empirical observations |
| `empirical_data_used` | false |
| `test_accessed` | false |

## Figure contract

**Core conclusion.** ISCF constructs one scope-indexed forecast field from independent scope-specific history projections, performs target-conditioned contraction along the scope axis to obtain one prefix-consistent trajectory, and uses BSCA only during training to maintain learning access to all scope slices.

**Panel map.**

- panel a, single-scope forecasting: one fixed latent-state sharing extent is applied across the complete future domain;
- panel b, ISCF field construction: a shared history representation is mapped by independent scope-specific projections into multiple scope-region states, while shared step-specific synthesis produces one `scope_field:[B,C,T,S]`;
- panel c, target-conditioned allocation: `allocation:[B,C,T,S]` contracts the scope axis of the field into `forecast:[B,T,C]`, from which horizon requests obtain nested prefixes;
- panel d, BSCA: direct scope-skill supervision and a uniform allocation anchor are shown as dashed train-only paths, while the solid inference graph remains unchanged.

**Evidence hierarchy.** Panel b is the hero panel because it carries the main architectural novelty. Panels c and d explain integration and optimization. Panel a is a quiet contrast that clarifies why one fixed sharing extent is insufficient.

**Reviewer risks and controls.**

- Multiple rows could be misread as independent forecasting models. The figure labels them as scope-conditioned slices within one field and shows their shared encoder and synthesis path.
- Sharing scope could be misread as forecast horizon. The scope axis and requested-prefix markers use distinct visual encodings, and the caption states that scope size is not a requested horizon.
- Target-conditioned allocation could be misread as using future labels. Panel c conditions on history state and future-step coordinate only.
- BSCA could be misread as an inference module. Every BSCA arrow is dashed and enclosed in a `training only` region; the caption states that it is removed at inference.
- Allocation colors could be misread as measured specialization. The figure uses schematic weights without dataset, metric or winner annotations.

## Export contract

- exact width: 183 mm;
- target height: 112 mm;
- SVG/PDF: editable text;
- PNG: 300 dpi;
- TIFF: 600 dpi with LZW compression;
- stable manuscript copies: `paper-figures/figure_iscf_bsca_method_overview.*`;
- source data file: not applicable because the figure is an architecture schematic.

## QA checklist

| Check | Initial result |
| --- | --- |
| Tensor direction and shapes match the frozen implementation | pass |
| Scope rows are not presented as independent models | pass; shared encoder, synthesis and field are explicit |
| Scope and requested horizon are visually distinct | pass; scope labels appear only in panel b and horizon markers only in panel c |
| BSCA is visibly training-only | pass; dashed coral training layer is separated from the solid inference graph |
| Python static preflight | 13 PASS / 1 WARN / 0 FAIL |
| Remaining warning | static parser cannot resolve the width constant; PDF media box verifies 183.000 mm |
| SVG editable text | pass; 75 text nodes |
| PDF media box | 518.74 × 317.48 pt, corresponding to 183 × 112 mm |
| PNG/TIFF dimensions and resolution | PNG 2161 × 1322 at 300 dpi; TIFF 4322 × 2645 at 600 dpi with LZW |
| Final-size PNG and PDF visual inspection | pass for initial draft; final visual hierarchy remains pending author review |

The figure remains a draft until the author confirms the information hierarchy and visual emphasis. Its creation does not modify the frozen model, launch experiments or establish effectiveness claims.
