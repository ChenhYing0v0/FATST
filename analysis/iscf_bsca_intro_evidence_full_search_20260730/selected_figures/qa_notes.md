# Figure QA Notes

| Field | Prefix figure | Sharing figure |
| --- | --- | --- |
| Core conclusion | horizon-specific models can disagree on the same future steps | preferred sharing extent can vary across future regions |
| Dataset | ETTh2 | ETTm2 |
| Split | validation | validation |
| Seed | 2021 | 2021 |
| Example selection | maximum six-pair origin-channel disagreement | lexicographic maximum sample heterogeneity |
| Source data | `source_data/prefix_etth2/` | `source_data/sharing_ettm2/` |
| SVG/PDF text | editable | editable |
| PNG | 300 dpi | 300 dpi |
| TIFF | 600 dpi, LZW | 600 dpi, LZW |
| Final width | designed at 183 mm; bbox output about 185.8 mm | designed at 183 mm; bbox output about 185.8 mm |
| Test accessed | false | false |

Nature static source preflight:

```text
PASS=13
WARN=1
FAIL=0
```

The single warning states that no static final width was detected. The script
uses `DOUBLE_COLUMN_WIDTH = 183.0 / 25.4`; manuscript insertion should scale
the bbox output to exactly 183 mm. Both PNG previews were visually inspected at
original resolution. SVG XML parsing passed and retained editable `<text>`
elements. `pdftotext` is unavailable in the local environment, so PDF text
extractability was not independently checked; the same matplotlib editable-text
settings and visible PDF rendering remain in force.

Claim boundary:

- illustrative validation examples；
- not prevalence evidence；
- not method-effectiveness evidence；
- not an untouched-test claim。
