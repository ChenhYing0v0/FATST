# Figure QA Notes

| Field | Prefix figure | Sharing figure |
| --- | --- | --- |
| Core conclusion | horizon-specific models can disagree on the same future steps | preferred sharing extent can vary across future regions |
| Dataset | ETTh2 | ETTm2 |
| Split | validation | validation |
| Seed | 2021 | 2021 |
| Example selection | maximum six-pair origin-channel disagreement | lexicographic maximum sample heterogeneity |
| Archetype | asymmetric quantitative composite | asymmetric quantitative composite |
| Panel structure | two aligned panels | two aligned panels |
| Hero evidence | integrated overlapping trajectories + mean-difference inset | region-wise excess-risk landscape |
| Supporting evidence | all-validation NCHPD | gain versus best fixed extent |
| Palette/marks | thin muted solid lines + sparse staggered shapes + subtle white strokes | ordered indigo-to-rose extent family + sequential excess-risk heatmap |
| Source data | `source_data/prefix_etth2/` | `source_data/sharing_ettm2/` |
| SVG/PDF text | editable | editable |
| PNG | 2161×924, 300 dpi | 2161×924, 300 dpi |
| TIFF | 4322×1848, 600 dpi, LZW | 4322×1848, 600 dpi, LZW |
| Final width | exact 183 mm | exact 183 mm |
| Test accessed | false | false |

Nature static source preflight:

```text
PASS=13
WARN=1
FAIL=0
```

The single warning states that no static final width was detected because the
validator does not evaluate `DOUBLE_COLUMN_WIDTH = 183.0 / 25.4`. Tight-bbox
export has been removed；both PDF files are exactly 518.74 pt=183 mm wide。
Both PNG previews were visually inspected at original resolution. SVG XML
parsing passed and retained editable `<text>` elements. `pdftotext` is
unavailable in the local environment, so PDF text extractability was not
independently checked；the same matplotlib editable-text settings and visible
PDF rendering remain in force。

Prefix trajectory encoding：

- all four horizon predictions use solid lines rather than fragile dash-only
  identification；
- color、marker shape和sparse staggered marker position provide redundant identity；
- thin lines与subtle white separation strokes preserve boundaries without
  visually merging the four predictions；
- $H=720$ is drawn as a lower-layer reference，so it cannot erase the shorter
  horizon curves；
- the removed raw-difference subplot is represented by the source-computed mean
  $|\Delta|$ inset，without changing the underlying statistic。

Sharing heatmap encoding：

$$
E_{s,b}
=
\frac{R_{s,b}-\min_{s'}R_{s',b}}
{\min_{s'}R_{s',b}}\times100\%.
$$

Outlined cells are region winners and therefore have $E_{s,b}=0$。This
replaces the older fixed-$s=720$ heatmap，whose entire $s=720$ row was
necessarily zero and white。Panel b still reports gain versus the sample-best
fixed extent $s=720$，so the fixed-reference comparison remains visible。

Claim boundary:

- illustrative validation examples；
- not prevalence evidence；
- not method-effectiveness evidence；
- not an untouched-test claim。
