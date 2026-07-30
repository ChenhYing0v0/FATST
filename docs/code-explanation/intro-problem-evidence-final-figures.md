# Introduction Problem-Evidence Final Figures 代码说明

## 1. 作用与边界

`scripts/plot_intro_problem_evidence_final.py`只读取冻结的validation source data，
生成两张Introduction问题证据图。它不训练模型、不访问test、不重新选择sample，
也不修改任何forecast values。

```text
ETTh2 prefix source CSV + summary + pair metrics
  -> Figure 1 SVG/PDF/PNG/TIFF

ETTm2 selected region-risk CSV + summary
  -> Figure 2 SVG/PDF/PNG/TIFF
```

## 2. Figure 1 source tensors and quantities

`selected_forecast_data.csv`每行对应一个relative step：

| Column | Source | Meaning |
| --- | --- | --- |
| `relative_step` | validation window index | history为负，future为1--720 |
| `phase` | artifact role | `history`或`future` |
| `history` | inverse-scaled `history[o,:,c]` | selected observed values |
| `ground_truth` | inverse-scaled H720 target | selected future truth |
| `prediction_hH` | inverse-scaled DLinear prediction | horizon H model output |
| `difference_hH_vs_h720` | prediction H minus prediction H720 | same-step disagreement |

Final Figure 1只展示last-48 history与first-96 future，因为96是四个requested
horizons的共同overlap。panel b重新从source predictions计算raw-value
difference与mean absolute difference，不使用图像像素或手工抄写。

`pair_metrics.csv`的`nchpd_l1`来自所有validation origins、future overlap steps与
channels的standardized mean absolute difference。panel c将六个pair values映射到
4×4 upper-triangular matrix；blank lower triangle表示未重复展示对称comparison。

## 3. Figure 2 source tensors and quantities

`selected_region_risk.csv`有60行：

```text
5 sharing extents × 12 future regions
```

每行`mse`为selected origin上对应scale、60-step region及all channels的mean
squared error。script构造：

$$
\Delta R_{s,b}
=
\frac{R_{s,b}-R_{s^\mathrm{fixed},b}}
{R_{s^\mathrm{fixed},b}},
$$

其中$s^\mathrm{fixed}=720$是该sample全域MSE最低的single fixed scale。panel a
绘制$100\Delta R_{s,b}$，outlined square标记
$\arg\min_sR_{s,b}$。

panel b的bar height为：

$$
G_b
=
\frac{
R_{s^\mathrm{fixed},b}-\min_sR_{s,b}
}{
R_{s^\mathrm{fixed},b}
}\times100\%.
$$

bar与baseline marker颜色来自region winner。winner恰为fixed s720时$G_b=0$，
仍用绿色square显示winner identity，避免零高度bar使region 10--12不可见。

## 4. Export and QA

脚本固定：

- Python/matplotlib backend；
- sans-serif publication fonts；
- editable SVG text与PDF Type 42 fonts；
- 183 mm double-column design width；
- PNG 300 dpi；
- LZW TIFF 600 dpi；
- source-data/selection/claim boundary写入`figure_manifest.json`。

Nature static source audit为13 PASS、1 WARN、0 FAIL。WARN只因validator不能从变量
表达式静态解析183 mm width；rendered PDF约185.8 mm，排版时缩放至183 mm。

## 5. Code-theory consistency

Intended claim：

1. independently optimized horizons可在相同future steps上分歧；
2. matched fixed sharing extents的region-wise risk ordering可发生明显变化。

Code realization：

- Figure 1把selected example与all-validation heatmap分离；
- Figure 2只展示matched neutral decoder risk，不出现ISCF/BSCA；
- maximum selection与same-validation descriptive oracle均在figure footer、
  manifest和caption中公开。

仍然只是proxy：

- selected examples不估计prevalence；
- region oracle不是learned out-of-sample policy；
- DLinear evidence不能代表所有varied-horizon models。

若source alignment、test boundary、CSV completeness、SVG/PDF/TIFF export或
winner reconstruction任一失败，figure package不得进入paper draft。
