# Stage C Main I TimeAlign Table-6-Style Builder 代码说明

## 1. 功能边界

`scripts/build_iscf_bsca_timealign_table6_style.py` 只负责审计、合并和渲染已有
结果，不训练模型、不访问远程机器，也不生成新的 official-test observations。
输入是 TimeAlign source PDF、当前 ISCF-BSCA selected scorecard、TimeAlign
official-native reproduction scorecard，以及已经人工核验的 published subset。

## 2. 数据流

1. `extract_table6_rows()` 使用 `pdfplumber` 读取 source PDF 第 22 页，将表头模型
   的横坐标与每个 dataset/horizon row 的纵坐标对齐，得到 12 个 published models
   的 MSE/MAE。
2. `validate_against_audited_selected()` 将抽取结果中的 TimeAlign、TimeMixer、
   DLinear、iTransformer、PatchTST 共 140 rows 与已有审计 CSV 精确比较。任一 key
   缺失或 metric 不同都会中止。
3. `load_iscf_rows()` 读取 terminal H4N freeze 的 standard-horizon rows，并固定其
   `system_role=unified_single_model_per_dataset`。
4. `override_reproduced_timealign()` 要求并替换 7 个 shared datasets × four
   horizons 的 TimeAlign MSE/MAE，共 28 rows；TimeAlign published rows只保留为
   source cross-check，不再进入dense table数值。
5. `validate_matrix()` 要求标准矩阵恰为
   $13\text{ models}\times7\text{ datasets}\times4\text{ horizons}=364$ 个唯一 keys。
6. `add_averages_and_styles()` 对每个 model-dataset block 重新计算 four-H arithmetic
   mean；随后按三位小数 displayed values 求不同数值中的 best 与 second-best。
7. `load_exchange_companion()`另外读取ISCF-BSCA与TimeAlign的Exchange four-H
   rows，生成明确标注source-informed bootstrap的companion CSV/LaTeX block。
8. `write_long_csv()`、`build_latex()` 和 `build_pdf()` 从同一个 merged row list
   分别生成 machine-readable CSV、LaTeX 与单页 PDF，避免三种载体发生数值漂移。

## 3. `table_data_long.csv` 字段定义

| Column | 来源/计算 | 含义 |
| --- | --- | --- |
| `model` | source header 或 ISCF fixed label | 系统名称 |
| `dataset` | frozen seven-dataset order | 数据集名称 |
| `horizon` | source row；或计算行 `Avg.` | `96/192/336/720` 或 four-H average |
| `mse` | source/full-precision reproduced value；Avg 行为算术平均 | 用于计算的 MSE |
| `mae` | source/full-precision reproduced value；Avg 行为算术平均 | 用于计算的 MAE |
| `mse_display` | `mse` 四舍五入到三位小数 | 表中显示的 MSE |
| `mae_display` | `mae` 四舍五入到三位小数 | 表中显示的 MAE |
| `mse_style` | 当前 dataset-horizon 的 displayed-value rank | `best`、`second` 或 `normal` |
| `mae_style` | 当前 dataset-horizon 的 displayed-value rank | `best`、`second` 或 `normal` |
| `value_origin` | 合并阶段写入 | published、local reproduction、ISCF selected 或 computed average |
| `system_role` | 合并阶段写入 | unified method、official-native fixed-H baseline 或 published context |

三位小数排名是必要的 precision control：published rows 原本只公开到三位小数，
若让本地 full-precision results 在未公开的小数位上打破并列，会制造虚假的比较精度。

## 4. 渲染约定

- 每个 dataset 跨五行显示，并用 horizontal rule 分隔；
- ISCF-BSCA 与 TimeAlign 两列分别使用浅橙和浅灰背景，便于读者定位；
- `best` 渲染为红色粗体，`second` 渲染为蓝色下划线；
- TimeAlign 表头与脚注说明全部7个shared datasets均为本地seed2021复现；
- PDF 使用单页宽幅画布，LaTeX 使用 `table* + resizebox`，二者共享同一排序和数值。

## 5. 可证伪条件

以下任一情况都应使 builder 失败或阻止表格进入论文：source PDF hash 变化、已审计
140-row subset 不再精确一致、TimeAlign reproduction 不覆盖全部32个Main I cells、标准矩阵不是
364 个唯一 keys、任一 model-dataset block 缺少 four-H rows。视觉 QA 还需确认 PDF
无截断、重叠或不可读脚注；代码验证不能替代这一项。
