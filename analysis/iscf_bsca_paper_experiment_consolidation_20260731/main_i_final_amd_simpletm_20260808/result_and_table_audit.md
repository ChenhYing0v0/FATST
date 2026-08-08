# AMD / SimpleTM Main I 完整结果与表格审计（2026-08-08）

## 1. 结论

Recovery root下的AMD与SimpleTM official-native复现已完整通过artifact gate，可进入Main I的`native_horizon_specific_accuracy_context`。新表已原子移除CMoS与TimeBase，并在相同位置加入AMD与SimpleTM；没有使用旧失败root、partial rows或published fallback。

- formal units：14/14；
- raw metric rows/checkpoints：AMD 28/28，SimpleTM 82/82，共110/110；
- unique checkpoint SHA256：110/110；
- aggregated table cells：56/56；
- dense Main I：14 models × 7 datasets × 4 horizons = 392 rows；含four-H Avg.后为490 rows；
- decision：`AMD_SimpleTM_complete_Main_I_atomic_replacement_pass_scope_closed`。

## 2. Artifact与protocol边界

正式结果只来自`/home/yingch/exp_outputs/r-2026-fatst/amd_simpletm_main_i_7dataset_20260807_recovery`。2026-08-06旧root因SimpleTM upstream `setting`遗漏repeat identity而使checkpoint覆盖，其三个partial units永久excluded。Recovery adapter只在checkpoint目录名末尾追加native `ii`，不改变data、model、RNG推进、objective、optimizer、validation early stopping或formal test计算。

AMD固定official commit `000d377a1ed8946aa817ff357cdf1de64b99abb9`、released `L=512` profiles与seed2024，每cell单次run。SimpleTM固定commit `3c77d820837b726afb03c943235ea95bc924243d`、released `L=96` profiles、fix_seed2025与native `itr`；table cell为对应2或3次repetitions的算术平均。SimpleTM的epoch-level test pass被移除，每个validation-selected checkpoint只在训练结束后执行一次formal test。两者均为source-native fixed-H accuracy context，不是matched mechanism attribution。

## 3. 七数据集four-H mean

| Dataset | AMD MSE | AMD MAE | SimpleTM MSE | SimpleTM MAE |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | 0.412120 | 0.427948 | 0.428115 | 0.432804 |
| ETTh2 | 0.366153 | 0.406445 | 0.356164 | 0.391663 |
| ETTm1 | 0.351348 | 0.378146 | 0.382932 | 0.396456 |
| ETTm2 | 0.253708 | 0.315372 | 0.280194 | 0.325080 |
| Weather | 0.224632 | 0.264975 | 0.242998 | 0.271476 |
| ECL | 0.162126 | 0.256594 | 0.167800 | 0.261907 |
| Solar | 0.204837 | 0.247551 | 0.186493 | 0.247270 |
| Macro | 0.282132 | 0.328147 | 0.292099 | 0.332380 |

## 4. Main I更新后的相对结果

ISCF-BSCA七数据集macro MSE/MAE为`0.262469/0.308281`：相对AMD低`6.970%/6.054%`，MSE/MAE分别在27/28、27/28 cells领先；相对SimpleTM低`10.144%/7.250%`，分别在24/28、27/28 cells领先。

在更新后的完整14-model表中，ISCF-BSCA为29/56 best、19/56 second。这里的56指28个dataset-horizon cells × MSE/MAE；排名按所有displayed values统一四舍五入到三位小数后计算，允许并列。该结果不等于此前five-comparator HPO口径的33/56，二者不得混用。

## 5. 输出与rollback

- machine-readable table：`table_data_long.csv`；
- paper-ready LaTeX：`table_iscf_bsca_main_i_qdf.tex`；
- Exchange companion：`table_exchange_companion_long.csv`与`table_exchange_companion.tex`；
- rendered PDF：`output/pdf/iscf_bsca_main_i_amd_simpletm_20260808.pdf`；
- raw synced evidence：`../amd_simpletm_main_i_reproduction_20260806/remote_lite/`。

若后续发现source hash、checkpoint hash、repeat count、test-access boundary或任一cell不一致，应整体回滚AMD/SimpleTM两列到替换前版本；禁止只删除不利cell或选择性保留某个repeat。
