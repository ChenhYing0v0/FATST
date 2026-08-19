# Efficiency Accuracy--Memory--Storage Result and Table Audit

日期：2026-08-17

Decision：`complete_nine_system_accuracy_advantage_resource_mixed`

## 1. 完整性

- canonical Main I accuracy：9 systems × 7 datasets × 4 horizons=`252/252` cells；
- peak-memory与checkpoint-storage service units：9 systems × 7 datasets=`63/63`；
- 既有独占RTX 3090 profiler复用ISCF-BSCA、TimeAlign、QDF的21 units；AMD/SimpleTM为14/14 actual-checkpoint fresh-process units；DLinear/iTransformer/PatchTST/TimeMixer为28/28 official-architecture-equivalent fresh-process units；
- table-role objects共`231/231`：119个actual trained checkpoints，112个按官方配置实例化并序列化的checkpoint-equivalent state dicts；
- new training=`0`；formal test access=`0`；本轮synthetic input profiling不访问dataset loader或labels。

## 2. Canonical result

| System | Main I MSE | Main I MAE | Peak memory (MiB) | Checkpoints (MiB) |
| --- | ---: | ---: | ---: | ---: |
| **ISCF-BSCA** | **0.261** | **0.306** | 38.817 | 17.677 |
| TimeAlign | 0.274 | 0.314 | 109.102 | 95.433 |
| QDF | 0.288 | 0.331 | 31.939 | 20.381 |
| AMD | 0.282 | 0.328 | 238.035 | 221.150 |
| SimpleTM | 0.293 | 0.333 | <u>14.841</u> | **2.835** |
| DLinear | 0.300 | 0.339 | **13.111** | <u>3.465</u> |
| iTransformer | 0.292 | 0.336 | 47.341 | 35.425 |
| PatchTST | 0.280 | 0.326 | 71.604 | 25.295 |
| TimeMixer | 0.283 | 0.332 | 53.422 | 37.182 |

Accuracy是Main I七datasets × four horizons的macro mean。Peak memory与checkpoint storage先按每dataset构造完整four-horizon service，再对七个datasets作macro mean。上表为文字审计，正式LaTeX中的bold/underline按九个systems逐列计算。

## 3. 比较结果

- ISCF-BSCA在九系统中取得最佳macro MSE/MAE；相对新增DLinear、iTransformer、PatchTST、TimeMixer的MSE改善分别为`13.147%`、`10.660%`、`6.733%`、`7.852%`，MAE改善分别为`9.608%`、`8.916%`、`6.236%`、`7.809%`；
- 相对TimeAlign与AMD，ISCF-BSCA的peak memory分别降低`64.422%`与`83.693%`，checkpoint storage分别降低`81.477%`与`92.007%`；
- 相对QDF，ISCF-BSCA checkpoint storage降低`13.269%`，但peak memory高`21.535%`；
- DLinear与SimpleTM是轻量级反例：DLinear的peak memory最低（`13.111 MiB`），SimpleTM的checkpoint-equivalent storage最低（`2.835 MiB`）；QDF的peak memory也低于ISCF-BSCA；
- ISCF-BSCA相对iTransformer、PatchTST与TimeMixer分别减少peak memory `18.007%`、`45.790%`、`27.340%`，并减少checkpoint storage `50.102%`、`30.117%`、`52.459%`；
- 因此本表支持的是`best macro accuracy + one-checkpoint service consolidation`，以及相对TimeAlign、AMD、iTransformer、PatchTST与TimeMixer的resource advantage；不支持“ISCF-BSCA在memory或storage上优于所有baseline”。

## 4. Measurement boundary

- hardware=`one exclusive NVIDIA GeForce RTX 3090`；precision=`FP32`；batch size=`1`；input=`synthetic standardized`；
- peak memory为fresh process中的`torch.cuda.max_memory_allocated()`，包含完整service的resident model weights、synthetic inputs与一次all-horizon inference activation peak；
- ISCF-BSCA resident一个unified checkpoint，执行一次$H=720$ forward并生成prefix views；
- 八个baseline resident四个native fixed-H models，并顺序执行四次native forwards；
- TimeAlign/QDF/AMD/SimpleTM的checkpoint storage使用实际文件bytes；DLinear/iTransformer/PatchTST/TimeMixer使用官方配置模型的标准`torch.save(state_dict)`序列化bytes，不是trained artifact totals；
- SimpleTM Main I accuracy沿用冻结multi-run summary；资源统计使用预先冻结的repeat 0，一个repeat代表一次非ensemble deployment，不将三个统计重复同时常驻；
- 该结果描述inference service GPU allocation，不等同于training peak memory、CUDA reserved memory或production end-to-end host memory。

## 5. Evidence-role boundary

- DLinear、iTransformer与PatchTST现有FATST资产主要是Main-II H720 checkpoints，TimeMixer也缺少完整7 × 4 inventory；本表未将单个H720文件乘四，而是按官方Main-I配置构造四个architecture-equivalent objects；
- architecture-equivalent rows可比较模型结构造成的resident memory与standard state-dict storage，不可声称是四次完成训练后的实际artifact footprint；
- TVNet只有ETTh2具备author-corrected local reproduction evidence；
- 其他Main I published-context baselines缺少完整7×4 locally audited或official-architecture-equivalent resource evidence，均不进入cost ranking。

## 6. Canonical artifacts

- protocol：`configs/iscf_bsca_efficiency_accuracy_memory_storage_protocol.json`；
- additional unit evidence：`additional_units/*.json`；
- dataset results：`efficiency_dataset_results.csv`；
- macro results：`efficiency_system_macro_results.csv`；
- summary：`efficiency_result_summary.json`；
- LaTeX：`table/table_iscf_bsca_efficiency.tex`；
- standalone LaTeX：`table/table_iscf_bsca_efficiency_standalone.tex`；
- review PDF：`output/pdf/iscf_bsca_efficiency_accuracy_memory_storage_20260817.pdf`。

上一版accuracy--parameters--one-epoch表保留为historical compute-trade-off audit，不删除其负向training-time evidence。
