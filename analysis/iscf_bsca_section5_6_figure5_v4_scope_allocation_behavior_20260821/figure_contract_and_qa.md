# Section 5.6 / Figure 5 v4：scope diversity and allocation behavior

## 1. 写作结论

Section 5.6 只回答 Full ISCF-BSCA 的内部行为，不重复 Section 5.5 的 component utility：

1. Scope-conditioned Forecasts 是否保留不同的预测信号，而不是退化为相同轨迹；
2. Scope Probabilities 是否随 future region 改变，而不是对所有 region 固定使用同一 scope；
3. CHPC 仅作为由 inference graph 保证且经数值复核的 architecture property 一句带过，不设置独立可视化。

## 2. Figure contract

- **Core conclusion**：jointly constructed scope field 保留 scope-wise forecast diversity；不同 future regions 呈现不同的最低误差 scope；learned soft allocation 未塌缩为全 future domain 的固定 scope。
- **Archetype**：`asymmetric mixed-modality quantitative figure`。
- **Backend**：Python / matplotlib only。
- **Final size**：183 mm × 140 mm。
- **Panel a**：从冻结的1,280个sequential validation probes中，先最大化8个regions内不同lowest-MSE scopes的数量，再最大化五条scope forecast的mean pairwise absolute disagreement。选中ETTh1 probe row 107；该example覆盖全部五种regional winners，mean pairwise disagreement为0.336375。
- **Panel b**：对每个dataset使用全部validation origin-variable rows，展示5 datasets × 8 regions的lowest-MSE scope；marker area编码同一cell内best-to-worst excess MSE gap。
- **Panel c**：使用相同5 × 8 grid，展示mean soft allocation中highest-weight scope。该panel只描述allocation profile的区域变化，不把highest-weight scope解释为hard route或oracle选择。

## 3. Evidence hierarchy

| Evidence | Split / aggregation | 支持 | 不支持 |
| --- | --- | --- | --- |
| Panel a selected trajectory | validation；完整1,280-row pool的披露选择规则 | 至少一个审计案例中五条scope forecasts明显不同，且regional lowest-error scope变化 | 典型性、出现频率或population prevalence |
| Panel b aggregate competence | validation；每dataset完整rows后按dataset-region聚合 | 五种scope均至少在一个dataset-region cell取得lowest MSE；不存在全域统一winner | learned allocation已识别best scope |
| Panel c aggregate allocation | validation；每dataset完整rows后按dataset-region聚合 | 五种scope均至少在一个cell获得highest mean probability；4/5 datasets的highest-weight scope随region改变 | hard routing、oracle recovery或causal specialization |
| CHPC audit | validation；5 datasets × 4 horizons | 20/20 comparisons的maximum absolute CHPD为0 | accuracy gain |

原始diagnostic audit中的near-uniform probability与allocation--competence alignment结果继续保存在canonical result/governance记录中。本轮按author要求不将alignment count放入正文或Figure 5，也不据此声称successful routing；正文通过“soft reweighting rather than hard routing”和“not oracle scope recovery”保留必要claim boundary。

## 4. Source-data integrity

- source artifacts：5个冻结Full ISCF-BSCA validation diagnostic objects；
- datasets：ETTm1、ETTm2、ETTh1、ETTh2、Weather；
- scopes：$\{1,48,144,360,720\}$；
- future regions：8；
- aggregate rows：每个dataset使用原artifact中的全部validation origin-variable rows；
- selected trajectory pool：5 datasets × 256 sequential probes = 1,280 rows；
- missing-value filtering：0；
- smoothing / interpolation / manual replacement：0；
- new training / formal test / checkpoint mutation：0 / 0 / 0。

Generated source data：

- `source_data/scope_trajectory_selected.csv`；
- `source_data/selected_region_preferences.csv`；
- `source_data/scope_competence.csv`；
- `source_data/scope_allocation.csv`；
- `source_data/trajectory_selection_audit.csv`；
- `figure_summary.json`。

## 5. Visual and export QA

- Nature Figure static preflight：14 PASS、0 WARN、0 FAIL；
- editable vector：SVG text保留，PDF使用TrueType embedding；
- raster：PNG 300 dpi，TIFF 600 dpi并使用LZW compression；
- palette：五种scope在全部panels中使用同一组low-saturation blue--violet colors；数字编码作为颜色之外的冗余通道；
- layout：Panel a作为hero evidence；Panels b/c共享5 × 8 grid以分别呈现competence与allocation，但caption明确二者不构成oracle matching test；
- final-size visual inspection：无clipping、overlap或不可读文字；最小显式font size为5 pt。

## 6. Decision

Decision=`section5_6_v4_scope_allocation_behavior_temporarily_fixed_usable`。

Section 5.6与Figure 5 v4可暂时作为paper-usable版本。旧Figure 5与v2/v3设计保留为历史审计材料，不再作为当前正文证据载体。本轮不修改Sections 1--4，不新增implementation、remote training或formal test。
