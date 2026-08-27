# Appendix C prediction export 说明

`scripts/export_iscf_bsca_appendix_c_predictions.py` 用于生成 Appendix C 的
validation-only 定性预测材料。该脚本不训练模型、不重新选择 profile，也不读取
ablation checkpoint；它只读取
`analysis/iscf_bsca_main_v1_hpo_20260731/final_hpo_freeze_20260806/selected_profile_manifest_final.csv`
中冻结的 Main-I/II selected profile。

## 数据与模型来源

脚本逐数据集读取 manifest 的 `training_artifact_dir/checkpoint.pt`，并校验
`checkpoint_sha256_before_test`。同时，它读取对应的 `effective_config.json`，检查
`hpo_trial_id`、`hpo_profile_id`、`readout_mode` 和 validation split 与 manifest 一致。
因此，Appendix C 的曲线来自调优后的统一 ISCF-BSCA 模型，而不是 ablation arm。

## 前向与样本选择

模型以 `pred_len=720` 在 validation split 上进行一次完整前向计算。validation loader
被显式设置为 `shuffle=False`，每个窗口保存其 validation index 与 raw forecast origin。
在固定 channel 0 上，按四个 prefix `{96, 192, 336, 720}` 的平均 scaled MSE 排序，
选择两个误差最低且 raw origin 至少相隔 720 steps 的窗口；若候选不足，脚本会记录
未满足间隔约束的回退选择。输出同时保存 scaled 与 inverse-transformed raw-scale 的
`prediction` 和 `ground_truth`，供 Figure C1 绘制。

## 输出与边界

每个数据集输出 `appendix_c_predictions.npz`、`selection.csv` 和 `metadata.json`，
并在运行根目录保存 `run_metadata.json`。这些材料只用于 Appendix C 的定性展示，
不改变 Section 5 的正式指标，也不构成新的 checkpoint 选择或 test-set 评估。

## Figure C1 绘制

`scripts/plot_iscf_bsca_appendix_c_figure.py`读取上述冻结预测数组，并输出SVG、PDF、PNG、TIFF与source-data CSV。正文负责提供`C. VISUALIZATION`章节层级，因此图片内部标题只保留`Representative validation trajectories`，不重复显示`Appendix C`。dataset名称在首列左侧纵向排布，并进一步向左避让y轴刻度；公共`Value`标签被移除，使两列trajectory panel能够使用更大的水平空间。图片保持183 mm全文宽，同时将高度压缩至约160 mm，使Appendix C引导文字和Figure C1能够在portrait页面中同页呈现。该布局调整不改变样本、曲线、prefix ruler、坐标轴范围或配色。
