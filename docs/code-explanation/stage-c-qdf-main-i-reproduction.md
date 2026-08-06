# Stage C：QDF Main I 与 Solar 复现代码说明

## 1. Source boundary

`baselines/qdf_official/` 固定 QDF 官方仓库 commit `eb0693a...`。`UPSTREAM_SOURCE.md` 记录 upstream hashes、MIT license、未纳入的数据目录及本地兼容性 patch。QDF 仍在自己的 native runner 中训练，没有改写成 ISCF-BSCA carrier。

## 2. 数据流与 tensor contract

Solar 输入由 `Dataset_Solar` 构造：

- `batch_x: [B, 96, 137]`；
- `batch_y: [B, 48 + H, 137]`；
- TQNet 产生 `outputs: [B, H, 137]`；
- QDF 学习的 loss module `A` 在 meta-training 中根据 multi-step error 训练，随后重新初始化 TQNet，并以该 loss 训练 forecasting model；
- test 阶段比较 `outputs` 与 `batch_y[:, -H:, :]`，保存顺序为 `[MAE, MSE, covariance_loss, RMSE, MAPE, MSPE, MRE]`。

## 3. 本地必要 patch

- `run.py`：将非本路径所需的 `cupy` 变成 optional dependency；增加 bounded smoke 与 `final_evaluation_split`，确保 resource smoke 不读取 test。
- `exp_long_term_forecasting_meta_ml3.py`：支持 train/eval batch cap；对 state dict 显式使用 `weights_only=True`，对官方序列化的 learned loss module 使用 `weights_only=False`，适配 PyTorch 2.6+。
- `utils/tools.py`：把 NumPy 2.x 已删除的 `np.Inf` 改为 `np.inf`。
- `scripts/Solar.sh`：从 ECL 的逐-H profile 派生 Solar 四任务，并按 H720 / H336+H96 / H192 做 workload-aware 三 GPU 调度。
- Solar runner 固定 `num_workers=0`，并在完整 checkpoint 已存在但 test 因基础设施失败时执行 evaluation-only retry；原训练日志改名保留，checkpoint 不重训。

这些 patch 不改变 QDF/TQNet forward、loss 公式或正式训练预算；batch cap 仅在 resource smoke 启用。

## 4. Audit path

- `configs/qdf_solar_reproduction.json` 冻结 source/data hash、profiles、授权、test role 与 gates。
- `scripts/check_qdf_solar_reproduction.py` 在 launch 前验证 executed-source hashes、四行 dry-run 与 24 个 published QDF rows。
- `scripts/remote/run_qdf_solar_reproduction.sh` 负责 remote environment provenance、resource-smoke gate、formal launch 与 artifact status。
- `scripts/analyze_qdf_solar_reproduction.py` 要求每个 horizon 恰好一个 checkpoint、一个 learned loss、一个 config 和一个 metrics array，并生成 table-facing CSV 与 artifact manifest。

## 5. Code-theory consistency

理论对象是 QDF 对 multi-step task covariance/structure 的 learned quadratic loss，代码确实保存并在 meta-test 阶段使用 learned module `A`。Solar profile 只沿用 ECL 的 released hyperparameters，不能证明其为 Solar 最优，也不能把结果解释成 QDF 的新机制贡献。若 resource smoke 或完整 matrix 出现数值/资源 pathology，只能否定这一 source-informed Solar reproduction contract，不能否定 QDF 方向。
