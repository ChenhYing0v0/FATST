# Phase0 Remote Runner 说明

## 功能边界

`scripts/remote/run_phase0_gate.sh` 是 Phase0 gate 的远程顺序执行入口，不改变
model、dataset 或 training 语义。它只负责把已经定义的三个 baseline 训练脚本按
固定矩阵调度到 `529_Lab-3090`。

## 数据流与执行流

1. 读取环境变量或默认值：
   - `DATASET_ROOT=${DATA_ROOT:-/home/yingch/dataset}`，兼容
     `529lab-3090.env` 中的 `DATA_ROOT`
   - `OUTPUT_ROOT=artifacts/runs/phase0`
   - `LOG_ROOT=artifacts/logs/phase0_gate`
   - `CONDA_ENV=${CONDA_ENV_NAME:-moe}`，兼容 `529lab-3090.env` 中的
     `CONDA_ENV_NAME`
   - `CONDA_BIN` 默认为自动定位：先查 `PATH`，再查
     `/home/anaconda3/bin/conda` 与 `/data/anaconda3/bin/conda`
   - `GPU_ID=1`
   - `SEED=2021`
   - `EPOCHS=100`
2. 固定遍历矩阵：
   - models: `DLinear`, `PatchEncoderFixedHead`, `SegTSFTDenseFixedHead`
   - datasets: `ETTh2`, `ETTm1`, `Weather`
   - horizons: `96`, `192`, `336`, `720`
3. 每个 run 启动前检查 `metrics.json` 与 `checkpoint.pt`。两者都存在时跳过，
   用于远程中断后的保守续跑。
4. 每个 run 启动前写出当前 `nvidia-smi`，随后通过
   `CUDA_VISIBLE_DEVICES=${GPU_ID}` 将训练进程限制到选定 GPU。
5. 每个训练脚本仍然自行写出标准 artifact：
   `checkpoint.pt`, `predictions_test.npz`, `metrics.json`,
   `metrics_by_horizon.csv`, `metrics_by_segment.csv`, `training_log.csv`,
   `effective_config.json`, `environment.json`。

## 一致性评价

[Fact] 该 runner 没有引入新的模型机制，也没有覆盖 baseline 内部默认训练逻辑。

[Strong Evidence] 它只把原有 `train.py` 的命令行参数显式化，并记录 Git commit、
GPU、dataset root、output root、conda env 与 conda executable。

[Speculative] 当前使用单 GPU 顺序运行，牺牲吞吐换取可审计性与显存安全。若后续
确认 GPU 1/2 长时间空闲，可以再增加并行调度，但那会改变资源冲突风险。
