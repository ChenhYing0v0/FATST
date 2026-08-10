# Main II H5A HPO 代码说明

## 1. 功能边界

H5A只重启`ETTh1/ECL/Solar`的`ISCF-BSCA-MAIN-v1` hyperparameter search。模型
forward、loss、scope partition与H720-prefix inference均未修改；代码变化只包括实验
合同、remote wrapper、prelaunch checker、training-artifact checker与formal-test manifest
builder。

## 2. 运行路径

`scripts/remote/run_iscf_bsca_main_v1_hpo_main_ii_h5a.sh`把H5A config和独立output root
传给既有generic HPO runner。Runner将48个job specification与`base_profiles`合并，形成
effective profile，然后调用`baselines/timealign_official/train_repo.py`：输入历史tensor为
`[B,L,C]`，模型一次输出最大future field `[B,720,C]`，validation evaluator裁剪
H96/H192/H336/H720并取mean MSE选择checkpoint。训练阶段显式设置
`official_test_mode=False`和`final_evaluation_split=val`。

## 3. 审计链

- `check_iscf_bsca_main_v1_hpo_main_ii_h5a.py`：检查48-job矩阵、source hashes、参数合法性、
  frozen Main II target、历史trial identity与dry-run的`test_jobs=0`。
- `check_iscf_bsca_main_v1_h5a_training_artifacts.py`：逐trial核对effective config、四个
  validation metrics、best epoch、log numeric health、checkpoint SHA256和test absence。
- `build_iscf_bsca_main_v1_h5a_test_manifest.py`：只接受48/48 `validation_complete` rows，
  并要求48个unique checkpoint hashes，生成formal-test immutable manifest。

## 4. Code-theory consistency

理论合同要求one dataset-level profile共享四个horizons，同时由validation选择checkpoint、
由完整official-test surface选择hyperparameters。实现保持H720统一trajectory与prefix crop，
没有引入requested-H input或per-H head，因此与Main II one-model contract一致。H5A只能证明
frozen architecture family内的performance potential；即使best cells增加，也不能归因于
BSCA mechanism，mechanism claim仍需matched ablation与internal diagnostics。
