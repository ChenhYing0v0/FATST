# Main II H720-prefix execution code explanation

## 1. Functional boundary

This execution path implements the frozen `source_native_h720_trained_one_model_all_horizons_system_benchmark`. It does not change ISCF-BSCA model code and does not turn heterogeneous external baselines into matched mechanism controls.

The path has four modules:

1. `configs/iscf_bsca_main_ii_h720_execution.json` freezes source commits, key source/script hashes, seven dataset hashes, the 21-job order, resource limits and Solar adaptation role.
2. `scripts/run_main_ii_h720_training_job.py` verifies an exact upstream checkout, copies a repo-external execution workspace, applies narrow runtime patches, extracts the released H720 command, and runs one smoke/training/test unit.
3. `scripts/evaluate_main_ii_h720_prefix_arrays.py` converts one upstream H720 prediction/target pair to canonical `[origin,time,channel]` and recomputes H96/H192/H336/H720 MSE/MAE from exact views of the same tensor.
4. `scripts/check_main_ii_h720_prelaunch.py` and `scripts/remote/run_main_ii_h720_training.sh` enforce the 21-training/70-evaluation matrix and the local → smoke → training → test ordering.

## 2. Runtime source patch

The runtime adapter deliberately changes only protocol hygiene and compatibility:

- removes construction and evaluation of the official test loader inside `train()`;
- preserves the native validation loss used by `EarlyStopping`;
- disables the automatic test call after training;
- replaces the upstream `testing` log marker with an explicit deferred-test marker so log audits do not confuse a disabled call with actual test access;
- makes formal `is_training=0` evaluation load from `args.checkpoints`;
- enables `true.npy` export where upstream commented it out;
- adds environment-bounded train/validation loops only for resource smoke;
- replaces `np.Inf` with `np.inf` for NumPy 2 compatibility.

PatchTST and DLinear Solar additionally copy the exact audited `Dataset_Solar` semantics from iTransformer and register `data='Solar'`. Their optimization command is the released ECL H720 profile with `data_path`, `enc_in=137` and `model_id` adapted as preregistered. These two cells remain labeled `source_informed_not_official`.

DLinear's native `data_factory` passes an additional `train_only` keyword to every dataset class. The DLinear Solar runtime class therefore accepts and ignores extra keyword arguments; this is a constructor-compatibility patch and does not alter data values, splits, scaling or optimization.

## 3. Command extraction and artifacts

Official shell scripts are executed with a temporary no-op `python` capture shim. All commands are captured after native shell-variable expansion, and exactly one command with `--pred_len 720` must exist. The adapter then changes only dataset absolute path, `num_workers=0`, isolated checkpoint root, execution phase and preregistered Solar fields.

Each unit writes:

- `effective_command.json`: exact argv, source/script identity and test role;
- `run.log`: stdout/stderr;
- `artifact_manifest.json`: checkpoint and log SHA256;
- `DONE`: completion token;
- formal test additionally writes `pred.npy`, `true.npy` and four-row prefix metrics.

Checkpoint SHA256 is computed before and after formal test; mutation is a hard failure.

## 4. Prefix metric semantics

For canonical tensors `prediction,target ∈ R^{N×720×C}`, horizon `H` uses `[:, :H, :]` from the same saved arrays. MSE and MAE are global elementwise means computed in float64. Each row records origin/channel counts plus hashes of both prefix tensors. This establishes exact within-checkpoint prefix identity by construction; it does not remove cross-system differences in native official-test loaders.

## 5. Falsification and rollback

The execution path fails closed if a source/script/dataset hash differs, an official script yields zero or multiple H720 commands, training still references `flag='test'`, a checkpoint is missing or mutated, tensors are non-finite/non-H720, or any matrix count is incomplete. A Solar adapter failure blocks the complete table but does not justify deleting Solar or reporting a favorable subset.
