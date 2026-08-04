# Stage C H4M与TimeAlign复现runner说明

## 1. H4M

`scripts/remote/run_iscf_bsca_main_v1_hpo_targeted_h4m.sh`只固定H4M config与output root，实际复用通用HPO runner。通用runner把每个job解析为显式的dataset、encoder widths、patch count、optimizer参数、PCSD rank、LayerNorm与per-job budget；H4M因此可令ETTm2使用60/12、Weather使用90/18，同时维持相同训练与four-H validation selector实现。

`scripts/check_iscf_bsca_main_v1_hpo_targeted_h4m.py`验证：

- architecture与selection invariants；
- 165个历史trial IDs及effective-profile fingerprints；
- 24个新profile无ID或effective-parameter重复；
- ETTm2 patch×LR grid、Weather context/patch pairs与uniform Weather budget；
- frozen evidence hashes与generic runner dry-run的24 jobs、test=0。

## 2. TimeAlign reproduction

`scripts/remote/run_timealign_official_ettm2_weather_reproduction.sh`从JSON按workload order生成8行job queue。每行字段依次为`run_id,dataset,H,seed,d_model,d_ff,dropout,lr,w_align,patch_num,LN,profile_hash`。每个worker把这些字段显式传入`train_repo.py`：tensor path仍是官方fixed-H `timealign-token-mlp` encoder与`official` readout；FATST adapter只改变test访问时机和artifact记录。

`resource-smoke`执行1 epoch、各2个train/eval batches且`final_evaluation_split=none`，不访问test。`run`执行10 epochs、无early stopping，保存last checkpoint后只加载一次test，并产出一个目标H的MSE/MAE、segment metrics和test predictions。

启动前runner逐一验证六个executed source files与两个remote dataset SHA256。完整run需存在checkpoint、effective config、environment、training log、target-horizon metrics、segment metrics、diagnostics、test predictions和run log；缺一项都不能被`status`计为complete。

`scripts/check_timealign_official_ettm2_weather_reproduction.py`验证8个官方preset、source hashes、single-seed full matrix、historical sanity-reference hash、official-last/no-early-stop/test-once边界和dry-run。它不把历史derived CSV升级为artifact-complete复现，也不把`license_unresolved`改写为可再分发许可。

## 3. Code-theory consistency

H4M仍是同一ISCF-BSCA architecture的dataset-level HPO，不改变理论对象。TimeAlign的目标是native external baseline复现而非matched mechanism attribution；即使数值接近论文，也只能支持对应source protocol角色。若adapter与raw official runner结果不同，应先归因test hygiene、runtime或local source deviation，不能把差异解释为ISCF-BSCA机制收益。
