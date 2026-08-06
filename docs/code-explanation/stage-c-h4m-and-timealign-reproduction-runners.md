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

`scripts/remote/run_timealign_official_ettm2_weather_reproduction.sh`从JSON按workload order生成8行job queue。每行字段依次为`run_id,dataset,H,seed,d_model,d_ff,dropout,lr,w_align,patch_num,LN,profile_hash`。这些参数用于checker逐项核对official preset；runner不传Stage C专用`legacy-*` overrides，而由`train_repo.py`按dataset与H直接加载已审计的official preset。tensor path仍是官方fixed-H `timealign-token-mlp` encoder与`official` readout；FATST adapter只改变test访问时机和artifact记录。

`resource-smoke`执行1 epoch、各2个train/eval batches且`final_evaluation_split=none`，不访问test。`run`执行10 epochs、无early stopping，保存last checkpoint后只加载一次test，并产出一个目标H的MSE/MAE、segment metrics和test predictions。

adapter当前会对未激活的historical grouped-MLP参数执行全局argument validation；runner因此传入可整除四个目标H的`grouped_mlp_scale=48`。该值不进入`readout_mode=official`的tensor path，只避免irrelevant default 144在H336上触发pre-training guard。

启动前runner逐一验证六个executed source files与两个remote dataset SHA256。完整run需存在checkpoint、effective config、environment、training log、target-horizon metrics、segment metrics、diagnostics、test predictions和run log；缺一项都不能被`status`计为complete。

`scripts/check_timealign_official_ettm2_weather_reproduction.py`验证8个官方preset、source hashes、single-seed full matrix、historical sanity-reference hash、official-last/no-early-stop/test-once边界和dry-run。它不把历史derived CSV升级为artifact-complete复现，也不把`license_unresolved`改写为可再分发许可。

## 3. Code-theory consistency

H4M仍是同一ISCF-BSCA architecture的dataset-level HPO，不改变理论对象。TimeAlign的目标是native external baseline复现而非matched mechanism attribution；即使数值接近论文，也只能支持对应source protocol角色。若adapter与raw official runner结果不同，应先归因test hygiene、runtime或local source deviation，不能把差异解释为ISCF-BSCA机制收益。

## 4. Returned-artifact与formal-test tooling

`scripts/check_iscf_bsca_main_v1_h4m_training_artifacts.py`在remote output root逐checkpoint验证24个training artifacts、effective config、four-H validation selector、numeric health、log failure patterns与checkpoint SHA256；它明确要求test artifact在freeze时不存在。`scripts/build_iscf_bsca_main_v1_h4m_test_manifest.py`只从通过审计的ledger生成24-row immutable manifest，固定remote artifact/test paths和pre-test hashes。

`scripts/remote/run_iscf_bsca_main_v1_hpo_targeted_h4m_test_audit.sh`复用generic atomic evaluator。`scripts/check_iscf_bsca_main_v1_hpo_targeted_h4m_test_audit.py`验证24 checkpoints、96 standard cells、一次formal-test authorization、dataset-level shared-profile selector和禁止per-H/per-metric/per-cell rescue的边界。

`scripts/analyze_timealign_official_ettm2_weather_reproduction.py`从8个remote-lite metadata directories和remote checkpoint-hash list构造artifact manifest，逐项核对official preset、10-epoch official-last contract、test-only final evaluation、MSE/MAE与failure patterns，并同时输出相对paper Table 6和historical local rerun的逐cell偏差。每个CSV列均直接来自对应`metrics_by_target_horizon.csv`或其明确reference：`*_vs_published_pct=100(reproduced/published-1)`，`*_vs_historical_pct`同理。

H4M formal test结束后，generic `scripts/analyze_iscf_bsca_main_v1_hpo_test_audit.py`重新计算checkpoint SHA256，要求每个test目录同时存在720行dense metrics、invariants与diagnostic NPZ，并从中抽取H96/H192/H336/H720形成`all_trial_scorecard.csv`。`scripts/analyze_iscf_bsca_main_v1_joint_objective.py`再把H1--H4M全部189个trials合并，先计算dataset-level joint MSE/MAE relative mean及1% guard，再按leading-cell、balanced metric coverage、joint score、validation score、parameter count和profile ID依次tie-break。`configs/iscf_bsca_main_v1_selected_profiles.json`只镜像该189-trial合法selector结果，不允许per-H、per-metric、per-seed或per-cell重选。

## 5. Eight-dataset Main I extension

`scripts/remote/run_timealign_official_main_i_reproduction.sh`把8 datasets × four H展开为32个fixed-H jobs。ETTm2/Weather的8个run IDs解析到旧artifact root并在hash/required-artifact检查通过后复用；其余24个解析到新root。每个new job的`profile_hash`由run ID、dataset、H、seed、epochs、batch size与official preset字段的canonical JSON计算。runner不传`learning_rate`、`w_align`或`legacy-*` overrides，实际tensor path继续由`train_repo.py::OFFICIAL_PRESETS`构建。

官方脚本的budget差异被显式保留：ETTh1 H96为1 epoch，ECL batch size为16；其他new jobs为10 epochs与batch size 32。新runs使用`--no-save-predictions`控制remote quota，但checkpoint、effective config、environment、training log、target/segment metrics、diagnostics与run log仍是complete gate。该retention差异只影响post-hoc prediction-level diagnostics，不影响Main I MSE/MAE。

`scripts/check_timealign_official_main_i_reproduction.py`验证12个executed source hashes、8个dataset hashes、32-run cross product、8 reusable/24 new边界、Exchange非官方preset标签、test-once contract与runner dry-run。Exchange的参数来自本地source-audited ETTh1 bootstrap，不能写成TimeAlign官方Exchange result。

`scripts/freeze_iscf_bsca_main_v1_final_profiles.py`独立扫描各HPO阶段的checkpoint manifests、training ledgers与test ledgers，逐dataset核对checkpoint test前后hash、four-H cells和profile means，再生成8-profile provenance manifest与32-cell terminal scorecard。该脚本只物化已冻结selector，不重新选择profile。

`scripts/analyze_timealign_official_main_i_reproduction.py`合并24个new remote-lite
directories与8个reused ETTm2/Weather directories，并以remote生成的32-checkpoint
SHA256清单作为binary immutability证据。它逐job核对dataset/H/seed、source hash、
dataset hash、official-last budget、test-once role、metrics shape和failure patterns；
对reuse runs还要求先前artifact manifest完全通过。输出字段中
`mse_vs_published_pct=100(local_mse/published_mse-1)`，MAE同理；Exchange没有
published reference，因此对应字段留空且`preset_role`固定为
`source_informed_etth1_bootstrap_not_official`。任何missing cell、重复hash、source
drift或numeric failure都会中止，不能生成可供table builder读取的32-row scorecard。
