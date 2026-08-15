# iTransformer-style Decoder-Transfer v1：Design and Prelaunch Gate

Decision：`itransformer_transfer_v1_local_gate_pass_remote_train_validation_authorized_formal_test_blocked`

## 1. Why this carrier is worth testing

[Fact] PatchTST decoder-only HPO v2.1 did not reverse the formal result: +ISCF-BSCA still trailed the native PatchTST readout by 0.436% MSE and 0.568% MAE. Both replacement heads trailed Original Decoder, while BSCA improved the matched +ISCF head. This points to a representation/readout compatibility problem rather than a clean rejection of the BSCA objective.

[Strong Evidence] The official iTransformer maps each full variate history to one token, applies self-attention over variates, and projects every variate token from $D$ to the prediction length. Its latent geometry is therefore `[B,C,D]`, unlike PatchTST's channel-independent patch memory `[B,C,P,D]`. The local carrier preserves that contract as `[B,C,1,D]`, so Original, +ISCF and +ISCF-BSCA consume the same end-to-end learned encoder state.

[Hypothesis] A single contextualized variate token may be more compatible with the ISCF future-domain readout than a flattened bank of PatchTST patch tokens. The experiment tests only this exact carrier compatibility hypothesis; it does not assume that iTransformer will pass.

## 2. Source audit and adopted contract

- official paper: ICLR 2024, *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*;
- official repository: `thuml/iTransformer`, audited commit `c2426e68ca13f74aaec08045c5c724d8ad328124`;
- official tensor path: `x [B,L,C] -> permute [B,C,L] -> Linear(L,D) -> variate self-attention/FFN -> [B,C,D] -> Linear(D,H) -> [B,H,C]`;
- local source-informed path: normalized `x [B,L,C] -> [B,C,L] -> InvertedVariateEncoder -> memory [B,C,1,D]`;
- Original Decoder uses the same shared `Linear(D,720)` semantics; +ISCF and +ISCF-BSCA flatten only the singleton memory axis and receive `[B,C,D]`;
- no official module is imported as a runtime dependency, and this is not labeled an exact native iTransformer reproduction.

Intentional deviations are explicit: the local attention block uses PyTorch `MultiheadAttention` while preserving the official full non-causal variate-attention, residual, LayerNorm and GELU-FFN semantics. Exact numerical parity is not the goal; matched end-to-end attribution is.

## 3. Frozen matrix

The new block contains 3 arms × 5 datasets × seed 2021 = 15 from-scratch joint-training runs:

1. `itransformer_original`: native shared linear readout;
2. `itransformer_iscf`: matched ISCF readout without BSCA objective;
3. `itransformer_iscf_bsca`: identical ISCF readout with the frozen BSCA objective.

Datasets are `ETTm1`, `ETTm2`, `ETTh1`, `ETTh2`, and `Weather`. Each run predicts 720 steps; one validation-selected checkpoint is evaluated on prefixes `{96,192,336,720}`. One dataset-level profile is shared by all three arms. No per-horizon, metric, cell, seed, or test-selected checkpoint is allowed.

Official H720 script priors are retained: `L=96`, batch 32, LR `1e-4`, dropout 0.1, 8 heads, 10 epochs, patience 3; ETTm1/ETTm2/ETTh2 use `D=128, FF=128, 2 layers`, ETTh1 uses `D=512, FF=512, 2 layers`, and Weather uses `D=512, FF=512, 3 layers`.

ISCF mode rank is capacity-matched to the native `Linear(D,720)` head by the nearest integer SIFF parameter count: rank 21 for `D=128` and rank 30 for `D=512`. This avoids turning the carrier test into an uncontrolled decoder-capacity expansion.

## 4. Gates and reporting boundary

Training/manifest gate:

- 15/15 checkpoints and validation scorecards complete and finite;
- 15 unique checkpoint hashes;
- five matched encoder-initialization triplets;
- no formal-test artifact before the immutable manifest.

Formal test remains blocked in this design phase. If later authorized, the complete 60-cell iTransformer block must be reported. Its positive gate requires +ISCF-BSCA versus Original Decoder to improve both macro MSE and macro MAE and win at least 3/5 dataset-mean MSE comparisons.

The existing PatchTST negative block is immutable and remains in the evidence record. If iTransformer passes, the strongest allowed claim is “transfer is supported on the evaluated DLinear-style and iTransformer-style carriers, but is not architecture-agnostic because PatchTST-style failed.” If iTransformer fails, the decoder-portability rescue closes and Section 5.7 remains a negative/limited-scope result.

## 5. Resource and rollback

- remote budget: 15 train/validation runs, no test jobs;
- scheduling: three RTX 3090 GPUs, dataset-major queue with the three Weather arms first, then ETTm1/ETTm2 and hourly ETT jobs;
- estimated wall time: shorter than the 50-run PatchTST HPO matrix; exact estimate is deferred to the three-job smoke;
- OOM rollback: reduce batch size symmetrically for all three arms of that dataset only, without changing model geometry;
- numeric failure rollback: stop the affected matched triplet and diagnose before continuing;
- effectiveness failure rollback: do not open a fourth carrier or another decoder HPO automatically; freeze the negative result and narrow the paper claim.

## 6. Authorization boundary

The user's earlier fallback instruction authorizes the iTransformer-style local source patch and remote train/validation after PatchTST HPO failure. It does not authorize formal test, table mutation, extra HPO, or extra seeds. These remain false until the 15-checkpoint immutable-manifest gate is complete and the user grants the next tier.

## 7. Local gate result

The 28-check local gate passed. It verifies the frozen profile hash, 15 unique jobs, formal-test and table-mutation blocks, joint-training contract, two capacity-matched rank counts, all 15 finite forward/prefix contracts, five matched encoder-initialization triplets, runner syntax, and a 15-job dry run. Protocol SHA256=`21c777aa96f5ea4e5d23658f7dd2d84b3249fef138ffbf28def26c8ee0e3a78b`; machine summary=`local_gate/prelaunch_summary.json`.
