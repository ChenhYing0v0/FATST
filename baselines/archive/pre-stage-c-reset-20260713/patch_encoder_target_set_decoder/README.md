# PatchEncoderTargetSetDecoder

Phase1-R target-set candidate.

This baseline keeps the Phase0 patch encoder contract but replaces the
horizon-specific fixed head with a mixed-horizon target-set decoder. Future
segments are explicit model inputs. Each target segment query reads history
patch states and conditions a dense history readout, so the candidate tests a
target-indexed forecasting interface without repeating the low-capacity
`SegmentQueryHead` design.

Default target horizons are `96,192,336,720`, with `segment_len=48` and
`max_pred_len=720`.
