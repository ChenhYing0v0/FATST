# PatchEncoderSegmentQueryHead Phase 1 Baseline

`PatchEncoderSegmentQueryHead` keeps the Phase0 patch encoder and replaces the fixed flatten
head with a future segment query decoder.

Purpose:

- test whether explicit future segment states are useful before adding future-aware alignment or MoE;
- keep one-to-one horizon training for the first gate;
- preserve the same dataset split, optimizer, loss, and metrics as `PatchEncoderFixedHead`.

Default segment length is `48`, so horizon `720` uses `15` segment queries.
