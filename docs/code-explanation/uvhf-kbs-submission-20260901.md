# UVHF KBS submission refinement

## Purpose

This update refines the canonical `UVHF_KBS_submission/` package without
changing the model, experiment results or paper claims.

## Manuscript changes

- Highlights use a one-line title and spell out all paper-specific concepts so
  that the metadata is readable independently of the Abstract.
- All display mathematics is numbered automatically with `equation`.
- The balance schedule is parameterized by
  $\lambda_{\mathrm{balance}}^{\max}$ (maximum balance weight) and
  $u_{\mathrm{ramp}}$ (normalized optimizer progress at which the maximum is
  reached). Appendix A records the frozen values $0.1$ and $0.25$.

## Figure 7

The grouped bars now follow the comparison order
`Original Decoder -> UVHF (MSD + BCA)`. The arrow therefore points from the
baseline bar to the proposed decoder bar. Source data and reported relative MSE
reductions are unchanged.

## Verification boundary

The update is verified through source-level equation counting, figure static
preflight, LaTeX compilation and rendered-page inspection. It does not rerun
training or evaluation because no experimental artifact changed.
