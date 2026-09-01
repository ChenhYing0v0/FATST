# UVHF KBS submission refinement audit

Date: 2026-09-01

## Requested changes

1. The Highlights title is typeset on one line.
2. Highlights are self-contained and avoid paper-specific abbreviations.
3. The Abstract closes with “effective foundation”.
4. Every display equation uses a numbered `equation` environment.
5. The BCA schedule is written with
   $\lambda_{\mathrm{balance}}^{\max}$ and $u_{\mathrm{ramp}}$; both symbols are
   defined in the method and Appendix A, where the frozen values are retained.
6. Figure 7 presents `Original Decoder` before `UVHF (MSD + BCA)`.

## Verification

- LaTeX compilation: passed with `latexmk -pdf`.
- Final manuscript length: 24 pages.
- Display-equation audit: 30 starts, 30 ends, and no remaining `\[...\]`,
  `$$...$$`, `equation*` or `align*` environments.
- Layout log: no overfull or underfull boxes after the Highlights title was
  locally reduced to `\small` on the Highlights page.
- Figure 7 static preflight: 13 passes, 1 reviewed width warning and 0
  failures. The 174-mm source canvas is intentional for the KBS full-width
  figure placement.
- Figure 7 visual QA: paired-bar order, legend order and annotation direction
  agree; values and percentage reductions are unchanged.
- Full PDF render: all 24 pages rendered successfully and representative pages
  covering Highlights, equations, Figure 7 and the Appendix were inspected.

## Result

The refined package remains scientifically identical to the approved canonical
UVHF manuscript and is suitable as the current KBS submission source.

## Frozen submission snapshot

The author confirmed this audited package as the temporarily frozen,
submission-ready version on 2026-09-01. The scientific content is anchored to
commit `c0eb5974` with the following SHA-256 digests:

- canonical TeX: `5f63fed2899f92177bb99e2abdd5bd3be30489162050f2da58ff1d626d8450ec`
- compiled PDF: `bce1823779ede5728f37e5abe79b17e72dfbf72009463cd8195c57bc8d159555`
- Figure 7 manuscript asset: `c6af340d87780d4d997a7d42209a53f86cb087aedacca3e49003fead02bfb7f0`

The freeze is a paper-governance state and does not alter any file inside the
flat `UVHF_KBS_submission/` package.
