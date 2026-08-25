# Appendix C candidate audit: Weather and ECL

## Scope and provenance

This audit evaluates whether the Appendix C qualitative samples can be
improved without changing the frozen paper-facing model. The probe used the
frozen Main-I/II selected-profile checkpoints, the validation split only,
channel 0, and the same four-horizon mean scaled-MSE score used by the export
rule. It retained the top 64 candidates per dataset and applied no retraining,
test-label access or ablation checkpoint. The existing diversity constraint
(a minimum raw-origin separation of 720 steps) was retained for the published
pair.

## Findings

### Weather

The published first sample (validation window 3914; raw origin 40800) remains
the best candidate in the probe, with a four-horizon mean scaled MSE of
0.03122. The next candidates are tightly clustered around the same seasonal
segment (raw origins 40799--40950), and their long-range trajectories exhibit
the same residual level shift. The second published sample (window 0; raw
origin 36886; score 0.05385) is the lowest-error candidate satisfying the
720-step separation requirement. The probe therefore does not identify a
strictly better, well-separated Weather pair under the frozen selection rule.

### ECL

The published first sample (validation window 222; raw origin 18633) remains
the best candidate, with a score of 0.02875. The lowest-error candidates are
clustered in validation windows 222--262 and share the characteristic ECL
spike pattern; the model follows the baseline level but does not reproduce
every isolated spike. The second published sample (window 1145; raw origin
19556; score 0.15498) is selected by the 720-step separation constraint rather
than by visual or accuracy preference. Relaxing that constraint would yield a
visually cleaner but temporally nearby pair, which would change the declared
selection protocol and reduce sample diversity.

## Decision

The published samples are retained for now. Weather has no materially better
separated candidate under the frozen rule, and replacing the ECL second sample
solely for appearance would introduce an ad hoc exception. The figure is
therefore improved through a cleaner visual encoding of the four nested
prefixes rather than through post-hoc sample cherry-picking. If the authors
prefer a less diverse but more visually homogeneous ECL pair, that should be
recorded as an explicit Appendix C selection-policy revision before changing
the exported arrays.
