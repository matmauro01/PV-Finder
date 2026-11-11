# Negative C: possible causes and remedies for discussion

This follows the [fit and history audit](resolution_c_audit_2026-09-17.md).
The paper, published plots, training targets, and production parameters have
not been changed. The additional checks below are for discussion.

The strongest conclusion is that C should not be interpreted as a measured
resolution floor. It is helping an approximate function describe the shape
of the fitted widths over a limited range. Its negative value remains a
problem for extrapolation. The old zero resulted from a non-negative bound,
and increasing the current statistical errors by a common factor does not
make the current full-range fit compatible with zero.

There are two separate modelling choices to improve: how each residual width
is measured, and the function used to describe those widths versus track
count. The checks support problems with both. They do not yet assign a
separate numerical contribution to each cause of the negative C.

**Why the curve can prefer a negative constant**

For sigma(n) = A n^(-B) + C, the fractional rate of decrease is

```
-d log(sigma) / d log(n) = B * A*n^(-B) / (A*n^(-B) + C).
```

With C = 0 this rate is constant. With C > 0 the curve gradually flattens.
With C < 0 it can become steeper as n increases, while still being positive
throughout the fitted range. This is extra freedom to approximate a changing
shape. It does not establish what the resolution does at arbitrarily large n.
The restricted-range fits in the earlier audit also show that one shape does
not describe all parts of the observed curve equally well.

I tested this explanation with an example whose width is always positive:

```
sigma_reference(n) = 0.252600 mm * (n + 2.62414)^(-0.681281).
```

This function tends to zero and has no negative resolution limit. I evaluated
it at the same 46 n values as the real fit, without adding random noise. I
then fitted these values with A n^(-B) + C, using the original point weights.
That fit gives C = -16.93 µm, close to the -17.97 µm in the real fit.

This is a constructed example, not evidence that the reference function is
the true detector response. It demonstrates that choosing the wrong curve
shape can produce a negative C even when every input comes from a positive
function with a zero limit. Statistical fluctuations and background
contamination are not required for that to happen.

**New check using shared tracks**

I read the first 10,000 events of the same r16633 ROOT file used for the
original fit. I reused the original greedy positional matcher and obtained
926,199 pairs. Their truth track counts match the corresponding prefix of
the saved pairs exactly. Residuals agree within 1e-7 mm, allowing float32
rounding. Each pair was then checked using the exported reconstructed-track
indices in `RecoVertex_assocTracks` and `TruthVertex_assocTracks`.

For a diagnostic selection, define

```
reco overlap  = number of shared tracks / number of reco-associated tracks
truth overlap = number of shared tracks / number of truth-associated tracks.
```

The high-overlap subset requires both fractions to be at least 90%. This is
an illustrative selection, not an official ATLAS category or a proposed
replacement for the paper's matching definition.

| Truth track count | Matched pairs | No shared tracks | Fraction passing both overlap cuts |
| --- | ---: | ---: | ---: |
| 2 | 96,560 | 14.00% | 2.97% |
| 10 | 32,432 | 1.18% | 6.75% |
| 20 | 14,797 | 0.31% | 6.86% |
| 50 to 69 | 12,682 | 0.016% | 3.93% |

Among pairs with |dz| > 0.3 mm, the no-shared-track fraction is 43.4% at n = 2
and 22.1% at n = 10. Thus most pairs in those tails share at least one track.
Shared tracks alone do not prove a correct, uncontaminated association;
partial matches and merged vertices can share tracks too. The result does
show why describing every broad residual as an unrelated match is too simple.

The [shared-track plot](../../outputs/09_17_2026_output/resolution_remedies/shared_track_check.pdf)
shows the total distribution, the subset with no shared tracks, and the
high-overlap subset on the same count scale. The subsets do not exhaust all
pairs. The no-shared-track shape also depends on n. A positional matcher
selects pairs by distance, and its one-to-one constraint couples the choices
within an event. A flat distribution of accidental matches is therefore an
assumption to test, not a consequence of the matching rule.

The strict overlap selection changes the sample substantially:

| Diagnostic at n = 2 | All positional matches | High overlap |
| --- | ---: | ---: |
| Half-width of the central 68% interval | 198.95 µm | 94.02 µm |
| Fraction with abs(dz) > 0.3 mm | 25.73% | 8.23% |
| Gaussian core width from integrated Gaussian + flat Poisson fit | 92.94 µm | 96.45 µm |
| Median fraction of central tracks, abs(eta) < 2.5 | 50% | 100% |

The fitted core width does not change in the same way as the central 68%
interval. These are different measures of spread when tails are present.
Their difference matters when deciding what the target width should mean.
The selected subset is also much smaller and contains more central tracks.
Its improved tail distribution alone is not evidence that it provides the
correct target widths for all eligible vertices.

**Track count is only an approximate measure of vertex precision**

In a simple weighted-mean model with independent position measurements,

```
sigma_vertex^2 = 1 / sum_i(1 / sigma_i^2).
```

The familiar 1/sqrt(n) rule follows only if the distribution of track
precision stays comparable as n changes. The actual AMVF fit also uses track
geometry, covariances, and adaptive weights. The formula above is a useful
check, not its full covariance calculation. The use of adaptive weights is
described in the [ITk paper, Section 4.2](https://arxiv.org/html/2412.15090v2#S4.SS2).

For the matched truth vertices in this 10,000-event sample, the median fraction
of central tracks rises from 50% at n = 2 to 60% at n = 10 and about 71% for
n = 50 to 69. I also computed the simple precision estimate above from the
associated tracks' `RecoTrack_ErrZ0` values. Its median is 96.1 µm at n = 2,
35.9 µm at n = 10, and 11.1 µm for n = 50 to 69. After multiplication by
sqrt(n), its median still changes from 135.9 to 113.4 to 83.5 µm.

That last quantity would be more stable if adding tracks were the only change.
Here the track mixture changes as well. This is a measured reason to doubt a
single fixed scaling with track count. It does not prove how much of C comes
from this effect. The current selection includes only positionally matched
vertices, and that selection can itself depend on track quality and n.

**What to improve in the residual fits**

The earlier full-sample check found large departures from Gaussian + flat,
including chi-square/ndf = 469 at n = 2 and 103 at n = 10. This is not just a
problem with the second fit. Several useful changes should be tested before
settling on new A, B and C values:

1. State the intended width clearly: the narrow core of correctly associated
   vertices, or a width that includes the broader matched population. Keep a
   central 68% interval as a cross-check, without calling it a Gaussian sigma
   when the distribution has substantial tails.
2. Use the shared-track information to separate and inspect the contributions.
   Try a core plus a broader component, or a background shape based on a
   validated matching control sample. Shared-track selections need checks of
   the population they remove. Neither a second Gaussian nor a chosen template
   is automatically the correct description.
3. Integrate the model over histogram bins. The original 20 µm residual bins
   are too coarse to treat their centres as exact samples for the narrowest
   widths. Use finer bins as a stability check, or an unbinned likelihood
   where practical. A Poisson likelihood treats the count statistics better
   in sparse bins. These practices are documented in the
   [ROOT fitting manual](https://root.cern.ch/manual/fitting/).
4. Resample whole events to estimate statistical errors, repeating both fit
   stages. This preserves the dependence between vertices in the same event.
   Assess the choice of residual model and fit range separately. Resampling
   does not remove bias from an inadequate function.
5. Use additional events to constrain the large-n region and reserve a separate
   event sample for validation. The original last point is essentially
   unconstrained, and the useful fitted points only reach about n = 105.

Finer bins, an integrated model, a free mean and Poisson fitting have already
been tested in the earlier audit. They do not by themselves restore C = 0.
These are improvements to how the fit is performed, not a way of selecting
the desired sign of C.

**Options for the width-versus-n curve**

| Option | What the checks say | Assessment |
| --- | --- | --- |
| Set C = 0 and refit A and B | Current chi-square rises from 2,591 to 15,745; chi-square/ndf rises from 60.2 to 357.8 | Positive, but a much poorer description of the present points |
| Require C >= 0 in the original formula | Earlier fits stop at zero | A constraint can enforce positivity but does not establish a measured zero floor |
| Add a positive floor in quadrature: sqrt(A^2 n^(-2B) + s0^2) | A new fit drives s0 to effectively zero and gives the same curve and chi-square as the pure power law | Physically sensible for certain independent contributions, but not sufficient for the present shape |
| Use A (n + n0)^(-B), with positive parameters | Same three-parameter count; chi-square/ndf improves from 60.2 to 21.35 | A useful candidate; still not an acceptable description under the original errors |
| Use a smooth positive curve through reliable width measurements | Not tested here | Could describe the training prescription without interpreting a global constant as a physical floor; requires smoothing, validation and explicit treatment outside the fitted range |
| Depend on track precision as well as n | The track mixture demonstrably changes with n | A longer-term possibility; requires a new target calibration and training study |

The shifted power is an empirical candidate. Its n0 should not automatically
be called a number of extra tracks or a detector parameter. Likewise, an
interpolated curve needs independent validation and should not be judged by
how exactly it passes through the input points. For example, interpolation
in log(width) ensures positive widths, but positivity alone says nothing
about extrapolation. A shape-preserving interpolator can help inside the
measured interval; its default extrapolation must be controlled explicitly
([SciPy PCHIP documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html)).

I would first improve and define the per-bin width measurement, then compare
a small number of positive curves on independent events. I would not choose
the track range or enlarge errors simply to obtain C consistent with zero.

**Implications for the existing results**

The professor is right about the mathematical extrapolation. The current raw
curve becomes negative near n = 309. The existing scan of all 5.17 million
converted events found n <= 214, a smallest raw width of 2.5545 µm and no use
of the 2 µm minimum. Thus those targets did not use negative widths. This
does not establish that the extrapolated widths between about 105 and 214
are well calibrated.

For n >= 2, the actual target prescription is
`sigma_target(n) = max(A n^(-B) + C, 0.002 mm)`. It remains positive even beyond
the observed track range and tends to 2 µm at very large n. That minimum is
an imposed limit in the target construction, not a measured detector floor.
Writing the complete applied rule makes the implementation clear; it does
not validate the extrapolation of the raw fitted curve.

The short-term option is to describe the existing formula honestly as the
empirical target-width prescription used for the reported training. State
its limited justification and do not interpret C as a physical floor. Keep
the professor's comment open while studying a replacement.

Replacing the curve changes more than its appearance in the note. For
example, at n = 214 the current full-precision curve gives 2.56 µm, whereas
the candidate shifted power gives 6.47 µm. The target-building code also
scales the histogram contribution by `max(1, 0.15 mm / sigma)`, so both width
and height would change. A new prescription should be checked in training
and evaluation before being used to describe a new production model. The
existing checkpoint's measured performance remains a result of its existing
training prescription and fixed evaluation settings.

**Draft update to the professor**

I checked the negative C and the older fit. The earlier C = 0 came from a fit
that required C to be non-negative. Refitting those same points without the
bound also gives a negative value. For the current fit, C is below zero even
after a simple rescaling of the errors, so I do not think we can explain it
as a value consistent with zero.

The checks suggest that the main issue is how we model the widths. The
negative constant lets the curve follow a changing slope over the fitted
range, but makes its extrapolation unphysical. The Gaussian plus flat
background also does not fully describe the residual distributions, and a
shared-track check shows that the tails contain a mixture of associations.

I propose to improve the residual-width fits first, then compare functions
that stay positive. One simple positive function already fits the current
points better with the same number of parameters, although it still needs
validation. The existing training targets did not use negative widths: the
largest track count was 214, below the zero crossing near 309. I would keep
the present prescription documented as the one used for the reported
training, and assess the effect of a replacement before changing it.

**Confidence and remaining work**

I am highly confident about the original parameter bound, the current fit
values, the failure of a simple C = 0 refit, the limited meaning of the formal
errors, and the numerical checks reported here. The constructed example
establishes that the function choice alone can produce a negative C. The
track-overlap study establishes that residual populations and track quality
vary with n.

The exact contribution of each effect to the real C is still unresolved.
No replacement residual model, width curve, or new training prescription has
been validated. The overlap check uses the first 10,000 events, not the full
sample, and is diagnostic rather than an independent final validation.

The new check is reproducible with
`venv/bin/python scripts/check_resolution_remedies.py`. It passed Ruff and
completed with the original-pair comparison above. It saves the track-overlap
pairs, [numerical results](../../outputs/09_17_2026_output/resolution_remedies/results.json),
and plot under `outputs/09_17_2026_output/resolution_remedies/`. The original
audit script and outputs remain available separately.
