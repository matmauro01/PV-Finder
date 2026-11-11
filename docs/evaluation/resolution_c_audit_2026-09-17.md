# Check of the negative C in the HL-LHC resolution fit

Checked on 17 September 2026. This is a diagnostic study. The paper, its figures,
and the parameters used to build training targets have not been changed.

The old HL-LHC fit did give C = 0. That fit required C to be non-negative and
stopped at the lower bound. Refitting the same saved points without that bound
gives a negative C. The old zero therefore does not show that an unrestricted
fit measured a value consistent with zero.

The current full-range fit also prefers a negative C. Zero is outside its
reported error bounds. However, the fit is poor, so those errors should not be
treated as a complete uncertainty on a physical resolution limit. The value
of C changes substantially with the track range and the way the residuals are
fitted. The professor's concern remains valid.

**What the earlier fits did**

All fits below use sigma(n) = A n^(-B) + C. A and C are shown in micrometres.

| Saved result | Residual fit | Constraint on C | A [µm] | B | C [µm] |
| --- | --- | --- | ---: | ---: | ---: |
| 1 June, original track acceptance | Gaussian only | C >= 0 | 178.979 | 0.72744 | 0 |
| 15 July, original track acceptance | Gaussian + flat component | Free | 123.872 | 0.45826 | -7.277 |
| 31 July, extended track acceptance, before background fit | Gaussian only | C >= 0 | 221.829 | 0.72035 | 0 |
| 31 July, extended track acceptance, current fit | Gaussian + flat component | Free | 143.175 | 0.36200 | -17.966 |

The original acceptance was |eta| < 2.5; the extended sample counts tracks out
to |eta| < 4. These samples should not be treated as identical.

The June code explicitly contains `f.SetParLimits(2, 0.0, 1e3)`, where parameter
2 is C. This line is present in both the initial residual-fit commit `8ae719c`
and the full-file fit commit `faaae0b`, both dated 1 June. The saved June C is
actually +1.44e-9 µm and the uncorrected July C is +7.55e-9 µm. These are
numerical values at the zero boundary, not evidence of a positive floor.

The older May files also have tiny positive values consistent with numerical
zero. They describe a track-based proxy, rather than these AMVF-truth residual
fits, so they are not included in the table. I found no clearly positive C in
the saved full-range HL-LHC fit files checked here.

To isolate the effect of the bound, I refitted the saved Gaussian-only widths,
keeping their original point errors:

| Saved points | With the original C >= 0 bound | With C free |
| --- | ---: | ---: |
| 1 June | C = 0 | C = -11.90 +/- 2.91 µm |
| 31 July, before background fit | C = 0 | C = -18.99 +/- 3.88 µm |

The constrained refits reproduce the saved A and B to better than 0.01%.
Thus the preference for a negative C was already present in the older widths.
It was not introduced solely by fitting a flat background.

Other settings also changed. The Gaussian-only method fitted a free mean and
used an adaptive residual range. It imposed a minimum point error of 10% of
the width. The current method fixes the Gaussian mean at zero, fits over
[-2, 2] mm, includes a flat component, and uses a 0.2% minimum point error.
These changes affect both the fitted widths and their weights. The old small
error quoted on a C at its lower bound is not an unrestricted, symmetric error
around zero.

**Does the current C include zero within errors?**

I reproduced all 46 saved widths and their errors from the 9,243,820 saved
AMVF-truth residual pairs. The second fit reproduces the saved parameters:

| Parameter | Value | Formal fit error |
| --- | ---: | ---: |
| A | 143.1752 µm | 0.1345 µm |
| B | 0.361997 | 0.001643 |
| C | -17.9660 µm | 0.2386 µm |

The chi-square is 2590.60 for 43 degrees of freedom, giving chi-square/ndf =
60.25. These formal errors use the supplied point errors as absolute errors.
They do not include the failure of the fitted functions to describe the data.
This is the meaning of `absolute_sigma=True` in
[SciPy's curve_fit documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html).

Several checks address whether a simple error treatment could make C = 0:

| Check | Result for C |
| --- | --- |
| Original covariance error | -17.97 +/- 0.24 µm |
| Multiply every point error by sqrt(60.25) | -17.97 +/- 1.85 µm |
| Refit A and B at each fixed C, original errors | Conditional 95% interval: [-18.43, -17.51] µm |
| Same profile, allowing a common error scale | Approximate 95% interval: [-21.90, -14.54] µm |

The last interval uses a chi-square increase of
`(chi-square/ndf) * F_0.95(1, 43)`. This is a diagnostic under a common error-scale
assumption, not a full treatment of model uncertainty. The correlation between
B and C is +0.983. Refitting A and B in the profile check allows for this
correlation; it does not restore compatibility with zero.

A separate linearized HC3 covariance calculation, which allows the point
scatter to vary across n, gives a C error of 6.32 µm. This is much larger than
the simple common-scale error and shows that the uncertainty estimate itself
depends on the assumptions. It is also not a complete physical uncertainty.
The n = 2 point has high influence and contributes 918 to the total
chi-square. I would not quote a number of standard deviations as evidence for
a negative physical resolution limit.

The saved pairs do not contain event identifiers. These checks therefore do
not include an event-level bootstrap or an estimate of correlations between
widths from vertices in the same event.

**Dependence on the fitted track range and weights**

| Change to the second fit | C [µm] | chi-square/ndf |
| --- | ---: | ---: |
| Current fit | -17.97 | 60.25 |
| Remove n = 2 | -12.64 | 23.18 |
| Use n >= 5 | -8.93 | 9.96 |
| Use n >= 10 | -10.04 | 10.66 |
| Use n >= 20 | -27.78 | 5.36 |
| Use n >= 30 | -27.93 | 5.73 |
| Use 40 <= mean n < 120 | -4.45 | 5.45 |
| Use 50 <= mean n < 120 | +4.71 | 1.34 |
| Use n <= 30 | -37.95 | 53.34 |
| Use n <= 100 | -18.10 | 62.66 |
| Raise the minimum relative error to 1% | -15.78 | 3.24 |
| Raise it to 3% | -13.75 | 0.49 |

For the fit above 40 tracks, the C error is 2.19 µm with the supplied errors,
or 5.12 µm after a common rescaling. For the fit above 50 tracks, these errors
are 1.32 and 1.52 µm. Both fits include C = 0 in the approximate 95% profile
test allowing a common error scale. The latter uses only eight points and has
five degrees of freedom; a symmetric covariance error alone misses some of
the shape of the C profile.

The positive value above 50 tracks is a new restricted-range check. It is not
an older full-range result, and it does not justify replacing the current fit
with that selected range. It shows how strongly C depends on which part of
the curve is fitted.

Removing each of the 46 points in turn gives C between -18.45 and -12.64 µm.
Five different starting parameter choices converge to the same full-range
minimum. Equal relative weights give C = -13.75 µm; equal absolute weights
give -22.80 µm. Those last two checks exclude the poorly determined final
point and test weighting only; their arbitrary error normalization is not a
statistical uncertainty model.

The last saved point, at mean n = 121.21, has a width of about 5.03 µm and a
formal error of 5620 µm. The original histogram cannot determine its width
well. Its weight is negligible, and removing it leaves the current C unchanged.
The original fit therefore has useful point constraints only up to roughly
105 tracks, despite listing a point above 120.

**Checks of the residual fits themselves**

The Gaussian plus a flat component does not fully describe the residual
histograms. For n = 2 the residual fit has chi-square/ndf = 469.3; for n = 10
it is 102.8. The tails rise towards the centre, which a flat component cannot
follow. See the [residual plots](../../outputs/09_17_2026_output/resolution_c_audit/residual_fit_checks.pdf).

I repeated these fits with different histogram widths, fit ranges, and fit
methods, then refitted A, B and C. Here, integrating means integrating the
Gaussian over each histogram bin instead of evaluating it only at the bin
centre.

| Residual-fit choice | C from the full-range width fit [µm] |
| --- | ---: |
| Original: 20 µm bins, evaluated at centres | -17.97 |
| 20 µm bins, integrated | -22.60 |
| 10 µm bins, evaluated at centres | -24.43 |
| 10 µm bins, integrated | -26.04 |
| 5 µm bins, integrated | -26.63 |
| Integrated 10 µm bins, residual range +/-1 mm | -34.04 |
| Integrated 10 µm bins, residual range +/-0.5 mm | -52.94 |
| Integrated 10 µm bins, Poisson likelihood | -26.01 |
| Integrated 10 µm bins, free Gaussian mean | -26.04 |

The original 20 µm residual bins are comparable to or wider than the Gaussian
widths at high track counts. Accounting for the finite bin width matters.
Finer bins do not bring C towards zero. Changing the residual range also
changes the fitted core width because the model does not describe the tails
well. These variations are checks of that dependence, not calibrated
systematic errors to combine in quadrature.

As another check, I enlarged each first-stage width error by the square root
of its own chi-square/ndf, without shrinking any errors, before applying the
0.2% minimum and repeating the second fit. This gives C = -14.74 +/- 0.88 µm
and second-stage chi-square/ndf = 1.71. A further common rescaling gives an
error of 1.15 µm. Again, C remains negative under that prescription.

**Why a negative C helps this formula**

For a positive curve sigma(n) = A n^(-B) + C, the magnitude of its logarithmic
slope is

```
-d log(sigma) / d log(n) = B * A*n^(-B) / (A*n^(-B) + C).
```

When C is negative, the curve can fall more steeply as n increases. The fit
uses this freedom to follow the changing slope across the measured range.
It is fitting the shape over that range; the data do not measure a resolution
at infinite track count. The negative limit is an unacceptable extrapolation
of this particular formula.

Forcing C to zero and refitting A and B makes the current fit worse:
chi-square/ndf rises from 60.25 to 357.84. Simply setting C to zero while
keeping A and B would not even be the best fit under that restriction.

As an example of another shape, the positive function
`sigma(n) = A * (n + n0)^(-B)`, with A, B and n0 non-negative, gives
chi-square/ndf = 21.35 with the same number of parameters as the current
formula. It stays positive and tends to zero. It is still a poor fit with the
original errors, so this is a useful test, not a validated replacement.
See the [curve comparison](../../outputs/09_17_2026_output/resolution_c_audit/resolution_model_comparison.pdf).

Differences in track precision, the track acceptance, and the positional
matching can all affect the residual shape. This study does not isolate their
individual contributions. It establishes that the present residual model,
point errors, and global curve are too restrictive to interpret C as a
well-measured physical floor.

The current raw curve crosses zero near n = 309. The training code requires
widths to be at least 2 µm. The existing 12 September audit of all 5.17 million
converted events found a largest track count of 214, a minimum raw width of
2.5545 µm, and no use of the floor. I checked that saved audit and the current
target-building code; I did not repeat that full dataset scan here. The
extrapolation problem therefore does not mean negative widths were used for
those training targets.

I would keep the professor's comment open. The next step is to choose a more
suitable description of the residual core and a positive width function, then
check both before changing the target prescription. The earlier C = 0 and
the present formal errors do not settle that choice.

**Files and reproduction**

Run `venv/bin/python scripts/check_resolution_c.py` from the PV-Finder root.
The script checks the saved parameters and every original residual width,
performs the numerical variations, and writes the plots and results to
`outputs/09_17_2026_output/resolution_c_audit/`. It records hashes of the source
data, historical fit files, and script. The script passed Ruff checks and ran
to completion. All numerical results are in
[results.json](../../outputs/09_17_2026_output/resolution_c_audit/results.json).

The main source files are:

- [Original fitter and C bound](../../src/pv_finder/diagnostics/amvf_resolution_vs_ntracks.py).
- [Current two-stage fitter](../../src/pv_finder/diagnostics/resolution_fit_v2.py).
- [Saved June fit](../../outputs/06_01_2026_output/amvf_resolution_residuals/fit_params.json).
- [Saved July central-acceptance fit](../../outputs/07_15_2026_note_figs/resolution_fit_v2.json).
- [Saved July Gaussian-only fit](../../outputs/07_31_2026_output/amvf_resolution_alleta/fit_params.json).
- [Saved current fit](../../outputs/07_31_2026_output/amvf_resolution_alleta/corrected/resolution_fit_v2.json).
- [Target-width calculation](../../src/pv_finder/data/root_to_h5.py).
- [Earlier full-dataset floor audit](../../../ANA-IDTR-2026-01-INT1/doc/target_width_floor_2026-09-12.json).

Some older prose in the repository still calls C an irreducible floor or
describes the large chi-square as only a sample-size effect. Those descriptions
are not sufficient given these checks and were not used as evidence here.
