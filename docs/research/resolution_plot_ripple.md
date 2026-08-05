# The ripple in the resolution plot is a binning beat

**Status: measured on held-out data 2026-08-05. Fixed.**

- Code: `src/pv_finder/utils/pairwise_dz.py`,
  `src/pv_finder/evaluation/vertex_finding/run_eval_pvf_run3.py`
- Tests: `tests/test_pairwise_dz_binning.py`
- Numbers and figures: `outputs/08_05_2026_output/ripple_study/`

Two questions were asked together — clean up the resolution plot, and cut the
fake rate post-hoc. The first has a free answer and is sections 1–4. The second
does not, and is [section 5](#5-the-other-half-no-post-hoc-lever-cuts-the-fake-rate).

The plateau of `resolution_plot.png` alternated between two levels by ~3.9 %,
four times its own error bars, across the whole ±6 mm range. It is not noise, it
is not satellites, and it is not the model. It is a beat between the 0.04 mm
quantisation of reconstructed peak positions and the 0.05 mm bins the plot used.
Changing the plot binning to 0.04 mm removes it, and moves no peak.

> **The sawtooth is exactly one day old: it arrived with the local-centroid
> estimator on 2026-08-04.** That estimator change was a genuine win on the
> physics and it stands, but it also made the resolution plot look dramatically
> worse for a purely presentational reason. Anyone comparing a plot from before
> 08-04 with one from after needs that sentence or they will conclude the
> resolution degraded. It did not — it improved by 13 %. See
> [the provenance measurement](#the-ripple-is-one-day-old).

> **Two different things have been called "the ripple".** This page is about the
> visible sawtooth in the *resolution plot*, whose cause is the position
> estimator meeting the plot binning. The sub-noise structure in the *predicted
> histogram* that the conjoined-split rule turns into satellite peaks is a
> different phenomenon with a different cause, and it is
> [pairwise_dz_bump](pairwise_dz_bump.md). Only this one has a free fix; the
> other one needs a retrain, for reasons that page now records.

---

## 1. The positions are combed at 0.04 mm

The peak finder reports a value-weighted centroid over the bins within 3 of the
region maximum. For a peak that is sharp compared with the 0.04 mm output grid,
that centroid sits very close to the maximum's own bin. Measured on 7680 held-out
r16443 events at the deployed operating point, position fractional part within a
bin, 10 sub-bins:

| sub-bin | 0.0–0.1 | .1–.2 | .2–.3 | .3–.4 | .4–.5 | .5–.6 | .6–.7 | .7–.8 | .8–.9 | .9–1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| counts (k) | 134.7 | 106.1 | 73.9 | 50.4 | 29.4 | 31.5 | 47.0 | 70.0 | 100.3 | 131.5 |

**135.8 %** peak-to-trough modulation, piled up at the bin boundaries. |Δz|
between two such positions is therefore combed at the same 0.04 mm pitch, with
**40.0 %** modulation.

This is a property of the estimator, not a defect in it: the alternatives that
de-quantise cost more than they buy (see
[section 4](#4-de-quantising-the-positions-instead-costs-more-than-it-buys)).

### The ripple is one day old

The comb, and therefore the sawtooth, arrived with the local-centroid estimator
on 2026-08-04. The historical full-region weighted mean averaged over the whole
region — median width 14 bins — so its positions were continuous and there was
nothing to beat against. Same peaks, same events, same 240-bin plot:

| position source | \|Δz\| comb | plateau pull RMS, 240 bins |
|---|---|---|
| **PV-Finder, local centroid ±3** (since 2026-08-04) | **39.4 %** | **4.41** |
| PV-Finder, full-region weighted mean (before) | 0.9 % | 1.77 |
| *AMVF* — continuous positions from a fit | 1.3 % | 1.64 |
| *truth vertices* | 0.8 % | 1.35 |

Both controls behave, and the two PV-Finder rows differ only in the estimator.
`outputs/08_04_2026_output/eval_v6_heldout/r16443/resolution_plot.png` (the last
eval before the estimator changed) has a visibly smooth plateau;
`outputs/08_04_2026_output/eval_v6_operating_point/r16443/resolution_plot.png`
(the first one after) has the sawtooth. So this is not a long-standing defect
that was missed — it is a side effect of an otherwise good change, and it costs
nothing to undo.

The comb is also independent of the operating point, as a quantisation artefact
must be: 39.7 % at integral 0.20 / height 0.00, 39.4 % at the deployed
0.30 / 0.03, 40.0 % on the resolution list at 0.50 / 0.03.

## 2. 0.05 mm bins beat against it at 0.20 mm

`lcm(0.04, 0.05) = 0.20 mm`, so a 240-bin histogram across ±6 mm gets a 4-bin
sawtooth. Held-out r16443, resolution peak list (the one the plot is built
from), plateau fitted with a straight line over 1.2–6 mm:

| plot bin width | bins across ±6 mm | commensurate | plateau pull RMS | 0.20 mm comb |
|---|---|---|---|---|
| **0.05 mm** | 240 (until 2026-08-05) | no | **4.17** | **3.88 %** |
| **0.04 mm** | **300 (now)** | yes | **1.60** | **0.04 %** |
| 0.06 mm | 200 | no | 2.19 | — |
| 0.08 mm | 150 | yes | 2.01 | 0.19 % |
| 0.10 mm | 120 | no | 2.14 | — † |
| 0.12 mm | 100 | yes | 2.28 | — |
| 0.20 mm | 60 | yes | 2.74 | — |

Pull RMS is `(observed − linear baseline)/√observed`; 1.0 is pure Poisson. The
comb column is the amplitude of a 0.20 mm modulation as a fraction of the
plateau.

† The comb estimator projects onto a 0.20 mm sinusoid and is blind at exactly
Nyquist, which is what 0.10 mm bins are for that period. Read the pull RMS for
that row: 0.10 mm is a bad binning, and `is_commensurate(120)` rejects it.

**Only integer multiples of 0.04 mm work.** Sub-multiples are just as bad — with
0.02 mm bins every second bin catches a comb tooth and the one between it
catches none — which is why `is_commensurate` requires a multiple ≥ 1 rather
than any exact divisor of the range.

Coarser commensurate binnings (0.08, 0.12, 0.20 mm) kill the comb but degrade
the σ fit, because the dip is 0.22 mm wide and needs points inside it. 0.04 mm
is both the finest commensurate choice and the best-conditioned one.

## 3. What the fix buys, on both held-out files

Same peaks, same events, same fit function — only the histogram bin width moves.

| file | peak list | pull RMS 240 → 300 | comb 240 → 300 | σ_vtx-vtx 240 | σ_vtx-vtx 300 |
|---|---|---|---|---|---|
| r16443 | efficiency (int ≥ 0.30) | 4.41 → **1.89** | 3.81 % → 0.06 % | 0.2139 ± 0.0044 | 0.2158 ± **0.0028** |
| r16443 | resolution (int ≥ 0.50) | 4.17 → **1.60** | 3.88 % → 0.04 % | 0.2270 ± 0.0040 | 0.2283 ± **0.0022** |
| r16638 | efficiency | 4.60 → **2.02** | 3.93 % → 0.14 % | 0.2151 ± 0.0047 | 0.2167 ± **0.0029** |
| r16638 | resolution | 4.30 → **1.71** | 3.96 % → 0.15 % | 0.2280 ± 0.0042 | 0.2289 ± **0.0022** |

- The comb drops by a factor **25–95**. Paired bootstrap over events (400
  replicas, same events resampled for both binnings): the reduction is
  **3.75 ± 0.14 %** on r16443 (26.5σ) and **3.77 ± 0.12 %** on r16638 (31.8σ);
  the plateau pull RMS falls by **2.41 ± 0.11** and **2.45 ± 0.11** (22σ). The
  bootstrap's own estimate of the *residual* comb at 0.04 mm bins is
  0.13 ± 0.07 %, larger than the 0.04 % point estimate because a modulus is
  positive-definite and resampling noise biases it up — read the point estimate,
  and read the bootstrap for the size of the change.
- The σ_vtx-vtx **fit error nearly halves** (0.0040–0.0047 → 0.0022–0.0029 mm).
  σ itself moves by +0.6 to +0.9 %, inside the old error bar. It is not a
  resolution improvement and must not be quoted as one.
- **The peak list is bit-identical.** The binning touches only how those peaks
  are histogrammed. Efficiency and fake rate move at all only because this eval
  derives its matching window from the fitted σ, and σ shifted by +0.5 %: on
  r16443 the full re-run gives efficiency 0.8643 → 0.8648 and 16.64 → 16.60
  fake/event. The truth-matched position residual is untouched.

### What is left is real

At 0.04 mm bins the far plateau is Poisson, and the residual excess is confined
to where the satellite shoulder lives (r16443, resolution list):

| \|Δz\| window | 1.2–2.0 | 2.0–3.0 | 3.0–4.5 | 4.5–6.0 |
|---|---|---|---|---|
| pull RMS, 240 bins | 4.57 | 4.09 | 3.96 | 3.86 |
| pull RMS, 300 bins | **2.44** | **1.11** | **1.08** | **0.90** |

A binning artefact should be flat across the plateau, and at 240 bins it is —
3.9 to 4.6 everywhere. At 300 bins everything beyond 2 mm is statistical, and
the 2.44 left at 1.2–2.0 mm is the tail of the satellite population, i.e. the
thing
[pairwise_dz_bump](pairwise_dz_bump.md) is about. Fitting a quadratic or cubic
baseline instead of a straight line moves the full-range figure only 1.60 → 1.55,
so it is structure, not curvature.

### Why this also matters for the band excess

`docs/research/pairwise_dz_bump.md` records that the far-plateau median
convention "is not robust to quantised positions" and returns +25.29 % for the
argmax estimator. That is this comb. Measured here on the same peaks, the
far-plateau band excess is 0.31 % at 0.05 mm bins, −0.56 % at 0.04 mm, and
**+16.05 %** at 0.12 mm — the convention is only usable on a commensurate
binning, and the fitted-baseline convention should still be preferred for
estimator comparisons.

## 4. De-quantising the positions instead costs more than it buys

The comb could also be attacked at its source. It was, on the same 7680 events,
with the peak list held **fixed** at the deployed operating point so only the
position estimator moves:

| position estimator | frac-bin mod | \|Δz\| comb | core IQR σ (µm) | efficiency | fake/evt | σ_vtx-vtx |
|---|---|---|---|---|---|---|
| **local centroid ±3 (deployed)** | 135.8 % | 40.5 % | **42.98** | 0.8699 | 16.16 | **0.2140** |
| local centroid ±5 | 41.7 % | 3.5 % | 43.51 | 0.8710 | 16.30 | 0.2151 |
| local centroid ±7 | **7.6 %** | **0.7 %** | 44.00 | 0.8719 | 16.47 | 0.2260 |
| 3-point parabolic | 52.5 % | 4.8 % | **42.72** | 0.8689 | 15.84 | 0.2366 |
| pure argmax | 1000 % | 525 % | 45.33 | 0.8685 | 15.89 | 0.2344 |
| centroid ±3 on a 0.75-bin Gaussian | 152.1 % | 51.7 % | 43.10 | 0.8698 | 16.14 | 0.2178 |
| centroid ±3 on a 1.5-bin Gaussian | 196.1 % | 89.6 % | 43.61 | 0.8698 | 16.07 | 0.2136 |
| parabolic on a 0.75-bin Gaussian | 40.6 % | **2.2 %** | 42.73 | 0.8690 | 15.81 | 0.2369 |

Efficiency and fakes at a fixed 0.2328 mm matching window; half B (3840 events).

- Widening the centroid window de-quantises (±7 takes the comb to 0.7 %) but
  costs 1.0 µm of core width and 0.012 mm of σ_vtx-vtx. Wrong trade.
- Parabolic interpolation gives the best core width by 0.26 µm and the worst
  σ_vtx-vtx of the continuous variants, reproducing what
  [peak_position_estimator](peak_position_estimator.md) already found.
- **Pre-smoothing makes the quantisation worse, not better** (135.8 % → 196.1 %
  at σ = 1.5 bins): a symmetric peak has its centroid closer to its own maximum,
  not further from it. Worth recording, because the opposite is the intuition.

So the estimator stays as it is and the binning is fixed instead. The two are
not equivalent — de-quantising would also change the peak positions themselves —
but nothing here shows any gain from moving them.

## 5. The other half: no post-hoc lever cuts the fake rate

**817 configurations**, selection half A / reporting half B, matching window
fixed. **0 of 817 beat the deployed operating point on either half.** Ranked
table in `ranked.csv`; the picture is `lever_pareto.png`, where every lever's
cloud sits on or below the black integral/height frontier.

Levers tried, each crossed with the integral threshold (0.20–1.00) and the
height floor (0.00–0.08):

| lever | what it does | n |
|---|---|---|
| integral / height | the two knobs already in the eval — the frontier | 24 |
| Gaussian pre-smoothing | `--smooth-sigma` 0.5–2.0 bins before region finding | 144 |
| Gaussian, positions from raw | detect on the smoothed histogram, localise on the raw one | 144 |
| anti-lattice notch | cos(ω), cos²(ω), cos³(ω), and a symmetric 4-bin boxcar | 96 |
| notch, positions from raw | as above | 96 |
| absolute prominence gate | topographic prominence ≥ 0.002–0.040 | 45 |
| relative prominence gate | prominence / height ≥ 0.05–0.60 | 54 |
| NMS | drop a peak within *d* of a taller one if the height ratio is below *r* | 64 |
| minimum separation | of any pair closer than *d*, keep only the taller | 32 |
| peakiness | height / integral ≥ 0.08–0.20 | 30 |
| combinations | prominence × separation | 96 |

Cheapest fake removal each family can manage — lowest efficiency cost per
fake/event among configurations removing ≥ 0.5 fake/event, chosen on A,
reported on B, against a budget of **0.2**:

| lever | cost | Δ eff (pts) | Δ fake/evt | core IQR σ (µm) | σ_vtx-vtx | band excess |
|---|---|---|---|---|---|---|
| *(deployed point)* | — | — | — | 42.98 | 0.2140 | +0.066 |
| **integral / height** | **0.216** | −0.33 | −1.55 | 42.53 | 0.2222 | −0.298 |
| Gaussian σ 0.5 | 0.236 | −0.38 | −1.59 | 42.52 | 0.2311 | −0.328 |
| notch cos(ω) | 0.246 | −0.48 | −1.96 | 44.46 | 0.3116 | −0.612 |
| Gaussian, positions from raw | 0.254 | −0.36 | −1.43 | 42.44 | 0.2311 | −0.208 |
| notch, positions from raw | 0.271 | −1.10 | −4.07 | 42.66 | 0.2762 | +0.346 |
| absolute prominence gate | 0.313 | −1.26 | −4.02 | 41.36 | 0.2434 | +0.130 |
| relative prominence gate | 0.322 | −1.33 | −4.12 | 41.30 | 0.2467 | +0.074 |
| peakiness | 0.422 | −1.65 | −3.91 | 41.21 | 0.2169 | +0.671 |
| NMS | 0.708 | −0.87 | −1.23 | 42.18 | 0.2858 | −0.723 |
| minimum separation | 1.581 | −1.56 | −0.99 | 42.33 | — † | +0.686 |

† the σ_vtx-vtx sigmoid does not converge for that configuration.

Paired bootstrap over events, 2000 replicas, on the score (Δ efficiency points
+ 0.2 × fakes removed) — the same resampling applied to both members of every
difference:

| lever | score | P(score > 0) |
|---|---|---|
| integral 0.40 (thresholds) | **−0.025 ± 0.010** | 0.009 |
| Gaussian σ 0.5 + integral 0.40 | −0.058 ± 0.012 | 0.000 |
| notch cos(ω) + integral 0.40 | −0.090 ± 0.018 | 0.000 |
| prominence ≥ 0.002 | −0.455 ± 0.019 | 0.000 |
| prominence/height ≥ 0.05 | −0.505 ± 0.020 | 0.000 |
| NMS 0.35 / 0.2 | −0.626 ± 0.015 | 0.000 |
| peakiness ≥ 0.08 | −0.869 ± 0.021 | 0.000 |
| minimum separation 0.30 mm | −1.365 ± 0.020 | 0.000 |

**Even the cheapest move available — just raising the integral threshold — is
2.5σ below break-even.** That is what a knee looks like, stated quantitatively.

### Why, in one number

Removing one truth-matched peak per event costs 1/111.2 of the truth
denominator = **0.90 efficiency points**; removing one fake/event is worth
**0.20**. So a post-hoc filter has to remove **≥ 4.5 surplus peaks per real peak
removed** to be worth taking. The height floor manages 4.9:1, which is why it is
in the operating point. A prominence gate manages 1.3–2.0:1, because at this
operating point the low-prominence population it targets has already been
removed: surviving surplus peaks have median prominence/height **0.89**, i.e.
they are standalone low peaks, not ripple on a flank. See
[pairwise_dz_bump §3.2](pairwise_dz_bump.md#32-prominence-gate-on-the-split-declined-recorded-for-completeness).

### Checks against fooling ourselves

- **Selection and reporting halves agree** to within 0.159 efficiency points and
  0.126 fake/event across all 817 configurations.
- **The ranking does not depend on the fixed window.** Re-scored at 0.2140,
  0.2328 and 0.2800 mm: the deployed point is best at all three, and zero levers
  score above it at any of them (`window_robustness.json`).
- **Every lever was verified against a reference** before any result was taken
  from it: the region scanner against `pv_locations_updated_res` (0/40 count
  mismatches, max 7 × 10⁻⁶ mm), NMS against `suppress_neighbor_peaks` (0/40
  events disagreeing), prominence against `scipy.signal.peak_prominences`
  (2/23 528 peaks outside the documented run-boundary convention).

## 6. A separate inconsistency, found while doing this, not fixed

`run_eval_pvf_run3.py` accumulates `pairwise_dz` over **every event it reads**
and applies the μ window only to the summary. On the held-out flat-μ files the
quoted σ_vtx-vtx is therefore a flat-μ number (⟨μ⟩ ≈ 70) sitting next to
efficiency and fake rates measured on μ ∈ [185, 215]: `eval.log` line 1038 reads
`mu in [185,215] -> 1920/25000 events` while the fit above it used all 25000.
Measured size of the effect on r16443 at 240 bins: **0.2190 mm** over all events
against **0.2270 mm** on the μ-window subset, a 3.5 % difference.

This was left alone deliberately. σ is fed back as the matching window, so
changing it moves the headline efficiency, and that is a decision about what the
published number means rather than a bug fix. It should be resolved before the
note quotes σ and efficiency in the same table.

## Reproducing

Every script is kept next to its output in
`outputs/08_05_2026_output/ripple_study/code/`, along with the region scanner and
matcher carried over from the 2026-08-04 study. `outputs/` is gitignored, so
this is the only copy; `README.md` in that directory indexes the results.

```bash
cd outputs/08_05_2026_output/ripple_study/code
export PYTHONPATH=/data/home/matmauro/codice/PV-Finder/src

python -u dump_heldout.py --root <held-out .root> --out h_<tag>.npy   # once
python -u verify_ripple.py       # levers vs their committed references
python -u binning.py             # the binning table
python -u dequant.py             # position estimators vs quantisation
python -u spectrum.py out.json   # lattice spectrum + mod-4 phase
python -u fake_decomp.py         # fakes sitting on nTrk<2 truth vertices
python -u scan_ripple.py --lever <name> --tag r16443 --full --out ...
python -u assemble_ripple.py     # the ranked table
python -u window_robust.py       # ranking vs the fixed matching window
python -u fig_binning.py         # the before/after figure

# the fix, end to end
DEVICE=3 bash scripts/eval_v6_operating_point.sh
```

The 0.04 mm choice is not a magic number: `pairwise_bins()` derives it as
`2 × 6.0 mm / 0.04 mm`, and `is_commensurate()` is what the eval warns on. If
the output grid ever changes, both follow it.
