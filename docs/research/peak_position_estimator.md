# Peak position estimator — local centroid

**Status: measured on held-out data 2026-08-04.** The peak finder's position
estimator was changed from the **full-region weighted mean** to a **local
centroid** over the bins within 3 bins (±0.12 mm) of the region maximum,
clipped to the region.

- Code: `src/pv_finder/utils/peak_finding.py` (canonical),
  `peak_finding_fast.py` (numba), `efficiency_res_optimized_atlas.py` (legacy copy)
- Tests: `tests/test_peak_finding.py`
- Numbers: `outputs/08_04_2026_output/peakfinder_operating_point/`

## Why

Each above-threshold region emits one PV. The historical position was
`sum(v*i)/sum(v)` over the **whole** region. After a
[conjoined split](pairwise_dz_bump.md) the "region" is only one side of an
overlapping pair, so its full-region centroid is dragged away from the peak it
actually represents. The same drag applies, more weakly, to any peak with an
asymmetric tail.

## Sample

7680 held-out events from
`ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root`, μ ∈ [185, 215],
⟨μ⟩ = 192.5, v6 model (`hllhc_alleta_v6_mse_2ep_phase2_epoch_2`). Peak finder at
the production setting `threshold=0.01, integral_threshold=0.2, min_width=3,
min_height=0.0`. Predicted histograms are bit-identical to those of the
production eval (verified against `eval_results.pkl`: peak positions agree to
1e-4 mm on 60/60 spot-checked events).

**The peak set is identical across all variants** — the position estimator does
not touch region detection — and the truth↔reco assignment used for the residual
is derived **once** from the baseline, so the core width compares estimators and
nothing else.

## Variant comparison

Efficiency and surplus use a **fixed** matching window of 0.2328 mm (the
production σ_vtx-vtx). Letting the window float with each variant's own fitted σ
would let a variant buy efficiency by degrading its resolution — see
[the window coupling](#the-window-coupling-caveat).

> **Read the band-excess column with its convention.** It is the *local-baseline*
> band excess defined [below](#two-band-excess-conventions-reconciled) — a
> straight line fitted to the |Δz| density over 1.2–6 mm and extrapolated into
> the band. It is **not** the number
> [pairwise_dz_bump](pairwise_dz_bump.md) quotes, which subtracts a flat plateau
> taken at |Δz| > 3 mm. Both are all-pairs; they differ by a fixed +4.8
> percentage points because the density is not flat. **Quote the bump page's
> number in the note**; this column exists to rank variants within this table.

| | core σ (µm) | IQR σ (µm) | mean bias (µm) | efficiency | Δeff (pts) | surplus/evt | σ_vtx-vtx (mm) | band excess, local baseline (pairs/evt) | satellite ratio |
|---|---|---|---|---|---|---|---|---|---|
| **(a) full-region weighted mean** (baseline) | 42.7 | 46.19 | **+2.9** | 0.8811 | — | 22.21 | 0.2403 | +1.76 | 1.357 |
| (b) local centroid ±2, clipped | 40.7 | 44.43 | +0.0 | 0.8772 | −0.39 | 22.22 | 0.2103 | −0.33 | 1.220 |
| **(b) local centroid ±3, clipped** | **40.7** | **44.41** | **+0.5** | 0.8777 | −0.34 | 22.22 | **0.2091** | +0.01 | 1.242 |
| (b) local centroid ±4, clipped | 41.0 | 44.6 | +1.1 | 0.8784 | −0.27 | 22.20 | 0.2101 | +0.24 | 1.261 |
| (b) local centroid ±5, clipped | 41.3 | 44.80 | +1.6 | 0.8789 | −0.22 | 22.19 | 0.2131 | +0.40 | 1.277 |
| (b) local centroid ±7, clipped | 41.7 | 45.2 | +2.3 | 0.8800 | −0.11 | 22.18 | 0.2273 | +1.19 | 1.324 |
| (b) local centroid ±10, clipped | 42.2 | 45.7 | +2.7 | 0.8809 | −0.02 | 22.18 | 0.2373 | +1.72 | 1.358 |
| (c) local centroid ±2, **unclipped** | 40.5 | 44.5 | −0.5 | 0.8758 | −0.53 | 22.27 | 0.2521 | −0.81 | 1.183 |
| (c) local centroid ±3, **unclipped** | 40.6 | 44.58 | −0.5 | 0.8757 | −0.54 | 22.28 | 0.2629 | −0.92 | 1.175 |
| (c) local centroid ±5, **unclipped** | 42.7 | 46.5 | −0.6 | 0.8750 | −0.61 | 22.25 | 0.3312 | −1.95 | 1.116 |
| (d) 3-point parabolic about the max | **40.3** | **44.28** | −0.4 | 0.8765 | −0.46 | 22.23 | 0.2338 | −0.80 | 1.185 |
| (e) pure argmax bin | 42.7 | 47.05 | −0.4 | 0.8762 | −0.49 | 22.25 | 0.2307 | −0.31 | 1.185 |

## Two band-excess conventions, reconciled

This page and [pairwise_dz_bump](pairwise_dz_bump.md) reported different band
numbers for the same estimator change. Both were run again on the *same* 7680
events and the *same* peak sets:

| | far-plateau (bump page) | | local baseline (this page) | |
|---|---|---|---|---|
| | % of plateau | pairs/evt | % of baseline | pairs/evt |
| full-region weighted mean | **+11.85** | +2.842 | **+7.01** | +1.758 |
| local centroid ±2, clipped | +3.14 | +0.755 | −1.32 | −0.331 |
| **local centroid ±3, clipped** | **+4.48** | +1.079 | **+0.03** | +0.008 |
| local centroid ±4, clipped | +5.58 | +1.341 | +0.97 | +0.245 |
| local centroid ±5, clipped | +6.47 | +1.551 | +1.58 | +0.396 |
| local centroid ±3, unclipped | +0.76 | +0.182 | −3.67 | −0.922 |
| 3-point parabolic | +1.48 | +0.354 | −3.16 | −0.795 |
| pure argmax bin | +25.29 † | +4.918 † | −1.25 | −0.308 |

**There is no bug and no conditioning difference.** Both metrics run over *all*
reco–reco pairs; neither is conditioned on matched peaks (the conditional
surplus-around-matched observable on this page is the separate *satellite ratio*
column). Reproducing the bump page's convention here gives +11.85 % → +4.48 %,
against its published +12.48 % → +5.11 % on 1920 events — agreement within the
sample difference.

The whole gap is the baseline. The |Δz| density is **not flat**, contrary to what
a "plateau" implies:

| \|Δz\| (mm) | 0.3–0.7 | 1.0–1.5 | 2.0–3.0 | 3.0–4.5 | 4.5–6.0 |
|---|---|---|---|---|---|
| pairs/evt/mm | 67.06 | 61.65 | 61.12 | 60.14 | 59.77 |

A straight line fitted over 1.2–6 mm has slope −0.651 pairs/evt/mm and predicts
**62.67** at 0.5 mm, where the median over |Δz| > 3 mm gives **59.95** — 4.5 %
lower. That offset is the 11.85 − 7.01 = 4.84 points between the two columns.
Neither convention is wrong; they must not be mixed, and the note should carry
one. Use the bump page's, because it is the plateau the resolution plot actually
shows — while recording that it is biased high by ~4.5 % from the residual slope.

† **The far-plateau convention is not robust to quantised positions.** Argmax
emits positions on the 0.04 mm bin grid, so |Δz| lands only on multiples of
0.04 mm while the histogram bins are 0.05 mm — a 0.2 mm beat that modulates the
counts. A *median* plateau lands on an arbitrary phase of that comb and returns
+25.29 %, which is an artefact. The fitted-density baseline averages it out
(−1.25 %). This does not affect any continuous-position variant, but it is a
reason not to use the median convention when comparing estimators.

**Core σ** is a Gaussian-core-plus-flat-background fit to the truth-matched
residual `z_reco − z_truth` over ±0.4 mm (2.5 µm bins). **IQR σ** is
`(q75 − q25)/1.349` on the same residuals — no fit, no window, and the estimator
quoted below because it is assumption-free. **Mean bias** is the fitted Gaussian
centre. **Satellite ratio** is the surplus-peak density at 0.3–0.7 mm from a
matched peak, over a flat baseline measured at 2–5 mm.

Paired bootstrap over events (200 replicas; same events, same peaks, so the
difference is far better determined than either absolute value):

| vs baseline | Δ IQR σ |
|---|---|
| local centroid ±2 | **−1.756 ± 0.052 µm (−3.80 %)** |
| local centroid ±3 | **−1.777 ± 0.049 µm (−3.85 %)** |

36σ. This is the decisive metric and it is unambiguous.

## Chosen: clipped, half-width 3 bins

> **One side effect, found 2026-08-05 and fixed in the plot rather than here.**
> The local centroid over a few bins around a sharp maximum lands very close to
> that maximum's own bin, so reconstructed positions become **quantised** on the
> 0.04 mm grid — 135.8 % modulation of the position fractional part, against a
> continuous distribution for the full-region weighted mean it replaced. Nothing
> below changes: the core width, σ_vtx-vtx and band-excess gains are all real.
> But the resolution plot's 0.05 mm bins beat against that comb and acquired a
> visible 3.9 % sawtooth the day this landed. The fix is the plot binning
> (`--pairwise-bins` now 300), not the estimator — de-quantising variants were
> re-measured and all cost more than they buy.
> See [resolution_plot_ripple](resolution_plot_ripple.md).

- Core width statistically tied with the best (±2 and parabolic).
- **Best σ_vtx-vtx of every variant tried: 0.2091 vs 0.2403 mm, −13.0 %**
  (fit errors 0.0052 / 0.0029 mm → ~5σ).
- Removes the baseline's +2.9 µm outward position bias (→ +0.5 µm).
- Band excess in the 0.3–0.7 mm pairwise-Δz window goes to zero:
  +1.76 → +0.01 pairs/event.

### Why not unclipped

Reading past the region boundary was measured, not assumed. It buys the best
band-excess and satellite numbers (−0.92 pairs/evt, ratio 1.175) — and it is
**rejected**, because it pays for them with the thing the note actually quotes:

- σ_vtx-vtx **degrades** to 0.2629 mm, worse than the 0.2403 baseline. Pulling a
  split peak back towards its neighbour compresses genuine close pairs.
- Worst efficiency at a fixed window of every variant (−0.54 pts).
- Its apparent efficiency *gain* under the self-consistent window (0.8892 vs
  0.8845) is entirely the wider window its worse σ buys. That is the circularity
  the fixed window exists to expose.

### Why not parabolic or argmax

Parabolic is marginally the best core (44.28 vs 44.41 µm IQR, a 0.13 µm edge
inside the bootstrap error) but its σ_vtx-vtx is 0.2338 mm against 0.2091, and it
silently degenerates to argmax whenever the 3-point curvature is not concave.

Argmax buys **nothing** on the decisive metric: 47.05 µm IQR, *worse* than the
42.7/46.19 µm baseline. Bin quantisation costs 0.04/√12 = 11.5 µm, which cancels
the estimator gain exactly as expected — a useful internal consistency check.
Argmax does reduce the band excess (−0.31), which is why it looked attractive in
the satellite study; it does so without improving position accuracy at all.

## What the change costs: 0.34 efficiency points, all of it merge-credit

At the fixed window, efficiency falls 0.8811 → 0.8777. Decomposed per event:

| | truth clean | truth merged | truth missed |
|---|---|---|---|
| full-region weighted mean | 84.563 | 13.435 | 13.227 |
| local centroid ±3 | 84.550 | 13.073 | 13.602 |
| **difference** | **−0.013** | **−0.362** | +0.375 |

**Cleanly reconstructed truth vertices are unchanged (−0.013/event, −0.015 %).**
The entire loss is in *merged* — truth vertices that never had a reco of their
own and were counted only because a neighbour's reco window happened to cover
them. A full-region centroid sitting *between* two truth vertices collects that
credit; a centroid sitting on the peak it actually found does not.

So the 0.34 points is not a loss of resolved vertices. It is the withdrawal of
credit for blur. On the reco side the same effect shows as fake 21.58 → 21.04
and split 0.62 → 1.18, with total surplus flat (22.21 → 22.22).

## Skeptical check: does clipping over-separate close pairs?

Clipping truncates the left peak's window at the valley and the right peak's
window after it, which could push a conjoined pair apart and fake a resolution
improvement. Measured — reconstructed minus true separation (µm) for truth pairs
where both vertices won their own reco:

| true separation (mm) | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.7 | 0.7–1.0 | 1.0–1.5 |
|---|---|---|---|---|---|---|---|
| full-region weighted mean | +309.0 | +133.8 | +69.5 | +41.8 | +21.4 | +4.2 | −13.8 |
| local centroid ±3 | **+291.3** | **+118.2** | **+59.0** | **+34.9** | **+19.6** | +6.5 | −9.4 |
| n pairs | 13 312 | 21 912 | 26 997 | 29 103 | 57 462 | 76 913 | 103 750 |

The outward bias is **reduced** at every separation below 0.7 mm, not inflated.
The synthetic double-Gaussian test (`test_conjoined_pair_separation_is_better_
recovered`) shows the same at separations of 8–20 bins against exact truth. The
concern is ruled out.

## The window coupling caveat

`run_eval_pvf_run3.py` feeds the fitted σ_vtx-vtx back in as the matching window.
Because the local centroid improves σ_vtx-vtx by 13 %, the window shrinks by 13 %,
and the *printed* efficiency falls much further than the 0.34 points above:

| | σ_vtx-vtx | window | efficiency | fake/evt |
|---|---|---|---|---|
| full-region weighted mean | 0.2403 | self | 0.8845 | 21.31 |
| local centroid ±3 | 0.2091 | self | 0.8652 | 21.91 |
| local centroid ±3 | 0.2091 | fixed 0.2328 | 0.8777 | 21.04 |

**1.9 of those 1.93 points are the window, not the estimator.** A better position
estimator is penalised by a convention that ties the acceptance window to the
resolution it just improved. Any A/B of position estimators must use a fixed
window; and the efficiency/σ pair quoted in the note should state which window
produced it.

## Backwards compatibility

`pv_locations_updated_res(..., centroid_halfwidth=0)` is the library **default**
and is bit-identical to the historical behaviour. The change is opt-in:

- `run_eval_pvf_run3.py --centroid-halfwidth` defaults to **3** — PV-finder
  evaluation uses the new estimator.
- Everything else — the GNN graph builders in `src/gnn/data/`, all diagnostics,
  `run_eval_pvf.py` — is untouched and still gets the full-region weighted mean.
  This is deliberate: the deployed TTVA checkpoints were trained on graphs whose
  vertex features are these positions and `pv_sigmas`, so moving them is a
  separate decision that needs its own A/B.

`pv_sigmas` is **unchanged by design** — it remains the whole-region weighted
standard deviation about the whole-region weighted mean, independent of
`centroid_halfwidth`, because it is a GNN input feature. It is a region-width
estimate, not the spread of the window the position came from.

## Reproducing

The variant sweep and the operating-point scan share the tooling in
`outputs/08_04_2026_output/peakfinder_operating_point/`. The held-out histogram
dump used there reproduces the production eval's predictions exactly; the region
scanner used for the sweep was verified bit-for-bit against
`pv_locations_updated_res` (positions, heights, peak bins, sigmas) and the
matcher against `compare_res_reco` before any result was taken from it.
