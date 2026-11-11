# Hard-scatter longitudinal resolution, ITk-paper definition (9 November 2026)

On the note's 1,920 PU200 events the hard-scatter (HS) longitudinal position
resolution is **12.38 ± 0.35 µm for PV-Finder** and **10.24 ± 0.28 µm for
AMVF** (difference 2.14 ± 0.37 µm, paired event bootstrap). AMVF reproduces
the "about 10 µm" of ITk Figure 28. PV-Finder is about 21 % wider on the HS,
although its all-vertex 68 % position residual is the same as AMVF's
(55.05 vs 56.45 µm).

## Definition

ITk paper, arXiv:2412.15090, §5.5 and Figure 28: "Longitudinal position
resolution of the reconstructed primary vertex, evaluated in tt̄ events with
⟨μ⟩ = 200", plotted against the local pile-up density in vertices/mm. The
paper does not state the estimator. The Athena monitoring that makes this plot
(`InDetPerfPlot_VertexTruthMatching`, `resolution_vs_PUdensity_hsVxTruthLong`)
does the following, and we follow it:

- residual between the true HS z and the reconstructed vertex matched to it,
  filled only when the HS is reconstructed;
- local PU density = other truth vertices within ±2 mm of the HS / 4 mm
  (`getLocalPUDensity`, radial window 2 mm);
- per bin, width = iterative RMS: restrict to mean ± 3 RMS and recompute until
  the selection stops changing (`ResolutionHelper::iterRMS_convergence`, the
  default; at most 100 iterations); error RMS/√(2N).

Our differences, all forced by the ntuples:

- **HS match.** Athena uses a truth-weight match (`bestHardScatterMatch`); the
  ntuples have no vertex-fit truth weights. We take the reconstructed vertex
  closest in z to the true HS within 0.1 mm, the ITk Fig. 26(b) reconstruction
  criterion. For AMVF we also do a track match, the AMVF vertex sharing the
  most tracks with the HS truth vertex: 99.6 % of events pick the same vertex
  and the width changes by 0.1 µm.
- **Density.** Only stored truth vertices (≥ 1 selected track) are counted, as
  in [hs_reconstruction_2026-09-17.md](hs_reconstruction_2026-09-17.md); the
  axis is labelled "stored-truth PU density".
- **Unbinned.** The iteration runs on the residuals, not on a histogram whose
  binning the paper does not give. Starting ranges of 0.1, 0.5 and 1 mm and HS
  radii of 0.1, 0.5 and 1 mm give identical widths to 0.01 µm.

Implementation: `src/pv_finder/evaluation/vertex_finding/itk_resolution.py`
(tests in `tests/test_itk_resolution.py`), driven by
`scripts/itk_hs_resolution.py` and `scripts/itk_hs_resolution_plots.py`.
PV-Finder positions are the production peaks of the note's comparison
(integral 0.30, height 0.03, local centroid). AMVF uses vertices with
nTrk ≥ 2. No inference.

## Results

PU200 (185 ≤ μ ≤ 215, 1,920 events, r16443):

| | iterative RMS [µm] | stat. (RMS/√2N) | bootstrap | events in core / HS reconstructed | median abs [µm] |
|---|---|---|---|---|---|
| PV-Finder | 12.38 | 0.20 | 0.35 | 1,873 / 1,918 | 8.15 |
| AMVF (closest) | 10.24 | 0.17 | 0.28 | 1,866 / 1,916 | 6.34 |
| AMVF (track match) | 10.15 | 0.17 | 0.29 | 1,859 / 1,913 | 6.33 |

All 25,000 flat-μ events: the width is flat in density and μ for both, with
PV-Finder/AMVF between 1.15 and 1.37 in every bin.

| stored-truth density [mm⁻¹] | 0-0.5 | 0.5-1 | 1-1.5 | 1.5-2 | 2-4 |
|---|---|---|---|---|---|
| PV-Finder [µm] | 12.38 ± 0.07 | 12.53 ± 0.11 | 12.82 ± 0.18 | 14.12 ± 0.40 | 13.97 ± 0.83 |
| AMVF [µm] | 9.78 ± 0.06 | 10.12 ± 0.09 | 10.09 ± 0.14 | 10.26 ± 0.29 | 12.09 ± 0.72 |

Versus HS truth n_trk (PU200) AMVF improves from 16.1 µm (20-30 tracks) to
7.5 µm (≥ 80), PV-Finder from 18.1 to 9.8 µm; PV-Finder's improvement is
weaker above 40 tracks (12.0, 13.0, 11.2, 9.8 µm vs 10.8, 9.5, 8.4, 7.5 µm).

Figures (PDF and PNG) in `outputs/11_09_2026_output/itk_hs_resolution/`:
`hs_resolution_vs_density`, `hs_resolution_vs_mu`,
`hs_resolution_vs_ntrk_pu200`, `hs_residuals_pu200`. Numbers in
`results.json`; per-event residuals in `events.npz`.

## Interpretation (not established)

The PV-Finder output histogram has 40 µm bins, four times the AMVF HS
resolution; a uniform position error over one bin alone has an RMS of
40/√12 = 11.5 µm. The local-centroid positions are not hard-quantised (the
fractional position within a bin spans the whole bin), but their distribution
is not flat: 2,759 HS positions in the 10 % of the bin next to one edge
against 2,128 in the bin centre, while AMVF is flat (2,393-2,551). Together
with the flattening of PV-Finder's width above 40 tracks, this suggests the
output granularity and position estimator, not the track information, limit
the HS precision. A finer output grid or a fitted peak position would test it.
The all-vertex 68 % residual (55 µm) is dominated by low-multiplicity vertices
whose precision is far above the bin width, which is why it does not show the
difference.

## Draft LaTeX for §4.1.1

```latex
We also measure the longitudinal resolution of the hard-scatter vertex with
the definition of Ref.~\cite{IDTR-2023-01} (Figure~28 there). For each event
with a reconstructed vertex within $0.1$~mm of the true HS position, we take
the difference $z_\mathrm{reco}-z_\mathrm{HS}$ for the closest such vertex.
The resolution is the iterative RMS of these differences: the RMS is
recomputed within three RMS of the mean until the selection no longer
changes, as in the ATLAS validation software. On the $1\,920$ PU200 events,
the resolution is $12.4\pm0.4~\mu$m for \pvf{} and $10.2\pm0.3~\mu$m for
AMVF. The uncertainties come from resampling the events. Matching the AMVF
vertex by its shared tracks instead of its position gives $10.2~\mu$m. Both
resolutions are nearly independent of the local pile-up density and of
$\mu$ (Figure~\ref{fig:hs_resolution}). The AMVF value agrees with the
approximately $10~\mu$m in Ref.~\cite{IDTR-2023-01}. The \pvf{} position is
less precise for the hard-scatter vertex, although the $68\%$ position error
over all vertices is similar for the two methods. The \pvf{} output uses
$40~\mu$m bins, and its resolution improves less than that of AMVF for
hard-scatter vertices with many tracks. This suggests that the output
granularity limits the precision for such vertices, but we have not tested
this directly.
```

## 2026-11-11: position estimators

Can the PV-Finder HS resolution be improved without retraining? Same
definition as above (closest vertex within 0.1 mm, iterative 3-RMS from
+-0.1 mm), on the **1,915 events where both PV-Finder (1,918) and AMVF
(1,916) reconstruct the HS**, of the 1,920 PU200 events. Errors: paired event
bootstrap, 2,000 resamples (seed 20261111). No inference: saved histograms and
saved top-1 GNN edges of the note's checkpoint (v4 noaug e107).
Code: `scripts/hs_position_estimators.py` (estimators in
`pv_finder.evaluation.vertex_finding.hs_position_estimators`, tests in
`tests/test_hs_position_estimators.py`); outputs in
`outputs/11_11_2026_output/hs_position_estimators/`.

**Headline (post-GNN measurement).** The PV-Finder + GNN HS position is the
inverse-variance mean of the z0 of the tracks the GNN assigns to the HS peak
at the production cut (score >= 0.98). Tracks more than 3 sigma(z0) from the
histogram peak are removed first, then iteratively from the refitted mean
(at most 5 passes). Cost: 50 us per vertex in Python.

| Estimator (1,915 events) | Resolution [um] | minus AMVF | minus centroid |
|---|---|---|---|
| AMVF | 10.25 +- 0.30 | | |
| PV-Finder + GNN, track refit (t=0.98) | **12.02 +- 0.38** | +1.77 +- 0.36 | -0.36 +- 0.34 |
| same, t=0.95 | 11.78 +- 0.36 | +1.54 +- 0.35 | -0.60 +- 0.33 |
| same, all assigned tracks (t=0) | 11.61 +- 0.35 | +1.36 +- 0.34 | -0.77 +- 0.32 |
| same + d0 significance < 3, t=0.98 | 12.05 +- 0.34 | +1.80 +- 0.31 | -0.33 +- 0.31 |
| PV-Finder histogram, local centroid (production) | 12.38 +- 0.35 | +2.13 +- 0.36 | |
| histogram, parabola on log heights | 12.00 +- 0.32 | +1.76 +- 0.34 | -0.38 +- 0.16 |
| histogram, parabola on heights | 12.85 +- 0.28 | +2.60 +- 0.32 | +0.47 +- 0.20 |
| Refit of the true HS tracks (ideal association) | 11.18 +- 0.38 | +0.93 +- 0.37 | -1.20 +- 0.35 |
| Training target read out with the centroid | 0.77 +- 0.06 | | |
| Training target, parabola on log heights | 0.58 +- 0.02 | | |

Diagnostics:

- **The 0.04 mm bins are not the limit.** The training target of each event,
  read out with the same peak finder, places the HS to 0.6-0.8 um. The
  encoding carries the sub-bin position; the network output does not
  reproduce it to better than about 12 um. Finer output bins alone are not
  expected to help. This replaces the bin-width interpretation above.
- **An unseeded refit fails.** The plain weighted mean gives 30 um (t=0.98)
  and 48 um with the true HS tracks: 15% of true HS tracks have
  |z0 pull| > 3, mostly secondaries. Seeding the outlier rejection at the
  histogram peak is what makes the refit work.
- **Association is not the main gap either.** Even the true HS tracks give
  11.18 um with this simple refit, 0.9 um above AMVF. The remaining gap is
  the fit itself (AMVF's adaptive multi-vertex fit with full track
  parameters), not which tracks are used.
- Versus HS nTrk, the refit gains most at 20-30 and 60-80 tracks
  (18.0 -> 15.0 and 11.2 -> 10.1 um) and stays about 1.2 x AMVF above 50
  tracks (`hs_resolution_vs_ntrk_estimators.png`).
- Mean residuals are below 0.8 um for every working estimator (no bias).
- Refit fallbacks to the centroid (no assigned track): 2 events.

### Adaptive vertex fit (added later on 2026-11-11)

`adaptive_beamline_fit` (`hs_position_estimators.py`): an AMVF-style adaptive
fit of (x, y, z) with a beam-spot constraint (ntuple `BeamPos*`, sigma_xy
12 um). Tracks enter with their (d0, z0) perigee covariance, linearised along a
straight line: d0 = -x sin(phi) + y cos(phi), z0 = z - (x cos(phi) +
y sin(phi)) cot(theta). Weights 1 / (1 + exp((chi2 - 9) / 2T)), annealing
T = 64, 16, 4, 2, 1.5, 1, seeded at the PV-Finder histogram peak. The 1D
version (`adaptive_z_fit`, z0 only) is kept for comparison.

**Why 3D matters.** ITk tracks reach |eta| = 4 (median |cot theta| 3.2,
90th percentile 18), so a 10-15 um transverse offset of the vertex from the
beam axis moves z0 by tens to hundreds of um. With the true HS (x, y) the
1D fit of the true HS tracks improves from 12.33 to 10.43 um (384 events).
The d0 sign was measured on the ntuple (regression of d0 on the true-vertex
prediction: slope -0.86 for the opposite convention). The `RecoTrack_x0/y0`
branches follow the opposite sign and are not used.

Common set (1,915 events), iterative 3-RMS, 2,000 paired bootstraps:

| Estimator | Resolution [um] | minus AMVF | minus simple refit (12.02) |
|---|---:|---:|---:|
| AMVF (stored vertex) | 10.25 +- 0.30 | | |
| 1D adaptive, GNN t >= 0.98 | 11.99 +- 0.35 | +1.74 +- 0.35 | -0.03 +- 0.21 |
| 1D adaptive, true HS tracks | 11.23 +- 0.31 | +0.98 +- 0.33 | -0.79 +- 0.30 |
| **3D adaptive, all GNN-assigned tracks** | **9.27 +- 0.25** | **-0.98 +- 0.21** | -2.75 +- 0.35 |
| 3D adaptive, GNN score as prior | 9.28 +- 0.24 | -0.96 +- 0.21 | -2.73 +- 0.34 |
| 3D adaptive, GNN t >= 0.95 | 9.42 +- 0.25 | -0.83 +- 0.21 | -2.60 +- 0.34 |
| 3D adaptive, GNN t >= 0.98 | 9.72 +- 0.26 | -0.53 +- 0.23 | -2.30 +- 0.33 |
| 3D adaptive, all tracks within 1 mm (no GNN) | 9.28 +- 0.28 | -0.97 +- 0.23 | -2.74 +- 0.36 |
| 3D adaptive, AMVF's own HS track list | 9.50 +- 0.25 | -0.75 +- 0.19 | -2.52 +- 0.35 |
| 3D adaptive, true HS tracks | 9.09 +- 0.23 | -1.16 +- 0.23 | -2.93 +- 0.34 |

Versus HS truth nTrk (um; bins 2-20, 20-30, 30-40, 40-50, 50-60, 60-80, 80-130):

| | 2-20 | 20-30 | 30-40 | 40-50 | 50-60 | 60-80 | 80-130 |
|---|---|---|---|---|---|---|---|
| histogram peak | 10.3 | 18.0 | 15.4 | 12.0 | 13.0 | 11.2 | 9.8 |
| 3D adaptive, all GNN tracks | 13.6 | 13.5 | 11.2 | 9.4 | 9.0 | 8.2 | 6.7 |
| AMVF | 12.3 | 16.1 | 12.8 | 10.8 | 9.5 | 8.4 | 7.5 |

HS track lists (GNN t >= 0.98 vs AMVF closest vertex, summed over events):
76,472 shared, 9,686 GNN only, 11,950 AMVF only; true HS tracks on the list
78,772 (GNN) and 78,118 (AMVF) of 105,448.

**Reading.** The remaining gap after the 1D refit was the fit model, not the
association: on the same tracks, the 3D fit beats AMVF's stored position
(9.50 vs 10.25 um), and GNN-assigned tracks do as well as true HS tracks to
within 0.2 um. The adaptive weights do most of the track cleaning (all tracks
within 1 mm give 9.28 um), so the GNN score cut matters little for the
position; looser cuts help because they keep more HS tracks. The fit costs
about 1.8 ms per event in numpy (60 tracks, 6 temperatures); a compiled
version is negligible next to inference. Why AMVF's stored z is less precise
than a single-vertex fit of its own tracks is not established (track sharing
between vertices in the multi-vertex fit is a candidate).
