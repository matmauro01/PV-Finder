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
