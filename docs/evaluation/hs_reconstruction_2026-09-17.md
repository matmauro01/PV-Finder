# Hard-scatter reconstruction efficiency, 17 September 2026

PV-Finder and AMVF both find the hard-scatter (HS) position in about 99.9% of
the evaluated HL-LHC events. This is a different measurement from the roughly
80% efficiency for all eligible truth vertices in the note.

## Definition and sources

[Section 5.5 of the ITk paper](https://arxiv.org/html/2412.15090v2#S5.SS5)
describes Figure 26(b) as the fraction of events with a reconstructed vertex
within 0.1 mm in z of the true HS position. We implement that description:

\[
\epsilon_{\rm HS}=
\frac{1}{N_{\rm events}}\sum_e
\mathbf{1}\!\left[\min_{r\in R_e}|z_r-z_{{\rm HS},e}|<0.1\;{\rm mm}\right].
\]

An event with no reconstructed vertices fails. We do not require the matched
vertex to be the leading vertex in sum(pT²), nor require exclusive matching
against other truth vertices. This measures finding the HS position. Selecting
the HS among the reconstructed vertices is a separate task, treated in Figure
27 of the ITk paper and in our GNN studies. No GNN is used here.

The HS is identified by `TruthVertex_type == 1`. No cut on its track count is
added. PV-Finder uses the saved production peaks: amplitude threshold 0.01,
integral threshold 0.30, minimum height 0.03, local centroid within three bins
of the maximum. AMVF uses the stored vertices with at least two associated
selected tracks, as in the main note. Both use the same 0.1 mm cut and events.
These settings were fixed before this study. In particular, the cut is not
the note's 0.1694 mm all-vertex window or its vertex-pair resolution.

There is a qualification to an exact comparison with the published ITk figure.
The Athena monitoring implementation in
[release 24.0.50](https://gitlab.cern.ch/atlas/athena/-/blob/release/24.0.50/InnerDetector/InDetValidation/InDetPhysValMonitoring/src/InDetPerfPlot_VertexTruthMatching.cxx)
and the current main branch uses `bestHardScatterMatch`, based on truth
matching, and a three-dimensional distance for the HS reconstruction cut.
It also has early returns for missing matching information. Those details are
more specific than the paper's description. We have not established which
revision and configuration produced Figure 26(b). Our result follows the
published positional description; it is not a verified reproduction of that
monitoring configuration. The ntuples do not contain the vertex-fit truth
weights needed to reproduce that matching exactly.

## Main result

The main comparison uses the same 1,920 events and production peaks as the
note's PU200 result. The selection is 185 <= mu <= 215; its actual range is
185 to 200, with mean 192.5. These events come from the separate flat-mu
validation sample. All 1,920 have exactly one stored HS vertex. Their HS
track counts range from 13 to 129.

| Method | Events passing | Efficiency | 68.27% interval |
|---|---:|---:|---:|
| PV-Finder | 1,918 / 1,920 | 99.8958% | 99.7586% to 99.9631% |
| AMVF | 1,916 / 1,920 | 99.7917% | 99.6272% to 99.8913% |

The intervals are equal-tailed Clopper-Pearson binomial intervals. They retain
a nonzero uncertainty when every event in a bin passes. Events are unweighted;
the available ntuple has no event-weight branch.

The paired event counts are: 1,915 pass both methods, three pass only
PV-Finder, one passes only AMVF, and one fails both. The difference is
0.1042 percentage points with a paired standard error of 0.1042 points.
An exact paired test (McNemar, two-sided) gives p = 0.625. These data do not
show a significant difference between the methods. The all-vertex efficiency
advantage does not establish an HS efficiency advantage.

Across all 25,000 validation events, PV-Finder gives 24,971 / 25,000 = 99.884%
and AMVF gives 24,966 / 25,000 = 99.864%. This wider sample mixes pile-up
conditions and is not an additional independent PU200 result. All 25,000 have
exactly one stored HS vertex; none were removed by an HS multiplicity cut.

## Plots and the density limitation

- [Efficiency versus stored truth density, PDF](../../outputs/09_17_2026_output/hs_reconstruction/r16443/hs_efficiency_vs_density.pdf)
- [Efficiency versus stored truth density, PNG](../../outputs/09_17_2026_output/hs_reconstruction/r16443/hs_efficiency_vs_density.png)
- [Efficiency versus mu, PDF](../../outputs/09_17_2026_output/hs_reconstruction/r16443/hs_efficiency_vs_mu.pdf)
- [Efficiency versus mu, PNG](../../outputs/09_17_2026_output/hs_reconstruction/r16443/hs_efficiency_vs_mu.png)
- [Full counts, intervals and checks](../../outputs/09_17_2026_output/hs_reconstruction/r16443/results.json)
- [Per-event measurements](../../outputs/09_17_2026_output/hs_reconstruction/r16443/events.csv)

The density figure follows the form of ITk Figure 26(b), with both algorithms
on the same axes. Its horizontal coordinate is explicitly **stored truth
pile-up density**:

\[
\rho_{\rm stored}=\frac{
N(\text{other stored truth vertices with }|z-z_{\rm HS}|<2\;{\rm mm})
}{4\;{\rm mm}}.
\]

The HS itself is excluded. All stored truth vertices are counted, including
those with one track. The 25,000 events contain 1,748,948 stored truth vertices;
none have zero selected tracks. The mean stored count in the PU200 subset is
134.45 per event, while mean mu is 192.5. The positions of interactions without
selected tracks are not available. Thus this axis cannot reproduce the full
interaction density defined in the ITk paper. Rescaling it by the ratio of
mean counts would not recover each event's local density.

The upper density bins are combined because only eight events have density
from 2.5 to 4 vertices/mm. Counts for both the plotted bins and the original
finer bins are saved in the JSON. Every PU200 event is included in the plot.
Bin boundaries are left-inclusive and right-exclusive, except for the final
upper edge. Horizontal bars show the bin widths.

| Stored density [vertices/mm] | Events | PV-Finder passing | AMVF passing |
|---|---:|---:|---:|
| 0 to 0.5 | 526 | 526 | 525 |
| 0.5 to 1 | 665 | 664 | 665 |
| 1 to 1.5 | 490 | 489 | 488 |
| 1.5 to 2 | 182 | 182 | 182 |
| 2 to 4 | 57 | 57 | 56 |

The mu plot uses all 25,000 events. It avoids the missing local-density
information and shows the comparison across the full validation range.

## Checks and limits

The ROOT and saved evaluation outputs were aligned using their explicit
entry indices. Truth positions, truth and AMVF counts, prediction counts,
pile-up values and the coordinate origin were checked for every input event.
The reconstructed positions and event list in the PU200 subset exactly match
the existing main-comparison cache for both methods. The beam z is zero.
The result also agrees with a separate direct calculation from that cache.
Input hashes, model checkpoint and file paths are recorded in the JSON.

Removing the AMVF track-count cut leaves its PU200 result at 1,916 / 1,920.
Checks at 0.05, 0.1694 and 0.5 mm are saved, without changing the primary
0.1 mm result. Failed event indices and residuals are saved so they can be
inspected without running inference again.

The alternative reconstruction (`r16638`) gives 1,886 / 1,888 = 99.8941% for
PV-Finder and 1,887 / 1,888 = 99.9470% for AMVF in its PU200 subset. It shares
generated and simulated events with the main sample, so it is a consistency
check and must not be combined as independent statistics. One event in its
full 25,000-entry range, entry 20222, has no stored HS vertex. The script
initially stopped on that condition. The explicitly requested secondary run
omits and reports that entry; it is outside the PU200 subset. Secondary
flat-mu numbers therefore use 24,999 events with a unique stored HS.

Five focused tests cover empty reconstruction, the positional match without
exclusive assignment, the 0.1 mm boundary, density counting, bin endpoints
and nonzero uncertainty for bins with perfect efficiency. All pass. Ruff
checks pass. Both main plots were inspected.

We can confidently report the positional efficiency of both algorithms on
these events. Before claiming an exact reproduction of the ITk figure, we
need its original plotting revision/configuration and the full truth
interaction positions. The paper and presentation have not been edited.

## Reproduction

From the PV-Finder repository root:

```bash
venv/bin/python -u src/pv_finder/diagnostics/hs_reconstruction.py \
  --cache outputs/08_05_2026_output/amvf_fair_comparison/cache_r16443.npz \
  --outdir outputs/09_17_2026_output/hs_reconstruction/r16443

venv/bin/python -u src/pv_finder/diagnostics/hs_reconstruction.py \
  --cache outputs/08_05_2026_output/amvf_fair_comparison/cache_r16638.npz \
  --outdir outputs/09_17_2026_output/hs_reconstruction/r16638 \
  --allow-missing-hs

venv/bin/python -m pytest -q tests/test_hs_reconstruction.py
```

Add `--reuse-events` to regenerate summaries and figures from the compact
per-event file. The main sample requires no missing-HS exception. ROOT paths
and saved prediction paths come from the existing comparison cache. No model
training or inference is repeated.
