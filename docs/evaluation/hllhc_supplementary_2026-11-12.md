# HL-LHC supplementary studies with the 12 Nov 2026 side ntuple

Status: done 12 Nov 2026, updated the same evening for the >= 2-track
headline rule. Frozen HL-LHC working point: s8, GNN v2 e527, t = 0.97.
Sample: r16443 entries 0-24,999.

Full tables, definitions, gates and the figure list (main/backup) are in
`outputs/11_12_2026_output/hllhc_final/supplementary/README.md`.

Code:

- `scripts/hllhc_final_supplementary.py`: compute pass (identity, per-event
  table, both track rules).
- `scripts/hllhc_final_supplementary_plots.py`: tables and figures.
- `src/gnn/evaluation/weighted_classification.py`: weighted
  Clean/Merged/Split/Fake, tested in `tests/test_weighted_classification.py`.

## Track-multiplicity rule

The headline applies the same rule to both algorithms: a reconstructed vertex
counts only if it has >= 2 tracks. For AMVF that means `assocTracks` (fit
tracks in the weighted cross-check); for PV-Finder, its GNN-assigned tracks at
t. The >= 1-track (drop-empty) numbers are kept as labelled backup. HS
selection and residuals under the rule come from
`outputs/11_12_2026_output/hllhc_final/chain/metrics/test_s8_v2e527_hsmin2.npz`
(`chain_t_metrics --hs-min-tracks 2`).

## The side ntuple

Trey's `/share/lazy/pv-finder/Paper_2026Nov12/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root`
holds the same 100,000 events as the old r16443 file. It is a side table, not
a replacement: it lacks `RecoTrack_x0/y0`, `Time`, `chisq`, `BeamPosSigmaX/Y`
and `Kernel*`.

On entries 0-24,999 these are identical between the two files:

- every track branch the chain uses;
- the AMVF vertices and their `RecoVertex_assocTracks`;
- mu;
- the truth vertices with nTracks >= 2, as a set;
- the hard scatter.

The old truth list is exactly the new one restricted to nTracks >= 1. The new
file adds 30.5 truth vertices per event with no selected track (100.5 vs
70.0 per event). The new file orders truth vertices differently, so compare
truth lists as sets. Pair them by (z, x, y): two distinct vertices can share a
float32 z.

`RecoTrack_ttvaVertex` is the inverse of `RecoVertex_assocTracks`. The new
`RecoVertex_fitTracks` and `RecoVertex_fitWeights` are the AMVF fit content.

- At PU200, 13% of fit (vertex, track) pairs are not in the TTVA list (5% of
  the weight).
- 23% of TTVA pairs are not in the fit.
- 15% of fit tracks are shared between vertices.
- The weight distribution peaks at 0.5.

## Results

**HS vs local density, ITk definition, >= 2-track rule.** The density counts
all truth vertices within +-2 mm of the HS, HS excluded, divided by 4 mm.

- HS reconstruction at 0.1 mm is saturated (>= 99.3%) for both algorithms.
- HS selection with the identical Athena rule: PV-Finder + GNN is above the
  AMVF rule by +2.7 to +5.0 pt in every all-mu bin.
- HS resolution: PV-Finder + GNN is 0.2-0.9 um narrower in every bin, flat
  in density at about 9.5-10 um.
- The rule changes PV-Finder selection by 0 to +0.19 pt and leaves AMVF
  unchanged.
- The old stored-truth density gives the same picture, with densities 44%
  lower.
- One PU200 bin (ITk density 1.5-2) shows no selection lead: +0.6 ± 1.4 pt.
  This is flagged as a probable fluctuation.

**Vertex classes at PU200** (vertices per event, 1,920 events).

| | Clean | Merged | Split | Fake |
|---|---|---|---|---|
| PV-Finder + GNN, >= 2 tracks, unweighted (primary) | 60.00 | 12.76 | 1.97 | 0.03 |
| AMVF, >= 2 tracks, TTVA links, unweighted (primary) | 34.42 | 41.72 | 21.65 | 0.07 |
| PV-Finder + GNN, >= 2 tracks, GNN-score weights | 60.03 | 13.24 | 1.46 | 0.03 |
| AMVF, >= 2 fit tracks, fit weights | 46.26 | 31.98 | 19.56 | 0.10 |
| PV-Finder + GNN, >= 1 track, unweighted (backup) | 74.28 | 12.73 | 6.81 | 0.17 |
| AMVF, >= 1 track, TTVA links, unweighted (backup) | 34.44 | 41.72 | 21.66 | 0.07 |

The >= 2-track rule drops 19.2 PV-Finder vertices per event at PU200 (20%):
14.3 Clean, 4.8 Split and 0.14 Fake. AMVF loses 0.03, because it has almost
no 1-track vertices. The Clean ratio becomes 1.74x (was 2.16x).

Classifying AMVF with its own fit weights raises its Clean count by +11.8 per
event (+34%) at PU200 under either rule. About 60% of that comes from the
different track list. With weights on both sides the Clean ratio is 1.30x
(+13.8 per event).
