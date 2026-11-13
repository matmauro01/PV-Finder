# HL-LHC supplementary studies with the 12 Nov 2026 side ntuple

Status: done 12 Nov 2026. Frozen HL-LHC working point: s8, GNN v2 e527,
t = 0.97. Sample: r16443 entries 0-24,999.

Full tables, definitions, gates and the figure list are in
`outputs/11_12_2026_output/hllhc_final/supplementary/README.md`.

Code:

- `scripts/hllhc_final_supplementary.py`: compute pass (identity, per-event
  table).
- `scripts/hllhc_final_supplementary_plots.py`: tables and figures.
- `src/gnn/evaluation/weighted_classification.py`: weighted
  Clean/Merged/Split/Fake, tested in `tests/test_weighted_classification.py`.

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

**HS vs local density, ITk definition.** The density counts all truth
vertices within +-2 mm of the HS, HS excluded, divided by 4 mm.

- HS reconstruction at 0.1 mm is saturated (>= 99.3%) for both algorithms.
- HS selection with the identical Athena rule: PV-Finder + GNN is above the
  AMVF rule by +2.7 to +5.0 pt in every all-mu bin.
- HS resolution: PV-Finder + GNN is 0.2-0.9 um narrower in every bin, flat
  in density at about 9.5-10 um.
- The old stored-truth density gives the same picture, with densities 44%
  lower.
- One PU200 bin (ITk density 1.5-2) shows no selection lead: +0.6 ± 1.4 pt.
  This is flagged as a probable fluctuation.

**Vertex classes at PU200** (vertices per event, 1,920 events).

| | Clean | Merged | Split | Fake |
|---|---|---|---|---|
| PV-Finder + GNN, unweighted (primary) | 74.28 | 12.73 | 6.81 | 0.17 |
| AMVF, TTVA links, unweighted (primary) | 34.44 | 41.72 | 21.66 | 0.07 |
| PV-Finder + GNN, GNN-score weights | 74.34 | 13.22 | 6.25 | 0.17 |
| AMVF, fit tracks + fit weights | 46.26 | 31.98 | 19.56 | 0.10 |

Classifying AMVF with its own fit weights raises its Clean count by +11.8
per event (+34%) at PU200 and +6.6 (+28%) inclusive. About 60% of that comes
from the different track list and the rest from the weighting. PV-Finder's
Clean lead shrinks from 2.16x to 1.61x and stays at +28 Clean per event.
