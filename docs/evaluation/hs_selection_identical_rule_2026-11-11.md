# HS vertex selection with one identical rule, 11 November 2026

Goal (Mattia): AMVF and PV-Finder+GNN must pick their hard-scatter candidate
with **the same rule**, and that rule must be the real ATLAS one. The 15 GeV pT
cap used on 9 November was only a diagnostic. It is not the ATLAS rule and is
not used here.

## The ATLAS rule (release 24.0, read from source)

Source: BNL LXR mirror of the Athena nightly clones, branch `24.0` (indexed
2026-11-09). The `25.0` branch is not on the mirror; `head` has the same code
in the files below.

1. **Default primary vertexing** is ACTS Gaussian AMVF:
   `Tracking/TrkConfig/python/VertexFindingFlags.py:45-59` (`VertexSetup.ActsGaussAMVF`
   unless heavy-ion / low-mu / special passes). For Run 4:
   `Tracking/Acts/ActsConfig/python/ActsPriVxFinderConfig.py:36-38` (`minWeight=0.02`,
   `maxIterations=200`).
2. **Track links.** `Tracking/Acts/ActsVertexReconstruction/src/AdaptiveMultiPriVtxFinderTool.cxx:326-348`.
   With `useFastCompatibility=true` (default, `.h:138`), a track is linked to a
   vertex unless its `vertexCompatibility > maxVertexChi2 = 18.42` (`.h:136`).
   A track can be linked to several vertices.
3. **Sorting.** `InDetPriVxFinder` runs with `doVertexSorting=True`
   (`InnerDetector/InDetConfig/python/InDetPriVxFinderConfig.py:10-22`) and the
   default `SumPt2Sorting` (`VertexFindingFlags.py:64`;
   `Tracking/TrkConfig/python/TrkVertexToolsConfig.py:7-20`). The tool
   re-sorts the finder output:
   `InnerDetector/InDetRecAlgs/InDetPriVxFinder/src/InDetPriVxFinder.cxx:121-122`.
4. **Sort key.** `Tracking/TrkVertexFitter/TrkVertexWeightCalculators/src/SumPtVertexWeightCalculator.cxx:36-64`
   with `DoSumPt2Selection=True` (`Tracking/TrkConfig/python/TrkVertexWeightCalculatorsConfig.py:6-12`):
   - sums over **all** `trackParticleLinks()` of the vertex;
   - adds `(|1/(q/p)| sin(theta) / GeV)^2`, from each track's own perigee;
   - no track weight, no pT cap and no quality cut.
5. **PriVtx.** `Tracking/TrkVertexFitter/TrkVertexTools/src/VertexCollectionSortingTool.cxx:85-114`:
   sort by descending score (`Vertex_pair::operator<` is `first > other.first`,
   line 19), then the first vertex is `PriVtx` and the rest `PileUp`.
   The finder's own `sumPt2 x sqrt(nTracks)` ranking (`AdaptiveMultiPriVtxFinderTool.cxx:404-431`)
   is overwritten by this sort.
6. **HS match** (ITk Fig. 27, Athena validation). `InnerDetector/InDetValidation/InDetTruthVertexValidation/Root/InDetVertexTruthMatchUtils.cxx:20-38`
   (`bestHardScatterMatch`): the vertex with the most HS tracks, with a strict
   `>`, so ties go to the first vertex in the sorted container.

**Rule.** score(v) = the sum of pT^2 over the tracks linked to v, where a
track is linked if it is assigned to v and compatible with it
(chi2 <= 18.42). The selected vertex is the argmax. HS-match ties go to the
earlier vertex in score order.

Implementation: `src/gnn/evaluation/pv_selection.py`
(`track_vertex_chi2`, `sumpt2_score`, `select_primary`, `best_hs_match`,
`sorted_container_order`). Tests: `tests/test_pv_selection.py`.

## One rule for both algorithms

`src/gnn/diagnostics/hs_selection_identical_rule.py` scores both algorithms
with the same functions. Only the inputs differ:

| | Track lists | Vertex position for chi2 |
|---|---|---|
| AMVF | stored `RecoVertex_assocTracks` | stored AMVF (fitted) position |
| PV-Finder+GNN | GNN top-1 assignment at score cut t | 3D adaptive fit of the assigned tracks (`rule_fit`); histogram peak on the beam spot (`rule_beam`) |

- **Chi2.** The (d0, z0) perigee chi2 relative to the vertex, with the
  straight-line prediction and the sign convention measured on this ntuple.
- **Calibration.** The median chi2 of tracks on their own AMVF vertex is
  1.28 (chi2(2) gives 1.39).
- **Events.** r16443, entries 0-24,999. PU200 = the note's 1,920 events. Only
  populated vertices, and the same peaks and GNN edges as `hs_chain_study`.
  No inference.

## Validation on AMVF

Our implementation, on AMVF's stored information, against Athena's stored
primary (`RecoVertex_type == 1`). The checks below use the first 3,000
entries (384 at PU200):

| Rule on the stored lists | Agreement with the stored primary, all | at PU200 |
|---|---:|---:|
| plain sum pT^2 (what the note did) | 86.8% | 79.7% |
| **+ compatibility chi2 <= 18.42 (Athena rule)** | **96.2%** | **94.8%** |
| + vertex position error (nominal 10/20 um) | 95.9% | 93.8% |
| all tracks within 3 mm, chi2 <= 18.42 (models shared links) | 95.7% | 92.7% |
| chi2 <= 9 / 25 / 50 | 96.9 / 95.4 / 93.3% | 94.3 / 92.7 / 90.4% |

- The ntuple gives every track to exactly one vertex and lists 99.5% of all
  tracks. 6% of the listed tracks fail AMVF's own compatibility cut with
  their vertex.
- **Conclusion:** the stored lists are not Athena's `trackParticleLinks`. They
  include tracks AMVF did not link and miss links to second vertices.
- Applying the compatibility cut, i.e. the ATLAS rule, recovers most of the
  difference. The remaining 4-6% is information the ntuple does not carry:
  shared links, the AMVF-refitted perigees used for the compatibility, and
  the vertex covariance (`RecoVertex_ErrX/Y/Z` are empty).
- On all 25,000 events, the reco-and-selected flag of our AMVF implementation
  agrees with the stored primary's in 96.2% of events, and in 93.9% at PU200.

## The one-event discrepancy (1,783 vs 1,784)

- **Where:** entry 9560 (mu = 189). AMVF vertices 0 (the stored primary) and 1
  both hold 27 HS tracks.
- **What each path did:**
  - `hs_metrics.hs_matched_index` broke the tie by our recomputed sum(pT^2)
    (vertex 1 = 1,702 GeV^2 against 1,302 GeV^2), so no match: 1,783.
  - `hs_selection_reconciliation` accepted any tied vertex: 1,784.
- **What Athena does:** `bestHardScatterMatch` gives the tie to the first
  vertex of Athena's sorted container. The stored primary is always index 0
  (checked on all 25,000 events), so the Athena-faithful count is **1,784
  (92.92%)**.
- **Fix:**
  - `hs_metrics.hs_matched_index` now takes the container order and uses
    `pv_selection.best_hs_match`.
  - The AMVF arms in `hs_chain_study` pass the stored order.
  - `hs_selection_reconciliation` uses the same function instead of its own
    tie rule.
  - The PV-Finder arms are unchanged: their default order is our sum(pT^2)
    order, stable on ties.
  - Both studies were re-run to `outputs/11_11_2026_output/` and give 1,784.
    Without the 0.1 mm requirement both give 1,785.
  - Six of the 25,000 events changed flags. At PU200 only the re-ranked
    variant moved, from 1,513 to 1,512.
  - The identical-rule study gives 1,513 for the same variant, because there
    each arm breaks HS-match ties in its own score order (the re-ranked
    container). The reconciliation uses the stored container for every flag.
    Both choices are stated. The headline numbers do not depend on this.
  - Test: `test_hs_matched_index_stored_order_for_athena_container`.

## Results at PU200 (1,920 events, reconstructed and selected, 0.1 mm)

Paired event bootstrap (2,000 resamples, seed 20261111). Output in
`outputs/11_11_2026_output/hs_selection_identical_rule/` (`summary.json`,
per-event `hs_selection_events.npz`).

| Arm | Efficiency | minus AMVF stored | minus AMVF, our code |
|---|---:|---:|---:|
| AMVF, Athena stored primary | **92.92 +- 0.58%** | - | +3.80 +- 0.56 |
| AMVF, identical rule (our code) | 89.11 +- 0.71% | -3.80 +- 0.56 | - |
| AMVF, plain sum pT^2 (note's 80.7% method) | 78.80 +- 0.93% | -14.11 | -10.31 |
| **GNN v4 e107 t=0.98, identical rule (fit)** | **94.90 +- 0.50%** | **+1.98 +- 0.54** | **+5.78 +- 0.66** |
| GNN v4 e107 t=0.98, identical rule (beam) | 94.17 +- 0.52% | +1.25 | +5.05 |
| GNN v4 e107 t=0.98, plain (current note) | 96.30 +- 0.43% | +3.39 +- 0.63 | +7.19 |
| GNN v4 e107 t=0.95, identical rule (fit) | 94.64 +- 0.51% | +1.72 | +5.52 |
| GNN v4 e107 t=0, identical rule (fit) | 92.08 +- 0.61% | -0.83 +- 0.60 | +2.97 |
| GNN full-mu e213 t=0.98, identical rule (fit) | 94.27 +- 0.51% | +1.35 +- 0.54 | +5.16 |
| GNN full-mu e213 t=0, identical rule (fit) | 91.04 +- 0.65% | -1.88 | +1.93 |

Versus mu (reconstructed and selected, %, 20-unit bins from 0-20 to 180-200):

| | 0-20 | 20-40 | 40-60 | 60-80 | 80-100 | 100-120 | 120-140 | 140-160 | 160-180 | 180-200 |
|---|---|---|---|---|---|---|---|---|---|---|
| AMVF stored | 98.2 | 97.4 | 97.1 | 95.4 | 95.5 | 94.7 | 93.4 | 93.2 | 93.3 | 92.8 |
| AMVF, identical rule | 98.2 | 97.2 | 96.3 | 94.9 | 94.0 | 93.0 | 90.9 | 89.9 | 90.0 | 89.2 |
| GNN v4 t=0.98, rule | 47.0 | 65.4 | 80.5 | 89.1 | 93.3 | 95.7 | 95.1 | 95.3 | 95.8 | 94.3 |
| GNN full-mu t=0.98, rule | 93.8 | 98.6 | 98.1 | 96.8 | 96.6 | 97.4 | 95.4 | 95.5 | 95.5 | 94.2 |
| GNN full-mu t=0, rule | 99.1 | 98.5 | 97.5 | 96.3 | 95.3 | 95.3 | 94.0 | 92.5 | 92.8 | 91.2 |

## Physics checks

- **High-pT incompatible tracks drive the wrong selections.** At PU200:
  - Where AMVF plain sum pT^2 picks the wrong vertex but the stored primary
    is right (330 events), the plain leader carries a track incompatible with
    it (chi2 > 18.42) above 50 GeV in 64% of events, and above 100 GeV in 38%.
  - Where the plain choice is right, this happens in 9% of events.
  - For the GNN at t=0 (no score cut), the same holds in 83% of the 258 events
    the rule fixes.
  - The compatibility cut is what removes these tracks. They are badly
    measured or secondary tracks with very large pT.
- **The GNN score cut already does most of this.** At t=0.98 the plain and
  rule results are within 1.4 points. At t=0 the rule adds 10.5 points.
- **Low mu.** The PU200-trained v4 GNN with a score cut loses HS tracks at
  low mu (47% at 0-20). The full-mu GNN keeps 94-99% in every bin.
  - Without a score cut, the full-mu GNN reaches 99.1% at 0-20 but falls to
    91% at 180-200.
  - So the best single t differs between low and high mu. This is input to
    the GNN working-point choice in the meeting plan.

## Which number to quote

The selection rule is now identical for both algorithms. Two comparisons are
valid:

1. **Conservative (recommended headline):**
   - GNN with the Athena rule: 94.9% (v4 e107, t=0.98).
   - AMVF with Athena's own stored primary: 92.9%.
   - **+2.0 +- 0.5 points.** Our implementation can only lose AMVF
     information, never add it, so this comparison favours AMVF.
2. **Strictly identical code on both:** 94.9% against 89.1%, +5.8 +- 0.7
   points. The code is the same, but the AMVF input lists are not Athena's
   links, so this overstates the GNN advantage.

The note's earlier +3.4 compared the GNN under the plain rule against AMVF
under Athena's rule. It should be replaced by comparison 1.

## Needed for an exactly identical AMVF computation

An Athena ntuple re-dump (Trey) with, per AMVF vertex:
- the full `trackParticleLinks` (non-exclusive) with the per-link
  `trackWeight` and `vtxCompatibility` from `vxTrackAtVertex`;
- the `sumPt2` decoration written by the sorting tool
  (`decorationName="sumPt2"`, `TrkVertexToolsConfig.py:16`);
- the filled vertex covariance `RecoVertex_ErrX/Y/Z`.

With the `sumPt2` decoration, our score can be checked vertex by vertex.
