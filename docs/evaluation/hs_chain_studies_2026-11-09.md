# Hard-scatter studies and Fig. 7.2 extensions, 9 November 2026

Covers note Sec. 4.1, 7.1, 7.4 and 7.5. All arms (PV-Finder + GNN, a z0
baseline, the truth-guided reference and AMVF) run on the same events and
use one unweighted, truth-based classification (`gnn.evaluation.hs_metrics`).
No network inference is repeated. The study replays the saved PV-Finder
histograms and the saved top-1 GNN edges.

## Inputs and conventions

- Sample: r16443 flat-mu ITk ttbar (single lepton), 14 TeV, entries 0-24,999,
  0 <= mu <= 200. PU200 means the note's 1,920 events with 185 <= mu <= 215
  (actual range 185-200, mean 192.5). Same events as Sec. 4.1 and 7.5.
- Peaks: the note's production settings (0.01 / 0.30 / width 3 / height 0.03 /
  centroid +-3 bins). They are recomputed from the histograms and checked
  against the saved evaluation for every event.
- **GNN checkpoint: v4 noaug epoch 107**
  (`model_weights/ttva_gnn_hllhc_v4_noaug/ttva_gat_alleta_k20_v4_noaug180k_epoch_107.pyt`).
  The note uses this checkpoint (Table tab:gnn_training: 180k all-hadronic,
  108 epochs, no augmentation, 146,269 Clean at t=0.98). The fullmu v1 e213
  checkpoint (trained over mu 0-200) is shown as a secondary series. Working
  points: t=0.98 for vertex yields, t=0.95 for HS selection (note values).
- Reco vertices: populated vertices (>= 1 assigned track, "drop-empty") as
  in the note. Results for >= 2 tracks are also stored (`reco_min_tracks_2`).
- Windows: **0.1 mm for every HS quantity; 0.1694 mm for all other vertex
  matching** (one-to-one finder match and truth-guided assignment). 0.5 mm
  versions are kept only to reproduce the current Fig. 7.2.
- Errors: 68.27% Clopper-Pearson intervals for event fractions; event-bootstrap
  standard deviation (1,000 resamples, seed 20261109) for ratios of sums,
  with paired differences.

Validation against the note on the 1,920 events (all exact unless stated):
GNN t=0.98 Clean 146,269, populated vertices 175,492 (note 175,491: one
replay split), Fake 303; AMVF Clean 66,127 / 187,960; one-to-one 0.5 mm 78.09%;
truth-guided 0.5 mm 73.66%; HS reconstruction 1,918 / 1,916; legacy HS-ID
AMVF 1,549 (80.68%), GNN t=0.95 1,814 (note 1,813; replay difference of one event).

## Definitions

HS truth vertex: `TruthVertex_type == 1` (exactly one per event; no track cut).
HS track: a selected reconstructed track (pT > 0.9 GeV, ITk Table 6 cuts,
|eta| < 4) listed in the HS truth vertex's `TruthVertex_assocTracks`.

- **(a) HS reconstruction efficiency.** The fraction of events with at least one
  reconstructed vertex with |z_reco - z_HS| < 0.1 mm. This is the ITk
  Sec. 5.5 / Fig. 26(b) text definition (z only). Athena uses a 3D distance
  to the truth-matched vertex; we do not.
- **HS-matched vertex.** The vertex holding the largest number of HS tracks.
  Ties go to the larger sum(pT^2). This is the ITk paper's definition and
  Athena `bestHardScatterMatch`, counted without weights.
- **(b) Clean-HS fraction.** The fraction of events in which some vertex has
  the HS as its dominant contribution and is classified Clean (>= 70% of its
  tracks from the HS, not Split). At most one such vertex exists per event.
- **(c) HS-track association fraction.** Sum over events of the HS tracks on the
  HS-matched vertex, divided by the sum of all HS tracks in the events.
  Unassigned HS tracks count in the denominator. The companion purity is HS
  tracks on that vertex divided by all its tracks. These are unweighted
  counts. The AMVF note (Sec. 7.5) instead counts fit-compatible tracks per
  eta bin.
- **(d) HS selection efficiency.** The selected vertex is the vertex with the
  largest sum(pT^2) of its assigned tracks. The main quantity is
  "reconstructed and selected": the selected vertex is the HS-matched vertex
  and lies within 0.1 mm of z_HS. This is the ITk Fig. 27 definition. The
  note's legacy rule ("the selected vertex's dominant labelled truth is the
  HS") is stored as `_plurality`. For AMVF there are two variants. "Stored
  primary" uses `RecoVertex_type == 1`, which is ATLAS's own choice. "Re-ranked"
  takes the largest sum(pT^2) over the stored track lists, the same code as
  the GNN.

## Results at PU200 (1,920 events, reco >= 1 track)

| | GNN t=0.95 | GNN t=0.98 | GNN fullmu t=0.98 | nearest z0 | AMVF |
|---|---:|---:|---:|---:|---:|
| (a) HS reco, any vertex | 99.84 | 99.84 | — | 99.90 (peaks) | 99.79 |
| (b) Clean-HS fraction | 90.73 +0.67/-0.71 | **96.35** +0.43/-0.48 | 96.6 | 92.81 | **92.50** +0.60/-0.65 |
| (c) HS-track fraction | 85.80 +- 0.22 | **74.87** +- 0.31 | 75.20 | 71.99 | **74.11** +- 0.30 |
| (c') HS-vertex purity | 83.89 | 91.31 | 90.83 | 88.71 | 88.37 |
| (d) reco+sel., ITk rule | 94.27 +0.53/-0.58 | 96.30 +0.43/-0.48 | 95.31 | 77.03 | 78.80 (re-ranked) / **92.86** (stored primary) |
| (d) legacy plurality | 94.48 | 96.46 | — | 80.0 | 80.68 |

(a) PV-Finder peaks 1,918/1,920 = 99.90 +0.07/-0.14%, AMVF 1,916/1,920 =
99.79 +0.10/-0.16% (unchanged from the Sept. 17 study). Paired differences
from AMVF, (c): GNN t=0.98 +0.76 +- 0.29 points; t=0.95 +11.69 +- 0.27;
no score cut +17.48 +- 0.27 (91.59%, at 73.4% purity). Full mu-bin tables
with errors are in `outputs/11_09_2026_output/hs_chain_study/tables.md`.

Behaviour versus mu (measured): the v4 checkpoint, trained at mu 190-210,
loses HS tracks at low mu under a score cut. At 0-20, (c) is 37.8% (t=0.95)
and (d) is 63.0%. The fullmu checkpoint at t=0.98 removes this: (d) is
94.5-98.3% in every bin, and is above the AMVF stored primary for mu >= 20.
AMVF re-ranked falls from 97.2% to 80.0% across 0-200; the stored primary
falls from 98.2% to 92.9%.

## Reconciliation of 80.7% with ITk Fig. 27 (~92%)

ITk paper (arXiv:2412.15090, Sec. 4.2, 5.5): the sample is ttbar at <mu> = 200,
Poisson around 190-210. The selected vertex is the largest sum(pT^2) PV. The
HS-matched vertex has "the largest number of tracks matched to true particles
from the HS". Fig. 27 plots "reconstructed and selected", about 92% at
mu = 200. Athena (`InDetPerfPlot_VertexTruthMatching`, via the BNL LXR mirror)
implements this as pointer equality between the event's stored primary and
`bestHardScatterMatch`. It applies a 3D distance cut. A reco-conditional
variant also exists. Track-to-particle matching there uses R_match > 0.5;
we use the stored truth-vertex association.

Steps on the identical 1,920 events
(`outputs/11_09_2026_output/hs_selection_reconciliation/reconciliation_table.md`):

| Step | Definition | Efficiency [%] | Change |
|---|---|---:|---:|
| 0 | Note: re-rank stored links by sum(pT^2); dominant truth = HS | 80.68 (79.73-81.59) | |
| 1 | + ITk criterion (most HS tracks) | 78.91 | -1.77 |
| 2 | + within 0.1 mm (reconstructed and selected) | 78.80 | -0.10 |
| 3 | + stored AMVF primary instead of re-ranking | **92.92** (92.28-93.50) | **+14.11** |
| 4 | + restrict to 190 <= mu <= 200 (ITk range) | 93.56 (92.81-94.24) | +0.64 |
| — | conditional on HS reconstruction (1,916 events) | 93.11 | |

The whole gap comes from the choice of selected vertex. The definition of a
match moves the result by only about 2 points, in the wrong direction. The
stored primary differs from the re-ranked leader in 428/1,920 events.
**Diagnostic:** re-ranking the same stored lists with track pT capped at
10/15/20 GeV reproduces the stored primary in 92.7/92.9/92.5% of events,
against 77.7% uncapped. 16.4% of re-ranked leaders hold a track with
pT > 100 GeV. Every track sits in exactly one stored list, whereas AMVF can
share a track between vertices. The stored lists are therefore not the
lists that the release-24 unweighted SumPt2 tool ranked. Which producer
setting causes this is not established. With the 15 GeV cap, the re-ranked
ITk-rule efficiency is 97.81%. This shows how sensitive sum(pT^2) selection
is to a few very-high-pT tracks. It is not a proposed AMVF number.

**Run 3 MC.** None exists in the repo data directories, `/share/lazy` or
`docs/data/`. Everything named Run 3 is collision data, which has no truth.
The closest sample is the 51k 13 TeV ttbar ID sample with flat mu 0-80
(`ATLAS_PVFinderData_TruthMatched.root` and `recoTracks_incamvfassoc.h5`).
The note calls it Run 2 MC. The ITk paper's Run 3 comparison curve uses a
uniform 0-80 profile of the same kind. On it, the AMVF stored primary gives
98.33 (98.27-98.38)% reconstructed and selected, and re-ranking gives 98.74%
(50,996 events; 4 events without a 2-track primary are skipped). The HDF5
truth-vertex order differs from the ROOT order. HDF5 types and lists are
used; track pT and AMVF z are checked against ROOT for every event.
A real Run 3 check needs truth-matched mc23 ttbar ntuples (13.6 TeV, Run 3
ID) with `RecoVertex_type`, AMVF track lists and truth associations;
ideally they would also contain fit weights and truth-match probabilities.

## Fig. 7.2 extension: z0 baseline

The baseline uses the same PV-Finder peaks and classification. Each track
goes to the peak nearest in |z0 - z_peak|. Variant 2 keeps the association
only if |dz| < 3 sigma(z0). PU200 clean/truth: nearest z0 31.90 +- 0.11%,
3-sigma 33.06 +- 0.11%. For comparison, AMVF is 30.94 +- 0.11%, GNN t=0.98
68.43 +- 0.12%, truth-guided 72.75% and the one-to-one finder 73.43% (both at
0.1694 mm; at 0.5 mm they are 73.66% and 78.09%). The GNN gain over the
baseline is therefore the learned assignment and score cut, not the peaks.
On the same peaks the baseline is close to AMVF at every mu.

Figures (PDF+PNG) are in `outputs/11_09_2026_output/hs_chain_study/figures/`:
`clean_yield_stages_pu200_0p1694` (proposed Fig. 7.2),
`clean_yield_stages_pu200_0p5` (note-consistent windows),
`clean_yield_stages_vs_mu`, `hs_reco_eff_vs_mu`, `hs_clean_fraction_vs_mu`,
`hs_track_fraction_vs_mu`, `hs_vertex_purity_vs_mu`, `hs_selection_eff_vs_mu`.
Files with the `_reco2` suffix use >= 2 tracks. The AMVF selection plot for
both samples is `outputs/11_09_2026_output/hs_selection_reconciliation/amvf_hs_selection_vs_mu`.

## Reproduction

```bash
export PYTHONPATH=src   # repo root, inside tmux, ulimit -v 33554432
python -u src/gnn/diagnostics/hs_chain_study.py --out outputs/11_09_2026_output/hs_chain_study --workers 6
python -u src/gnn/diagnostics/hs_chain_summary.py --study outputs/11_09_2026_output/hs_chain_study
python -u src/gnn/diagnostics/plot_hs_chain.py --summary outputs/11_09_2026_output/hs_chain_study/hs_chain_summary.json -o outputs/11_09_2026_output/hs_chain_study/figures
python -u src/gnn/diagnostics/hs_selection_reconciliation.py itk   --out outputs/11_09_2026_output/hs_selection_reconciliation
python -u src/gnn/diagnostics/hs_selection_reconciliation.py lowmu --out outputs/11_09_2026_output/hs_selection_reconciliation
python -u src/gnn/diagnostics/plot_hs_selection_reconciliation.py --dir outputs/11_09_2026_output/hs_selection_reconciliation
python -m pytest -q tests/test_hs_metrics.py
```

The study takes about 2 minutes on 6 CPU workers and needs no GPU. Tests:
the vectorized classifier matches `classify_assignments` on 40 random events,
including ties, unlabelled tracks and empty vertices. They also cover the
HS definitions, the baselines, the windows and the error helpers.
