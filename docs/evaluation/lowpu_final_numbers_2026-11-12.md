# Low pile-up final numbers, 12 November 2026

Production recompute of every low pile-up paper/note number (R1-R4 of
[paper_questions_2026-11-11.md](../additional_studies/paper_questions_2026-11-11.md),
Part 2) at the frozen `lowpu` working point. Samples: **Run 2 MC**, Run 2 data,
Run 3 data. There is no Run 3 MC.

All outputs are in `outputs/11_12_2026_output/lowpu_final/`; paths below are
relative to it.

## Working point (`get_regime("lowpu")`)

| | |
|---|---|
| Density model | PVF-e400 (two-phase, no KDE), `model_weights/lowpu_e400/e2e_mlpHist50_e2e400_1latent_mse_phase2_epoch_400_fullstate.pth`, sha256 `64e36a69...189baf03` |
| Peaks | amplitude 0.01, width 3, integral 0.50, no height cut, local centroid ±3 |
| Windows | 0.17 mm; hard scatter 0.1 mm; residual assignment 1 mm |
| σ_vtx-vtx | fitted on the counted peaks, 300 bins over ±6 mm |
| GNN | `model_weights/ttva_gnn_mu60_v2/ttva_gat_mu60_v2_aug_epoch_115.pyt`, sha256 `3a6d6596...273acefe`, **t = 0.97, frozen** (section 3) |
| Vertices after association | ≥ 2 associated tracks, for PV-Finder + GNN and AMVF (decided 12 Nov 2026) |

Every result file records the regime dict, the git hash and the checkpoint
sha256. No override was used in any production run.

## 1. Headline table

PV-Finder and AMVF are evaluated on the same events. Errors are event
bootstrap (2,000 replicas for vertex finding, 1,000 for the chain).

### R1. Run 2 MC vertex finding (2,550 test events, 72,189 truth vertices with ≥ 2 tracks, ⟨μ⟩ = 38.9)

| | PV-Finder | AMVF | PVF − AMVF |
|---|---|---|---|
| Efficiency at 0.17 mm | **81.55 ± 0.14 %** | 75.74 ± 0.17 % | +5.81 ± 0.14 |
| One-to-one efficiency | 76.24 % | 70.62 % | |
| Fake fraction (reco without truth in window) | 9.86 ± 0.12 % | 11.06 ± 0.14 % | |
| Fakes per event | 2.363 ± 0.040 | 2.487 ± 0.044 | −0.124 ± 0.035 |
| Reco vertices | 61,102 | 57,328 | |
| σ_vtx-vtx | **0.392 ± 0.009 mm** | 0.789 ± 0.015 mm | ratio 2.01 |
| 68% residual | 72.5 µm | 66.8 µm | |
| 0.17 mm / 68% residual | 2.35 | 2.54 | |
| HS efficiency at 0.1 mm | 98.94 % (2,523/2,550; 68% CI 98.70-99.14) | 99.29 % (2,532) | |

**The "2.5×" sentence.** 0.17 mm is 2.35 times PV-Finder's 68% residual and
2.54 times AMVF's. The HL-LHC rule used the larger of the two residuals, which
gives 2.35 here. Suggested wording: "about 2.4 times the 68% residual at low
pile-up". (The ep300 model's 69.5 µm gave 2.45.)

### R3. Collision data, AMVF (nTrk ≥ 2, beam-spot corrected) as reference, 0.17 mm

| | Run 2 data (2,500 evt, ⟨μ⟩ = 42.5) | Run 3 data (2,500 evt, ⟨μ⟩ = 50.0) |
|---|---|---|
| AMVF-vertex recovery | **94.39 ± 0.10 %** | **94.03 ± 0.09 %** |
| PVF vertices without AMVF vertex, per event | 3.006 ± 0.040 | 3.192 ± 0.040 |
| as a fraction of PVF vertices | 12.36 % | 11.18 % |
| PVF / AMVF vertices per event | 24.31 / 22.51 | 28.56 / 26.91 |
| σ_vtx-vtx, PV-Finder (all events) | **0.356 ± 0.010 mm** | **0.333 ± 0.008 mm** |
| σ_vtx-vtx, AMVF (same events) | 0.808 ± 0.015 mm | 0.813 ± 0.016 mm |

Data vs MC: Run 2 MC with AMVF as the reference (the data convention) gives
93.09 ± 0.12 % recovery and 2.998 ± 0.056 vertices without reference per event,
against 94.4 / 94.0 % and 3.0 / 3.2 in data. The large gap to the MC truth
numbers (81.6 %, 2.36) is therefore the choice of reference, as found on
9 November. Data/MC pairwise-Δz shapes agree on the plateau (χ²/ndf 202/200 and
197/200).

### R2. Run 2 MC chain (PV-Finder e400 peaks at integral 0.50 → GNN), t = 0.97

**Headline convention (12 Nov 2026): a reconstructed vertex counts only if it
has ≥ 2 associated tracks, for PV-Finder + GNN and for AMVF alike.**
Candidates with fewer tracks are dropped before classification, and their
tracks count as unassigned. Track-purity taxonomy (70 %).

| | PV-Finder + GNN (t = 0.97) | AMVF |
|---|---|---|
| Vertices (≥ 2 tracks) | 59,151 | 56,338 |
| Dropped candidates with 1 track (with 0 tracks) | 1,358 (593) | 593 (398) |
| Clean | 87.24 ± 0.17 % | 77.48 ± 0.21 % |
| Merged | 11.31 ± 0.16 % | 20.40 ± 0.20 % |
| Split | 1.45 ± 0.05 % | 2.11 ± 0.06 % |
| Fake | 0.003 % (2 vertices) | 0.004 % (2) |
| **Clean / truth (matched-vertex yield)** | **71.48 ± 0.24 %** | 60.47 ± 0.28 % |
| Track efficiency | 84.05 ± 0.11 % | 83.87 ± 0.12 % |
| Track purity | 89.91 ± 0.10 % | 85.88 ± 0.12 % |
| Track F1 | 86.88 ± 0.11 % | 84.86 ± 0.12 % |
| HS identification | 99.06 ± 0.19 % (2,526 / 2,550) | 98.75 ± 0.21 % (2,518) |

- **Track level, plainly:**
  - the GNN's track efficiency equals AMVF's (+0.18 points, about 1.5σ);
  - its purity is 4.0 points higher;
  - the F1 gain of 2.0 points comes entirely from purity.
- **HS identification:**
  - the largest-Σp_T² vertex is correct when the plurality of its
    truth-labelled tracks comes from the HS (`pv_type == 1`);
  - the same rule is used for both algorithms;
  - the difference of +0.3 points is about 1.5σ;
  - AMVF's stored primary gives 98.31 %.

**Backup, July convention (≥ 1 track, drop-empty)**, t = 0.97:
- PV-Finder + GNN: 60,509 vertices, Clean 86.90 %, Merged 11.05 %, Split
  2.04 %, Clean/truth 72.84 %, track F1 86.95 %, HS-ID 99.06 %;
- AMVF: 56,931 vertices, 77.23 / 20.19 / 2.53 %, Clean/truth 60.91 %, track
  F1 84.89 %, HS-ID 98.75 %.

The ≥ 2 rule removes mostly 1-track Split vertices: PV-Finder + GNN Split
drops from 1,235 to 858, Clean from 52,583 to 51,602. Files: `gnn_mc_ntrk2/`
(headline) and `gnn_mc/` (backup).

### R4. Run 3 data association, the same 2,500 events and peaks as R3

GNN vertices need ≥ 2 GNN-assigned tracks (headline). The tracks of dropped
vertices are unassigned, and dropped vertices take no part in the matching or
in the leading-vertex choice. AMVF: nTracks ≥ 2.

| t | GNN assigned | AMVF assigned | agreement / both-assigned (0.17 / 1 mm) | agreement / AMVF-assigned (0.17 / 1 mm) | leading-vertex agreement (0.17 mm) | 1-track GNN vertices dropped / event |
|---|---|---|---|---|---|---|
| **0.97** | **92.23 %** | 94.83 % | **95.18 / 96.13 %** | 91.29 / 92.21 % | **91.20 %** | 0.63 |
| 0.99 | 81.10 % | 94.83 % | 97.01 / 97.83 % | 82.62 / 83.32 % | 88.32 % | 1.24 |
| 0.95 | 95.02 % | 94.83 % | 94.61 / 95.59 % | 92.68 / 93.64 % | 91.32 % | 0.45 |
| 0.98 | 88.98 % | 94.83 % | 95.84 / 96.75 % | 89.11 / 89.96 % | 90.48 % | 0.84 |

At t = 0.97 there are 27.66 GNN vertices per event, against 26.91 for AMVF
(28.56 peaks). These are agreement numbers, not efficiencies. The ≥ 1
backup (`run3_assoc/`) at t = 0.97 gives 92.38 %, 95.09 / 96.03 %, 91.31 /
92.22 % and 91.20 %.

## 2. Quality gates (all passed)

- **Identical event sets.**
  - Run 2 MC: h5 test events 48450-50999 = ROOT entries `qibin[i]`. Truth and
    AMVF counts and μ re-read from ROOT match the eval's own per-event records
    in all 2,550 events.
  - The GNN graphs have the eval's peak count in all 2,550 events. Peaks
    re-found from the saved histograms equal the eval's list in 2,550/2,550.
  - Run 2 / Run 3 data: the same entries as April 2026 (Run 2 entries
    0-2,499; Run 3 the first 2,500 loader-accepted entries in 0-3,158). The
    AMVF reference lists equal the April ones in 2,500/2,500 events.
  - R4 uses the R3 pickle's entries. Its peaks, re-found from the stored
    histograms, and its AMVF lists equal the pickle's in 2,500/2,500 events.
    The AMVF self-agreement check is 1.
- **Bookkeeping.** Clean + Merged + Split + Fake = vertices, and correct ≤
  assigned-with-truth ≤ assigned, asserted per event for every t and for AMVF.
- **Eval vs recomputation.** Recomputed PVF efficiency and fakes equal the
  eval's own matcher to within one knife-edge vertex per sample. Run 2 MC:
  58,870 vs 58,871 found. Run 2 data: one AMVF vertex at |Δz| = 0.17000055 mm
  (float32). Run 3: identical (metric_definitions §5.2).
- **Regression of the chain code.** `lowpu_chain` on the July 2026 graphs
  (integral 0.2) reproduces paper Table 5 exactly:
  - t = 0.99: 61,845 vertices with tracks, 55,897 / 4,587 / 1,348 / 13, i.e.
    Clean 90.38 %, Merged 7.42 %, Split 2.18 %, Fake 0.021 %, Clean/truth
    77.43 %; 4,627 empty candidates;
  - max track F1 0.874 at t = 0.95;
  - AMVF 56,931 / 77.23 % / 60.91 %, F1 0.849, HS-ID 98.75 %.
  - The GNN HS-ID of 98.98 % is reproduced at t = 0.99 (and 0.8), not at the
    t = 0.5 that JOURNAL 2026-07-14 names (98.86 % there).
  - `gnn_regression_july/`.
- **AMVF anchors** (data_mc_validation_2026-11-09) through the new code on the
  same 2,550 events: 75.67 % and 2.50 per event at 0.1694 mm, σ 0.790 mm with
  the old 60-bin fit (doc: 0.789). At 0.17 mm AMVF is 75.74 %; only the window
  changed.
- **Histograms.** The in-repo e400 checkpoint reproduces the November 2025 e400
  MC histograms to a maximum of 7.8e-4 absolute (values up to 4.4; GPU
  rounding).
- **Independent recomputation:** section 5.

## 3. GNN checkpoint and t

**Checkpoint.** The low pile-up GNN used before is `ttva_gat_mu60_v2_aug`
epoch 115 (JOURNAL 2026-07-14; behind paper Table 5 and the July Run 3 study).
It was trained on the 48,450 non-test Run 2 MC events, so the 2,550 test events
are unseen. It is now set in `working_points.LOWPU`.

**t = 0.97 (frozen).**
- Rule (Mattia, 12 Nov): use the HL-LHC t = 0.97 unless the Run 2 MC scan
  shows it is clearly worse all-round.
- Operational test: does some other t beat 0.97 in Clean/truth, HS-ID and
  track F1 together, each by more than 2 paired standard errors? No t does,
  under either vertex convention, so t = 0.97 is kept.
  (`gnn_mc_ntrk2/t_options.json`, `gnn_mc/t_options.json`)
- Options, derived on the 2,550 test events:
  - There is no GNN-independent validation sample at low pile-up.
  - The option values are the same on the even and odd halves, except B,
    which is flat between 0.85 and 0.95.

Headline convention (vertices with ≥ 2 tracks):

| Option | t | Clean/truth | Clean/reco | Merged | Split | track F1 | HS-ID | Clean/truth μ 0-20 / 20-40 / 40-60 / 60-80 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Chosen (= HL-LHC)** | **0.97** | **71.48 ± 0.24** | 87.24 | 11.31 | 1.45 | 86.88 | 99.06 | 87.4 / 79.5 / 71.1 / 65.3 |
| A: max Clean/truth | 0.985 | 71.94 ± 0.22 | 90.03 | 9.07 | 0.90 | 85.79 | 98.94 | 86.6 / 79.0 / 71.7 / 66.3 |
| D: most μ-robust | 0.98 | 71.85 ± 0.23 | 88.83 | 10.03 | 1.13 | 86.41 | 98.98 | 87.0 / 79.4 / 71.6 / 65.9 |
| B / C: max HS-ID / track F1 | 0.94 / 0.95 | 70.57 / 70.80 | 85.09 / 85.53 | 13.1 / 12.7 | 1.8 / 1.7 | 87.02 / 87.03 | 99.10 | 87.3 / 79.3 / 70.6 / 64.2 (0.95) |
| July proposal | 0.99 | 71.29 ± 0.22 | 91.32 | 8.06 | 0.62 | 84.41 | 98.94 | 85.6 / 78.0 / 70.9 / 66.0 |
| no cut | 0 | 68.83 | 82.68 | 15.41 | 1.91 | 86.46 | 98.78 | 86.2 / 77.6 / 68.5 / 62.0 |
| AMVF | - | 60.47 | 77.48 | 20.40 | 2.11 | 84.86 | 98.75 | 84.2 / 71.6 / 59.4 / 52.0 |

**Paired differences against t = 0.97** (points):
- **t = 0.985:**
  - Clean/truth +0.45 ± 0.07;
  - track F1 −1.09 ± 0.02;
  - HS-ID −0.12 ± 0.10;
  - Clean/reco +2.8.
- **t = 0.99:**
  - Clean/truth −0.19 ± 0.09, so 0.99 is no longer the best once 1-track
    vertices are dropped;
  - track F1 −2.47 ± 0.03;
  - HS-ID −0.12 ± 0.11.
- **t = 0.95:**
  - Clean/truth −0.68 ± 0.06;
  - track F1 +0.15 ± 0.01;
  - HS-ID +0.04 ± 0.04.

**Reading.** t = 0.97 is within 0.5 points of the best Clean/truth and within
0.15 points of the best F1, and its HS-ID equals the best within errors.
Higher t gives purer vertices (Clean/reco) at the cost of track efficiency.
This matches HL-LHC, where 0.97 was chosen for the same reasons. One t serves
both regimes.

Backup (≥ 1 track, `gnn_mc/t_options.md`):
- t = 0.97: Clean/truth 72.84 %, track F1 86.95 %, HS-ID 99.06 %;
- the maximum Clean/truth is at 0.99 (74.35 %), with track F1 −2.4 points.

## 4. What changed against earlier numbers

| Quantity | Earlier (source) | Now | Why |
|---|---|---|---|
| MC efficiency | 91.59 % at 0.416 mm; 80.85 % at 0.1694 mm (ep300, full-region mean) | 81.55 % at 0.17 mm | window (most of it), e400, centroid |
| MC fakes / event | 0.419 at 0.416 mm; 2.013 at 0.1694 | 2.363 | window; e400 makes 0.5 more peaks per event |
| MC σ_vtx-vtx | 0.416 ± 0.011 mm (60 bins) | 0.392 ± 0.009 mm | e400 separates better (ep300 is 0.410 at 300 bins) |
| AMVF MC σ_vtx-vtx | 0.793 (other 2,500 events) | 0.789 ± 0.015 (same events) | same events (Q24) |
| AMVF MC efficiency | not given | 75.74 % | new (Q24) |
| PVF 68% residual | 69.5 µm (ep300) | 72.5 µm | e400 model (the centroid improves it by 0.8 µm) |
| HS efficiency 0.1 mm | 99.53 % (ep300, recomputed) | 98.94 % | e400 model (the centroid adds 5 events) |
| Run 2 / Run 3 data recovery | 95.31 / 94.78 % at own σ (0.368 / 0.348 mm) | 94.39 / 94.03 % at 0.17 mm | window; ep300 at 0.17 mm gives 93.47 / 93.22 % |
| Data, no reference / event | 1.821 / 1.959 | 3.006 / 3.192 | window (ep300 at 0.17: 2.72 / 2.82) and e400's extra peaks |
| Data σ_vtx-vtx | 0.368 / 0.348 mm | 0.356 / 0.333 mm | e400; now 300 bins |
| Chain Clean/truth | 77.43 % (integral 0.2, t = 0.99, ≥ 1 track) | 71.48 % (integral 0.50, t = 0.97, ≥ 2 tracks); 72.84 % with ≥ 1 track | integral 0.50 finds 5,370 fewer candidates (61,102 vs 66,472), mostly small vertices; then the 2-track rule (−1.4 points) and t |
| AMVF chain Clean/truth | 60.91 % (all AMVF lists) | 60.47 % (≥ 2 tracks; 593 1-track AMVF vertices dropped) | same rule for both |
| Chain Merged / Split | 7.42 / 2.18 % | 11.31 / 1.45 % | t = 0.97 instead of 0.99; 1-track splits dropped |
| Track F1 | 0.874 (t = 0.95, a different cut) | 0.869 at the working point (AMVF 0.849) | one t |
| HS-ID | 98.98 % | 99.06 % (AMVF 98.75 %) | |
| Run 3 association | t = 0.98, KDE model, 2,000 other events, 1 mm: assigned 83.4 %, agreement 95.7 %, leading 88.7 % | t = 0.97, e400, R3 events, ≥ 2 tracks: 92.2 %, 96.1 % (1 mm), 91.2 % | frozen model, same events (Q27), t |

Centroid vs full-region mean on the e400 peaks
(`summary/diag_centroid_vs_mean.json`), efficiency / fakes per event / p68 /
HS: 81.55 % / 2.363 / 72.5 µm / 2,523 against 81.59 % / 2.375 / 73.2 µm /
2,518. The estimator matters little at low pile-up. The position and HS
changes come from the model.

## 5. Independent recomputation

A second agent wrote separate code from the definitions only. It did not
import the matcher, the classifier or any of the scripts above. Results are in
`crosscheck/`. See the comparison below.

**R1 and R3: every count identical.**
- Run 2 MC: found 58,870 / 54,679, fakes 6,026 / 6,342, one-to-one 55,037 /
  50,983, reco 61,102 / 57,328 (PV-Finder / AMVF).
- 68% residuals 72.46 / 66.80 µm; HS 2,523 / 2,532.
- σ_vtx-vtx 0.39165 / 0.78850 mm.
- Run 2 data: 53,123 / 56,282 found, 7,516 without reference, σ 0.35586 mm.
  Run 3 data: 63,251 / 67,267, 7,980, σ 0.33334 mm.
- AMVF data σ: 0.8075 / 0.8131 mm.
- The cross-check's σ errors are smaller because it fills both signs of every
  pair. The central values are equal.

**R2, Run 2 MC chain.**

Headline, ≥ 2 tracks, t = 0.97:

| | primary | cross-check |
|---|---|---|
| Clean / Merged / Split / Fake | 51,602 / 6,689 / 858 / 2 | 51,604 / 6,684 / 855 / 3 |
| Vertices / dropped 1-track | 59,151 / 1,358 | 59,146 / 1,360 |
| Clean / truth | 71.482 % | 71.485 % |
| track efficiency / purity / F1 | 84.050 / 89.909 / 86.880 % | 84.049 / 89.908 / 86.879 % |
| HS identification | 2,526 | 2,526 |
| AMVF | 43,653 / 11,494 / 1,189 / 2, 56,338, 593 dropped, F1 84.862 %, HS 2,518 | identical |

Other comparisons:
- At t = 0.99: Clean 51,462 vs 51,463, HS 2,523 in both.
- The ≥ 1 backup agrees equally well: at t = 0.97, 52,583 vs 52,584 Clean and
  60,509 vs 60,506 vertices; AMVF is identical.

The GNN differences of a few vertices come from one event, h5 49540. There,
303 tracks have a score of exactly 1.0 (saturated float32 sigmoid) to two
neighbouring peaks. Production (`get_top_k_associations`, `np.argpartition`)
and the cross-check (lowest edge index) break that tie differently. The CPU and
GPU scores otherwise agree to 2e-5, with no top-1 change and no crossing of
t = 0.97 in any event. This is a tie-breaking convention, not a definitional
difference. It moves Clean/truth by 0.003 points.

**R4, Run 3 data.** At t = 0.97, every count is identical in both conventions:
- ≥ 2 tracks: 994,433 GNN-assigned tracks, 980,612 both-assigned, 933,361
  agreeing at 0.17 mm and 942,705 at 1 mm, 2,280 leading-vertex agreements,
  1,563 dropped 1-track vertices;
- ≥ 1 track: 995,996, 981,768, 933,574, 942,817 and 2,280.

At t = 0.99 the two codes agree within 1 track, a CPU/GPU score at the cut.


## 6. Figures (ATLAS style, PDF + PNG)

| Paper figure | File |
|---|---|
| Fig. 8(a): Run 2 MC Δz, PV-Finder and AMVF | `summary/figures/fig8a_run2mc_deltaz` |
| Fig. 8(b): Run 2 MC vertices per event vs μ | `summary/figures/fig8b_run2mc_yield_vs_mu` |
| Fig. 9(a): matched-vertex efficiency vs truth n_trk, t = 0.97, ≥ 2 tracks | `gnn_mc_ntrk2/figures/fig9a_run2mc_eff_vs_ntrk` (backup ≥ 1: `gnn_mc/figures/`) |
| Fig. 9(b): categories, t = 0.97, ≥ 2 tracks | `gnn_mc_ntrk2/figures/fig9b_run2mc_categories` |
| Fig. 10(a, c): Run 2 / Run 3 data Δz with Run 2 MC | `summary/figures/fig10a_run2_deltaz`, `fig10c_run3_deltaz` |
| Fig. 10(b, d): data yields vs μ with Run 2 MC | `summary/figures/fig10b_run2_yield_vs_mu`, `fig10d_run3_yield_vs_mu` |
| t scan | `gnn_mc_ntrk2/figures/gnn_t_scan_run2mc` |

Fig. 10 (Q28): the panels have no "integral=0.2" title. Every panel states
"PV-Finder integral threshold 0.50", which is what was run. Simulation panels
say "Simulation Internal"; data panels say "Internal". The Run 3 panels say the
MC is Run 2 MC.

## 7. Things to know

- **Fix applied** (commit 20946fa). Since 23 April 2026, `run_eval_pvf_run3.py`
  read the empty `TruthVertex_*` branches of collision-data files as MC truth.
  Every data event then had zero reference vertices. The April data pickles
  predate the change and are fine. Any data eval between April and now with
  that script would be wrong; none is quoted in the note.
- **Parameter count.** The e400 model has 358,713 trainable parameters. The
  359,616 in `working_points.py` and the earlier docs includes the batch-norm
  running statistics. metric_definitions §5.3 is corrected; the frozen note
  string is left unchanged so that the stored regime dicts stay comparable.
- **e400 vs ep300 at low pile-up.**
  - e400 is better on efficiency (+0.6 points) and separation (σ −0.02 mm).
  - It is worse on position (p68 +3 µm) and HS efficiency at 0.1 mm (−0.6
    points, 15 events).
  - It makes more peaks (+0.5 per event, +0.36 fakes per event).
- **No validation set for t at low pile-up.** The GNN trained on all non-test
  events. t is a single parameter on 72k vertices, so the bias is negligible.
  The halves agree.
- **Not recomputed here:**
  - the truth-guided / oracle bounds of the July chain (81.9 %), quoted in the
    paper text;
  - the low pile-up HS selection with the Athena rule and the 3D-fit HS
    resolution, which exist only for HL-LHC.
- **src_dirty = True** in the result files. It comes from untracked files
  under `src/` that are not used by these runs (`peak_classifier*.py` and the
  HL-LHC agent's work in progress). The evaluation code that ran is in commits
  20946fa and bdae061.

## Reproduction

```bash
# inside tmux, repo root, venv active
SAMPLE=mc   DEVICE=2 bash scripts/run_lowpu_final_vf.sh
SAMPLE=run2 DEVICE=2 bash scripts/run_lowpu_final_vf.sh
SAMPLE=run3 DEVICE=2 bash scripts/run_lowpu_final_vf.sh
B=outputs/11_12_2026_output/lowpu_final; export PYTHONPATH=src
python -u scripts/lowpu_mc_histograms.py --device 2 --out $B/mc_hists
python -u -m gnn.data.pvf_to_graphs -r $B/mc_hists/hists_e400.npy \
    -f /share/lazy/qibinlei/recoTracks_incamvfassoc.h5 -i configs/qibin_test_main_indices_v2.p \
    -o $B/gnn_mc/graphs_e400_lowpu.pt --regime lowpu
W=model_weights/ttva_gnn_mu60_v2/ttva_gat_mu60_v2_aug_epoch_115.pyt
# headline (>= 2 tracks, the default) and backup (>= 1 track)
python -u -m gnn.evaluation.lowpu_chain --graphs $B/gnn_mc/graphs_e400_lowpu.pt --weights $W \
    --vf-pkl $B/vf_mc/eval_results.pkl -d 2 --n-boot 1000 --min-tracks 2 --out $B/gnn_mc_ntrk2
python -u -m gnn.evaluation.lowpu_chain --graphs $B/gnn_mc/graphs_e400_lowpu.pt --weights $W \
    --vf-pkl $B/vf_mc/eval_results.pkl -d 2 --n-boot 1000 --min-tracks 1 --out $B/gnn_mc
python -u -m gnn.evaluation.run3_assoc_lowpu --vf-pkl $B/vf_run3/eval_results.pkl \
    --root data/run3/file_3.root --weights $W -d 2 --min-tracks 2 --out $B/run3_assoc_ntrk2
python -u -m gnn.evaluation.run3_assoc_lowpu --vf-pkl $B/vf_run3/eval_results.pkl \
    --root data/run3/file_3.root --weights $W -d 2 --min-tracks 1 --out $B/run3_assoc
python -u scripts/lowpu_final_summary.py --base $B
python -u scripts/lowpu_gnn_t_options.py --dir $B/gnn_mc_ntrk2
python -u scripts/lowpu_gnn_t_options.py --dir $B/gnn_mc
# regression on the July graphs
python -u -m gnn.evaluation.lowpu_chain --graphs outputs/07_12_2026_ttva_reproduction/regen/graphs_pvf_e400_regen.pt \
    --weights $W --vf-pkl outputs/04_16_2026_output/reproduction_ep300_mc/eval_results.pkl \
    -d 2 --n-boot 50 --min-tracks 1 --out $B/gnn_regression_july
```

Numbers: `summary/vf_results.json` (R1, R3, April and anchors),
`gnn_mc_ntrk2/summary.json` and `gnn_mc_ntrk2/t_options.{json,md}` (R2
headline; `gnn_mc/` is the ≥ 1 backup), `run3_assoc_ntrk2/summary.json` (R4
headline; `run3_assoc/` backup). The ≥ 1 runs (`gnn_mc`, `run3_assoc`,
`gnn_regression_july`) were made before `--min-tracks` existed; their
behaviour is `--min-tracks 1`.
