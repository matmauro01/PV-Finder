# HL-LHC density-model freeze: candidates at the final working point (12 November 2026)

Step 2a of the consolidated recompute
(`docs/additional_studies/meeting_2026-11-11_plan.md`, decision D3). Four
candidates are compared with production v6 on the two held-out PU200 files,
at the final HL-LHC working point (`pv_finder.utils.working_points`,
`--regime hllhc`). The comparison covers two of the three freeze criteria:
vertex finding before the GNN, and resolution. The third, vertex categories
after the GNN, needs the frozen GNN v2 (step 2b). **No freeze is
recommended here.**

## Setup

- **Candidates:**

  | Model | Checkpoint | U-Net ch / latent | Training |
  |---|---|---|---|
  | v6 (reference) | `hllhc_alleta_v6_mse_2ep_phase2_epoch_2_fullstate.pth` | 280 / 4 | 2+2 epochs |
  | s5 1+1 | `hllhc_alleta_v6_s5_1p1ep_phase2_epoch_1_fullstate.pth` | 112 / 1 | 1+1 |
  | s5 2+2 | `hllhc_alleta_v6_s5_2p2ep_phase2_epoch_2_fullstate.pth` | 112 / 1 | 2+2, as v6 |
  | s8 | `hllhc_alleta_v6_s8_1p1ep_phase2_step_450000_fullstate.pth` | 80 / 1 | 1+1, step 450k of 465,624 |

  s5 2+2 final validation loss is 0.016224, against 0.016352 for s5 1+1 and
  0.015755 for v6.
- **Events:**
  - r16443: the first 25,000 entries, of which 1,920 events have 185 <= mu <= 215.
  - r16638: the same selection, 1,888 events.
  - The PU200 event set is asserted identical across all models and AMVF
    (sha1 of the event indices in each `compare.json`).
  - The density models never saw either file. r16638 is now GNN v2
    training data, which does not matter for this vertex-finding-only study.
- **Working point:**
  - matching window 0.17 mm; HS window 0.1 mm;
  - peak finding: amplitude 0.01, width 3, integral 0.30, height 0.03,
    local centroid ±3;
  - sigma_vtx-vtx fitted on the counted peaks.
- **Code:**
  - Evaluations: `scripts/eval_small_models_heldout.sh` with
    `EXTRA=--save-histograms`, git 9739747. The INVOCATION "src dirty" flag
    comes from untracked files only; see `PROVENANCE_AT_LAUNCH.txt`.
  - Comparison: `scripts/run_freeze_compare.sh`, which runs
    `matched_fp_scan.py`, `freeze_candidates_compare.py` and
    `freeze_candidates_plots.py`, at git e541312.
  - Outputs: `outputs/11_12_2026_output/freeze_candidates/`.
- **Errors:**
  - Paired event bootstrap, with the same resampled events for both arms:
    2,000 replicas for counts, 200 for the sigma and p68 refits.
  - HS efficiency: 68% Clopper-Pearson intervals.
  - sigma is quoted with its fit error; the bootstrap error is about half
    of it (1.4-1.8 um, against 3.0-3.6 um).

## Checks passed before any number is quoted

- **Replay:** for every model, replaying the saved settings reproduces the
  stored per-event found, fake and split counts in all events.
- **Bookkeeping:** clean + merged + split + fake = reconstructed vertices,
  and truth clean + merged + missed = truth vertices, in every event.
- **Eq. (1) identity:**
  - The efficiency recomputed from positions alone (truth with at least
    one reconstructed vertex within 0.17 mm) equals the matcher's
    merged-credit numerator event by event, for every model and for AMVF.
  - This holds by construction of the closest-first matching, but only if
    the same float32 bin-unit arithmetic is used. In float64 mm, 2 of
    213,738 truth vertices sit within 2e-6 mm of the window edge and flip.
  - So Eq. (1) of the paper is exactly the code's "Eff".
- **sigma_vtx-vtx:** recomputing it from the saved peaks reproduces the
  value stored by `run_eval_pvf_run3.py` to 1e-8 relative.
- **AMVF anchors:** the counts match the saved evaluation, and the anchors
  reproduce the previously published values:
  - efficiency 79.85% (79.80% at 0.1694 mm);
  - sigma_vtx-vtx 0.305 mm;
  - 68% residual 56.45 um;
  - HS 1,916 / 1,920.
- **Spot check:** `compare_*/spot_check_*.png` shows five events. The
  densities of the models nearly overlap; their peak lists differ only on
  weak, low-density structures.

## 1. Vertex finding before the GNN

Efficiency is Eq. (1): truth vertices with n_trk >= 2 found within 0.17 mm,
divided by all truth vertices. Differences are paired, in percentage points.

**r16443 (1,920 events)**

| Model | Cuts (integral / height) | Eff [%] | dEff [pt] | Fakes / event | Fake fraction [%] | Splits / event | HS reco (0.1 mm) |
|---|---|---:|---:|---:|---:|---:|---:|
| v6 | 0.30 / 0.03 | 83.20 | | 18.70 | 18.51 | 0.535 | 1,918 (99.90%) |
| s5 1+1 | 0.30 / 0.03 | 82.88 | -0.33 ± 0.04 | 17.85 | 17.90 | 0.466 | 1,917 |
| s5 1+1 | 0.273 / 0.0273 (matched) | 83.02 | **-0.18 ± 0.04** | 18.71 | 18.57 | 0.477 | 1,917 |
| s5 2+2 | 0.30 / 0.03 | 83.01 | -0.20 ± 0.04 | 17.83 | 17.85 | 0.465 | 1,918 |
| s5 2+2 | 0.271 / 0.0271 (matched) | 83.16 | **-0.05 ± 0.04** | 18.71 | 18.53 | 0.481 | 1,918 |
| s8 | 0.30 / 0.03 | 82.41 | -0.80 ± 0.04 | 15.59 | 16.07 | 0.467 | 1,918 |
| s8 | 0.182 / 0.0182 (matched) | 83.07 | **-0.14 ± 0.04** | 18.71 | 18.54 | 0.562 | 1,918 |
| AMVF | n_trk >= 2 | 79.85 | -3.35 ± 0.06 | 19.65 | 20.08 | 0.184 | 1,916 (99.79%) |

**r16638 (1,888 events)**

| Model | Cuts | Eff [%] | dEff [pt] | Fakes / event | Fake fraction [%] | Splits / event | HS reco |
|---|---|---:|---:|---:|---:|---:|---:|
| v6 | 0.30 / 0.03 | 83.29 | | 18.47 | 18.41 | 0.546 | 1,886 (99.89%) |
| s5 1+1 | matched 0.276 / 0.0276 | 83.16 | **-0.13 ± 0.04** | 18.45 | 18.44 | 0.491 | 1,884 |
| s5 2+2 | matched 0.272 / 0.0272 | 83.39 | **+0.09 ± 0.04** | 18.47 | 18.40 | 0.501 | 1,886 |
| s8 | matched 0.186 / 0.0186 | 83.20 | **-0.09 ± 0.04** | 18.48 | 18.43 | 0.540 | 1,886 |
| AMVF | n_trk >= 2 | 80.13 | -3.16 ± 0.06 | 19.53 | 20.04 | 0.189 | 1,887 (99.95%) |

At the common cuts on r16638, s5 1+1, s5 2+2 and s8 are -0.26, -0.05 and
-0.69 points, with 0.8, 0.9 and 3.0 fewer fakes per event.

- **The common cuts are not a fair comparison for the small models:** they
  run at a stricter operating point, with fewer fakes. At v6's fake rate,
  the matched rows are the comparison.
- **Reverse check:** raising v6's height floor to each small model's fake
  rate gives the same gaps with the opposite sign: +0.19 / +0.13 (s5),
  +0.06 / -0.08 (s5 2+2), +0.21 / +0.14 points (s8).
- **Matched cuts at 0.17 mm** equal those found at 0.1694 mm on 11 November.
  - s5: 0.273 and 0.276;
  - s8: 0.182 and 0.186.
- **HS reconstruction** at 0.1 mm is the same for every model within one
  event.

## 2. Resolution (same peaks as the counting)

sigma_vtx-vtx: pairwise Delta-z sigmoid fit, 300 bins over ±6 mm.
p68: 68th percentile of |z_reco - z_truth| for the optimal one-to-one
assignment within 1 mm. Both differences are paired and in um.

| Model, cuts | r16443 sigma [um] | d sigma | r16638 sigma [um] | d sigma | r16443 p68 [um] | d p68 | r16638 p68 [um] | d p68 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v6, common | 218.2 ± 3.5 | | 214.8 ± 3.6 | | 55.05 | | 54.17 | |
| s5 1+1, matched | 228.4 ± 3.3 | +10.2 ± 1.4 | 226.5 ± 3.3 | +11.6 ± 1.6 | 55.12 | +0.07 ± 0.16 | 54.09 | -0.08 ± 0.14 |
| s5 2+2, matched | 226.8 ± 3.4 | +8.6 ± 1.5 | 224.6 ± 3.3 | +9.8 ± 1.5 | 55.27 | +0.22 ± 0.17 | 54.32 | +0.15 ± 0.16 |
| s8, matched | 217.2 ± 3.4 | -0.9 ± 1.5 | 218.0 ± 3.4 | +3.2 ± 1.6 | 55.60 | +0.55 ± 0.17 | 54.77 | +0.60 ± 0.17 |
| AMVF | 304.7 ± 5.7 | +86.5 ± 1.9 | 298.6 ± 6.0 | +83.7 ± 2.0 | 56.45 | +1.40 ± 0.21 | 55.40 | +1.23 ± 0.21 |

At the common cuts, sigma is +11.0 / +12.2 um for s5 1+1, +9.7 / +10.4 for
s5 2+2 and +4.9 / +8.2 for s8. Pair-density plots are
`compare_*/pair_dz_*.png`.

- **v6 sigma at the single threshold** is 0.218 mm (r16443) and 0.215 mm
  (r16638). This replaces 0.229 mm, which came from the separate 0.50
  threshold.
- **s5 at either schedule separates close vertices worse:** +9 to +12 um,
  4-5%, at 6-7 standard deviations in both files. Training longer (2+2)
  recovers the efficiency but not the separation.
- **s8 at matched cuts is compatible with v6 in sigma:** -0.9 ± 1.5 and
  +3.2 ± 1.6 um, mean +1.2 ± 1.1 um. Its single-vertex precision is 0.6 um
  (1%) worse.

## 3. Against the bar

Bar: an efficiency loss below 0.2 points at a matched fake rate, and no
worse resolution.

| Model | dEff at matched FP (r16443 / r16638) | sigma | p68 | Meets the bar |
|---|---|---|---|---|
| s5 1+1 | -0.18 / -0.13 | +10 / +12 um | equal | no (sigma) |
| s5 2+2 | -0.05 / +0.09 | +9 / +10 um | equal | no (sigma) |
| s8 | -0.14 / -0.09 | equal (+1.2 ± 1.1 um) | +0.6 um | yes (sigma and efficiency); p68 1% worse |

Only s8 meets both parts of the bar. v6 remains strictly the best model:
s8 is 0.09-0.14 points below it, 2-3.5 standard deviations.

## CPU speed

ONNX Runtime, 1 thread, real PU200 events, packed-track inference; from
`cpu_small_model_2026-11-09.md`. s5 2+2 has the s5 architecture.

| Model | Packed [ms / event] | vs v6 padded (1,601 ms) | vs v6 packed (1,159 ms) |
|---|---:|---:|---:|
| v6 | 1,159 | 1.4x | 1.0x |
| s5 | 250 | 6.4x | 4.6x |
| s8 | 178 | 9.0x | 6.5x |

## Still needed for the freeze

- **Step 2b:** vertex categories after GNN v2, with the frozen GNN working
  point. Compare v6 and s8 on r16443 events, against mu.
- **s8 cuts, updated in `working_points.py` to 0.184 / 0.0184:**
  - The two-file derivation gives 0.182 and 0.186; the frozen-candidate
    value is their mean.
  - Check on the saved histograms:

    | s8 cuts | Fakes / event minus v6 (r16443 / r16638) | dEff [pt] (r16443 / r16638) |
    |---|---:|---:|
    | 0.18 / 0.018 (previous value) | +0.09 / +0.24 | -0.13 / -0.06 |
    | 0.184 / 0.0184 (now) | -0.06 / +0.10 | -0.15 / -0.08 |

  - 0.184 matches v6's fake rate on average (+0.02 per event), so the bar
    ("no extra fakes") holds at the frozen value. The efficiency gap is
    -0.15 / -0.08 points.
