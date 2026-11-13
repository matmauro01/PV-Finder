# HL-LHC final numbers at the frozen working point (12 November 2026)

Production recompute of every HL-LHC paper / note / EB number (items H1-H8 of
[paper_questions_2026-11-11.md](../additional_studies/paper_questions_2026-11-11.md),
H8 = everything with s8). PV-Finder vs AMVF on identical events.

**Frozen working point** (`get_regime("hllhc")`, commit 4f04b53): density model
s8 (`hllhc_alleta_v6_s8_1p1ep_phase2_step_450000_fullstate.pth`, sha256
`29e11c4b…3f431`), peaks amplitude 0.01 / width 3 / integral 0.184 / height
0.0184 / local centroid ±3, window 0.17 mm (HS 0.1 mm), GNN v2 epoch 527
(sha256 `aabccb41…f170e`), t = 0.97, sigma_vtx-vtx on the counted peaks.

**Vertex rule after the GNN (decided 12 Nov, coordinator for Mattia):** a
reconstructed vertex counts only with >= 2 tracks, for both algorithms
(PV-Finder: GNN-assigned tracks at t; AMVF: unique `assocTracks`). The old
drop-empty (>= 1 track) convention is kept as labelled backup.

**Samples.** Pre-GNN: r16443 and r16638, first 25,000 entries each; PU200 =
185 <= mu <= 215 (1,920 / 1,888 events). The density model never saw either
file. Post-GNN: r16443 entries 0-24,999 only (r16638 is GNN v2 training data).
All outputs: `outputs/11_12_2026_output/hllhc_final/` (below `F/`).

## Headline table (r16443 PU200, 1,920 events, 213,738 truth vertices with n_trk >= 2)

Errors: event bootstrap (differences paired); sigma: fit error; HS
efficiencies: 68% Clopper-Pearson or bootstrap.

| Quantity | PV-Finder (+GNN) | AMVF | Diff | Source |
|---|---:|---:|---:|---|
| **Before the GNN** | | | | |
| Vertex-finding efficiency, Eq. (1), 0.17 mm [%] | 83.06 ± 0.08 | 79.85 ± 0.09 | +3.20 ± 0.06 | `F/h1_vertex_finding/h1_summary.json` |
| ... r16638 (1,888 events) [%] | 83.22 ± 0.08 | 80.13 ± 0.09 | +3.08 ± 0.06 | same |
| One-to-one efficiency (= finder cap) [%] | 73.36 ± 0.09 | 70.10 ± 0.09 | +3.26 ± 0.06 | same |
| Fake fraction of reconstructed [%] | 18.48 ± 0.09 | 20.08 ± 0.09 | -1.60 ± 0.08 | same |
| Fakes / event | 18.64 ± 0.10 | 19.65 ± 0.10 | -1.01 ± 0.09 | same |
| Splits / event | 0.560 ± 0.016 | 0.184 ± 0.010 | +0.376 ± 0.019 | same |
| sigma_vtx-vtx [mm] (single threshold) | 0.217 ± 0.003 | 0.305 ± 0.006 | -0.087 ± 0.002 | same |
| 68% single-vertex residual [um] | 55.58 ± 0.24 | 56.45 ± 0.24 | -0.87 ± 0.22 | same |
| 3 x 68% [mm] (window derivation) | 0.1667 | 0.1694 | | same (rounds to 0.17) |
| HS reconstruction, 0.1 mm [%] | 99.90 (1,918) | 99.79 (1,916) | +0.10 ± 0.10 | same |
| s8 minus v6, Eq. (1) [pt] (r16443 / r16638) | -0.15 ± 0.04 / -0.08 ± 0.04 | | | `F/s8_vs_v6/compare_*.json` |
| **After the GNN (t = 0.97, >= 2 tracks)** | | | | |
| Vertices | 143,528 | 187,908 | | `F/chain/chain_headline.json` |
| Clean / Merged / Split / Fake | 115,192 / 24,491 / 3,791 / 54 | 66,096 / 80,104 / 41,573 / 135 | | same |
| Clean / truth (matched-vertex yield) [%] | 53.89 ± 0.11 | 30.92 ± 0.11 | +22.97 ± 0.10 | same |
| Clean / reco [%] | 80.26 | 35.17 | +45.08 ± 0.12 | same |
| Merged / reco [%] | 17.06 | 42.63 | -25.57 ± 0.13 | same |
| Split / reco [%] | 2.64 | 22.12 | -19.48 ± 0.10 | same |
| Fake / reco [%] | 0.04 | 0.07 | -0.03 ± 0.01 | same |
| Positional taxonomy (labelled): truth found within 0.17 mm [%] | 73.45 ± 0.09 | 79.85 ± 0.09 | -6.41 ± 0.07 | same |
| Positional taxonomy: fake / reco [%] | 4.78 ± 0.06 | 20.08 ± 0.10 | -15.31 ± 0.09 | same |
| Vertices per event, Clean / Merged / Split / Fake | 60.00 / 12.76 / 1.97 / 0.03 | 34.42 / 41.72 / 21.65 / 0.07 | | `F/supplementary/classes.json` |
| Finder cap (one-to-one, all peaks) [%] | 73.36 | 70.10 | +3.26 | `F/chain_extras/chain_extras_summary.json` |
| Truth-guided = oracle Clean/truth [%] | 68.55 | | | same |
| Eq. (4) baseline, s_max = 3 (best s_max = 2) [%] | 32.26 (32.67) | | | same |
| z0 nearest / z0 within 3 sigma [%] | 31.31 / 32.34 | | | same |
| Track efficiency / purity [%] | 55.98 / 85.55 | 68.32 / 68.65 | -12.34 ± 0.04 / +16.90 ± 0.04 | `F/chain/chain_headline.json` |
| Track F1 [%] | 67.68 | 68.48 | **-0.80 ± 0.03** | same; xcheck |
| Wrong-assignment rate [%] | 14.4 | 31.4 | | `F/chain_extras/...summary.json` |
| Unassigned truth-matched tracks [%] | 34.6 | 0.5 | | same |
| Clean HS vertex [%] | 94.58 ± 0.52 | 92.50 ± 0.61 | +2.08 ± 0.49 | `F/chain/chain_headline.json` |
| HS tracks on the HS-matched vertex [%] | 80.88 ± 0.32 | 74.11 ± 0.33 | +6.77 ± 0.28 | same |
| HS selection, identical Athena 24.0 rule [%] | 95.00 ± 0.52 (1,824) | 89.11 ± 0.69 (1,711) | | same |
| HS selection, AMVF stored primary [%] | | 92.92 ± 0.56 (1,784) | GNN +2.08 ± 0.56 | same |
| HS longitudinal resolution, 3D adaptive fit [um] | 9.48 ± 0.28 | 10.24 ± 0.27 | -0.77 ± 0.24 | same (1,915 common events) |
| **CPU at mu = 200** | | | | |
| s8 packed, ONNX, 1 thread, sneezy (measured) [ms] | 174 (v6 padded 1,892; 10.9x) | | | `F/cpu_timing/cpu_timing.json` |
| GNN v2 forward, torch, 1 thread (measured) [ms] | 98 | | | same |
| Full chain on sneezy (measured, no Athena) [ms] | 277 | not measurable here | | same |
| Athena-equivalent chain (projected) [ms] | 413 (range 336-456) | ~650 (600-750, read off note) | 1.57x faster | same |

Inclusive (25,000 events, mu 0-200): Eq. (1) 83.36 vs 80.10%; Clean/truth
60.12 vs 40.48%; HS selection 96.96 vs 95.14% (stored) / 93.36% (rule);
HS resolution 9.56 vs 9.93 um; track F1 73.27 vs 74.43%.

## Changes against the previous (v6, GNN v4 / t = 0.98) numbers

| | before | now |
|---|---|---|
| Eq. (1) at 0.17 (0.1694) mm | 83.2% (+3.36 pt) | 83.06% (+3.20 pt) |
| sigma_vtx-vtx | 0.229 mm (hidden 0.50 threshold), 0.218 at 0.30 | 0.217 mm |
| 68% residual | 55.05 um | 55.58 um (window still 0.17 mm) |
| Clean/truth | 68.4% (v4, t = 0.98, >= 1 track) | 53.9% (>= 2 tracks); 66.7% with >= 1 track |
| Clean HS / HS tracks / HS purity | 96.4 / 74.9 / 91.3% | 94.6 / 80.9 / 88.2% |
| HS selection | 96.3% (plain sum pT^2, v4) | 95.0% (Athena rule) |
| HS resolution | 9.7 um (v4, t = 0.98) | 9.5 um |
| CPU projection s8 + GNN | 455 ms (9.0x) | 413 ms (10.9x, interleaved) |

## Plots

ATLAS style (mplhep), PNG + PDF, "Simulation Internal", PV-Finder vs AMVF on
the same events. **M** = paper main, **B** = backup. `{tag}` = r16443 / r16638.

**Before the GNN** (`F/h1_vertex_finding/plots/`)
- M `fig6_vertices_vs_mu_{tag}`: reconstructed vertices per event vs mu, with truth (n_trk >= 2).
- M `eff_vs_mu_{tag}`: Eq. (1) efficiency vs mu, 10-unit bins, difference panel.
- M `fake_fraction_vs_mu_{tag}`, `fakes_per_event_vs_mu_{tag}`: fake fraction and fakes per event vs mu.
- M `window_justification_{tag}`: |z_reco - z_true| (one-to-one within 1 mm), 68%, 3 x 68% and 0.17 mm marked.
- M `window_scan_{tag}`: efficiency vs window 0.1-1 mm (Eq. (1), one-to-one, chance matches), difference panel.
- M `paper_figures_r16443/fig4_hllhc_deltaz`: Fig. 7(a), pairwise Delta z with sigmoid fits.
- M `fig7b_residuals_{tag}`: Fig. 7(b), signed and absolute residuals with 68% values.
- B `eff_one_to_one_vs_mu_{tag}`, `splits_per_event_vs_mu_{tag}`: one-to-one efficiency, splits per event vs mu.
- B `paper_figures_r16443/fig3a_hllhc_categories_pu200`, `fig3b_hllhc_categories_vs_mu`, `fig6_categories_vs_mu_amvf_{tag}`: positional categories at PU200 and vs mu.
- M `F/chain_extras/plots/eff_vs_n_trk_pregnn_{pu200,inclusive}`: Eq. (1) efficiency vs truth n_trk (PUB-note style), ratio panel.
- M `F/chain_extras/plots/eff_vs_density_pregnn_{pu200,inclusive}`: Eq. (1) efficiency vs local pile-up density.
- B `F/chain_extras/plots/eff_vs_dz_nearest_pregnn_*`: efficiency vs distance to the nearest truth vertex.
- M `F/resolution_vs_mu/resolution_vs_mu_p68`: 68% residual vs mu (the paper resolution plot).
- B `F/resolution_vs_mu/resolution_vs_mu`, `per_bin_fits.pdf`: with sigma_vtx-vtx (open markers below mu = 100) and the per-bin fits.

**After the GNN** (`F/chain/plots/`, >= 2-track rule; `F/chain/backup_min1/plots/` = >= 1-track backup)
- M `categories_clean_per_truth_vs_mu`: Clean / truth vs mu.
- M `categories_{clean,merged,split,fake}_per_reco_vs_mu`: category fractions of reconstructed vertices vs mu.
- M `hs_reco_eff_vs_mu`: HS reconstruction efficiency (0.1 mm) vs mu.
- M `hs_selection_vs_mu`: HS reconstructed-and-selected, Athena rule, with AMVF stored primary and AMVF same-rule.
- M `hs_resolution_vs_mu`: HS longitudinal resolution vs mu.
- M `track_f1_vs_mu`; B `track_efficiency_vs_mu`, `track_purity_vs_mu`.
- B `hs_clean_fraction_vs_mu`, `hs_track_fraction_vs_mu`, `hs_resolution_vs_local_density` (old density), `positional_eff_vs_mu`, `positional_fake_vs_mu` (positional taxonomy, labelled).
- M `F/supplementary/plots/main/classes_per_event_pu200_primary_min2`: Clean/Merged/Split/Fake per event at PU200 (AMVF-note style).
- M `F/supplementary/plots/main/{clean,merged,split,fake}_per_event_vs_mu_primary_min2`: the same vs mu.
- M `F/supplementary/plots/main/hs_{reco_eff,sel_eff,resolution}_vs_density_itk_min2`: HS reconstruction, selection and resolution vs ITk local density (all truth within ±2 mm of the HS, excluding it, / 4 mm).
- B `F/supplementary/plots/backup/*_weighted_min2`, `classes_per_event_pu200_all_variants_min2`: AMVF classified with fit tracks and fit weights, PV-Finder with GNN scores as weights.
- B `F/supplementary/plots/backup/*_min1`, `*_old*`, `*_pu200*`, `local_density_definitions`, `amvf_fit_weight_distribution`.
- M `F/chain_extras/plots/clean_yield_stages_pu200`: Fig. 7.2 stages (finder cap, truth-guided, GNN, Eq. (4), z0 baselines, AMVF). B `clean_yield_stages_vs_mu`.
- M `F/chain_extras/plots/eff_vs_n_trk_postgnn_{pu200,inclusive}`, `eff_vs_density_postgnn_*`, `clean_per_truth_postgnn_vs_mu`: fraction of truth vertices with a Clean vertex vs n_trk, density, mu.
- M `F/chain_extras/plots/trk_f1_vs_mu`, `trk_eff_vs_pur_scan_{pu200,inclusive}`, `clean_vs_pur_scan_*`: track F1 vs mu; GNN t scan vs Eq. (4) s_max scan, frozen t marked.
- B `F/chain_extras/plots/finder_cap_vs_mu`, `truth_guided_clean_vs_mu`, `eq4_clean_vs_mu`, `z0_baselines_clean_vs_mu`, `stages_vs_{n_trk,density,dz_nearest}_*`.
- B `F/chain_extras/plots/trk_eff_vs_mu`, `trk_pur_vs_mu`, `wrong_rate_vs_mu`, `labelled_unassigned_frac_vs_mu`, `unlabelled_left_unassigned_vs_mu`, `lost_unassigned_frac_vs_mu`, `trackdiag_*` (why the low-mu F1 is lost).
- B `F/chain_extras/plots/sigma_pv_peakwidth_vs_param_*`: the sigma_PV the chain graph uses vs the Appendix B parameterisation.
- B every `F/chain_extras/plots/*_min1`: the >= 1-track convention.

## Quality gates

| Gate | Result |
|---|---|
| Frozen configuration | Pre-GNN evals: `regime_overrides` empty, density from the regime. Chain inputs checked against `get_regime("hllhc")` (cuts, both checkpoint sha256 re-hashed, windows, t) before any number is read. |
| Identical events | Event-index sha1 equal for both algorithms (pre-GNN); chain entry sha256 `c228abeb…` for every post-GNN file. Pre- and post-GNN peaks identical in all 25,000 r16443 entries (max \|dz\| = 0). |
| Category bookkeeping | Asserted per event in every arm, both rules, both taxonomies (chain_t_summary, chain extras, supplementary). |
| Eq. (1) identity | Position-only recomputation equals the matcher's merged-credit numerator in every event (pre- and post-GNN). |
| Regression | `chain_t_metrics --hs-min-tracks 1` bit-identical to the 12 Nov metrics; chain extras and supplementary equal `chain_t_metrics` event by event. |
| AMVF anchors | Efficiency 79.85% (79.854), sigma 0.305 mm (0.3047), 68% 56.45 um, HS reco 1,916/1,920, HS selection stored primary 92.92% (1,784), HS resolution 10.24 um (10.25 anchor): **all reproduced**. |
| Independent recomputation | Separate code from the definitions only (`F/crosscheck/`): peak finding reproduces the stored peaks exactly in all 25,000 events; Eq. (1), fakes, p68, HS reco, all post-GNN categories (both rules), Clean/truth, track efficiency/purity/F1 per mu bin, HS-track and clean-HS fractions, HS selection (1,824) and HS resolution (9.475 / 10.243 um) agree exactly. AMVF same-rule HS selection differs by 1/1,920 (3/25,000) events (tie order). |
| Physics trends | Clean/truth falls with mu, Merged rises; efficiency rises with n_trk to 100%; PV-Finder Eq. (1) flat in density while AMVF falls; HS resolution flat in mu and density; 68% residual rises with mu (below). |

## Things that look odd (not smoothed over)

1. **After the GNN, PV-Finder's positional efficiency falls below AMVF.** With
   >= 2 tracks, 26 candidates per event have fewer than 2 GNN tracks at
   t = 0.97 (19.2 have exactly one). The surviving list finds 73.45% of truth
   within 0.17 mm against AMVF 79.85% (pre-GNN: 83.06%). The Clean/truth lead
   (+23 pt) is unaffected, but "PV-Finder finds more vertices" holds before the
   GNN only.
2. **Clean/truth depends strongly on the vertex rule:** 66.7% (>= 1 track), 63.0%
   (dominant with n_trk >= 2), 53.9% (>= 2 tracks). 7,886 Clean 1-track GNN
   vertices are dominated by n_trk = 1 truth vertices outside the denominator.
   AMVF is 30.9% in all three.
3. **Track F1 is not a GNN win.** 67.7 vs 68.5% at PU200 (>= 2 tracks); 68.8
   vs 68.5% with >= 1 track; at mu 0-10 0.868 vs 0.966. The GNN has higher
   purity (0.856 vs 0.687) and lower efficiency (0.560 vs 0.683). At mu < 10
   the cut t = 0.97 removes 19.5% of tracks the GNN assigns correctly at t = 0
   (all vertex sizes, 40-48% for n_trk 3-7). Without a cut the low-mu F1 equals
   AMVF (0.965 vs 0.966). One PU200-tuned t costs the low-mu track F1.
4. **AMVF sigma_vtx-vtx sits on a fit boundary.** The production fit (unweighted,
   c >= 0) returns c = 4e-14 for AMVF: 0.305 mm. A sqrt(N)-weighted fit gives
   0.298 mm, an unbounded unweighted one 0.282 mm (cross-check). PV-Finder is
   stable (0.2174-0.2177). The AMVF / PV-Finder ratio is 1.30-1.40 depending on
   the fit; quote the 0.305 with this caveat.
5. **The paper's sigma_PV (l. 441) is not what the chain uses.** The GNN edge
   feature and Eq. (4) use the histogram peak width (median 0.072 mm), not the
   Appendix B n_trk parameterisation (agree at n_trk = 2, up to 2x apart above
   10 tracks).
6. **Eq. (4) is no better than nearest z0** (32.3-32.7 vs 31.3-32.3%); the GNN
   gain is the learned assignment plus the score cut.
7. **Small PV-Finder position bias:** median signed residual +0.55 um (AMVF -0.06).
8. **mu < 10:** more reconstructed than truth vertices for both (4.69 / 4.41 vs
   3.57), fake fraction ~31% for both.
9. **HS selection vs ITk density, PU200 bin 1.5-2/mm:** no PV-Finder lead
   (+0.6 ± 1.4 pt) against +3 to +8 pt elsewhere; probably a fluctuation.
10. **The pre-GNN evals were killed by an external SIGTERM at 19:01** (not by any
    agent of this recompute) and rerun from scratch; the killed logs are kept.
11. **D5 datasets table:** counts verified against
    `outputs/11_11_2026_output/datasets/hllhc_dataset_inventory.json` (all six
    pool files, 5,173,600 events, splits). Two rows are stale: r16638 is no
    longer only "held-out evaluation" (GNN v2 trains on 89,820 of its events),
    and the GNN table lacks v2 (149,820 events = 60,000 r16443 entries >= 25,000
    + 89,820 r16638).

## t rescan under the >= 2-track rule (validation, pending decision)

t = 0.97 was chosen with >= 1-track vertices. Rescan on the 7,500 GNN
validation entries (disjoint from the test set), s8 peaks, both algorithms
with >= 2-track vertices (`F/t_rescan_min2/t_rescan_min2.md`, script
`scripts/hllhc_t_rescan_min2.py`). **t is unchanged until decided.**

| t | Clean/truth incl. / PU200 / mu<20 | pos. eff incl. / PU200 | Clean/reco incl. | Split/reco incl. | trk F1 incl. / PU200 / mu<20 | HS sel incl. | HS res incl. [um] |
|---:|---|---|---:|---:|---|---:|---:|
| 0 | 49.11 / 39.41 / 80.84 | 79.18 / 78.04 | 60.41 | 6.76 | 75.23 / 69.26 / 93.92 | 96.04 | 9.38 |
| 0.8 | 50.08 / 40.55 / 81.63 | 79.07 / 77.93 | 61.81 | 6.65 | 75.42 / 69.55 / 93.61 | 96.39 | 9.38 |
| 0.9 | 54.50 / 46.12 / 81.17 | 78.27 / 77.09 | 68.91 | 5.82 | 76.04 / 70.42 / 92.38 | 96.69 | 9.44 |
| 0.92 | 56.15 / 48.18 / 80.27 | 77.76 / 76.60 | 72.01 | 5.32 | **76.06** / **70.49** / 91.66 | **96.87** | 9.44 |
| 0.94 | 57.96 / 50.54 / 79.16 | 76.96 / 75.83 | 75.94 | 4.58 | 75.80 / 70.22 / 90.46 | 96.87 | 9.39 |
| 0.95 | 58.89 / 51.81 / 78.26 | 76.38 / 75.23 | 78.35 | 4.07 | 75.41 / 69.81 / 89.50 | 96.92 | 9.44 |
| 0.96 | 59.72 / 52.86 / 77.03 | 75.58 / 74.38 | 81.07 | 3.45 | 74.70 / 69.04 / 88.17 | 96.89 | 9.53 |
| **0.97** | **60.31** / 53.83 / 74.92 | 74.46 / 73.28 | 84.06 | 2.69 | 73.40 / 67.66 / 86.27 | 96.68 | 9.57 |
| 0.98 | 60.22 / **54.25** / 72.20 | 72.66 / 71.50 | 87.26 | 1.83 | 70.97 / 65.16 / 83.35 | 96.28 | 9.71 |
| AMVF | 40.76 / 30.94 / 76.76 | **80.07** / **79.84** | 43.65 | 21.45 | 74.60 / 68.54 / 94.08 | 95.20 (stored) / 93.67 (rule) | 9.87 |

Statistical errors: Clean/truth ±0.11 (incl.), ±0.22 (PU200), ±0.7 (mu<20);
HS selection ±0.2 (incl.); HS resolution ±0.13-0.17 um.

- **No t makes the post-GNN positional efficiency reach AMVF:** even t = 0
  gives 79.2 vs 80.1% (inclusive), 78.0 vs 79.8% (PU200). With >= 2 tracks
  the loss is intrinsic to dropping 1-track candidates.
- t = 0.97 still maximises inclusive Clean/truth (0.98 is equal within error,
  and best at PU200 by +0.4).
- t = 0.92 maximises track F1 (+1.5 pt above AMVF inclusive, +2.0 at PU200;
  t = 0.97 is 1.2 / 0.9 pt below AMVF) and HS selection, at -4.2 pt
  Clean/truth inclusive (-5.7 at PU200). At mu < 20 every t >= 0.8 is below
  AMVF in track F1; t <= 0.5 matches it.

## Resolution vs mu (s8)

Same events as the v6 study (44,999 events from r16443 + r16638, capped per
bin; selection file copied byte for byte), s8 at the frozen cuts. Fix applied:
`scripts/resolution_vs_mu.py` took its cuts from `HLLHC_PEAKS` (the old v6
0.30 / 0.03); it now uses `get_regime("hllhc")`. AMVF values are identical to
the v6 study in all 132 numbers (asserted). Paper plot: 68% residual vs mu
(`F/resolution_vs_mu/resolution_vs_mu_p68`).

| mu | events | PVF p68 [um] | AMVF p68 [um] | PVF sigma [um] | AMVF sigma [um] |
|---|---:|---:|---:|---:|---:|
| 0-10 | 9,999 | 33.7 ± 0.3 | 32.1 ± 0.3 | fit fails (nan) | (139.0 ± 6.8) |
| 10-20 | 10,000 | 43.5 ± 0.2 | 42.6 ± 0.3 | (101.2 ± 0.3) | (160.0 ± 3.1) |
| 20-30 | 6,000 | 45.9 ± 0.3 | 45.4 ± 0.3 | (101.9 ± 0.5) | (182.5 ± 2.8) |
| 30-40 | 3,000 | 46.8 ± 0.3 | 47.2 ± 0.3 | (117.8 ± 3.0) | (197.5 ± 2.9) |
| 40-60 | 2,000 | 48.5 ± 0.4 | 48.8 ± 0.4 | (141.1 ± 4.0) | (224.6 ± 3.5) |
| 60-80 | 2,000 | 49.3 ± 0.3 | 49.9 ± 0.3 | (169.4 ± 2.3) | (244.3 ± 2.8) |
| 80-100 | 2,000 | 50.2 ± 0.3 | 51.2 ± 0.3 | (184.4 ± 2.2) | (263.6 ± 2.4) |
| 100-120 | 2,000 | 51.1 ± 0.2 | 52.1 ± 0.2 | 196.3 ± 2.2 | 275.1 ± 2.1 |
| 120-140 | 2,000 | 52.0 ± 0.2 | 53.1 ± 0.3 | 207.0 ± 2.1 | 287.8 ± 1.9 |
| 140-160 | 2,000 | 53.4 ± 0.2 | 54.8 ± 0.2 | 209.5 ± 1.9 | 294.4 ± 1.8 |
| 160-180 | 2,000 | 54.4 ± 0.2 | 55.4 ± 0.2 | 213.9 ± 1.6 | 298.6 ± 1.7 |
| 180-200 | 2,000 | 55.5 ± 0.2 | 56.5 ± 0.2 | 216.9 ± 1.7 | 304.3 ± 1.6 |

sigma_vtx-vtx in parentheses (mu < 100) is not quoted: the close-pair excess
exceeds the dip there (resolution_vs_mu_2026-11-11.md). The 180-200 bin agrees
with the PU200 headline (0.217 / 0.305 mm; 55.6 / 56.5 um). The s8 sigma fit
fails at mu 0-10 (v6: 58 um; the dip is not separable from the notch).
PV-Finder p68 is 0.4-1.4 um below AMVF above mu = 30 and 0.9-1.6 um above it
at mu < 20. s8 minus v6 on the same events: p68 +0.3 to +0.8 um at every
mu >= 10; sigma +0.7 to +3.4 um for mu >= 100 (each within ~2 sigma); s8
chi2/ndf 15-30 for mu >= 120 (v6 7-19). Details:
[resolution_vs_mu_2026-11-12_s8.md](resolution_vs_mu_2026-11-12_s8.md).

## Reproduction

```bash
# tmux, repo root, venv, PYTHONPATH=src, ulimit -v 33554432
TAG=r16443 DEVICE=0 bash outputs/11_12_2026_output/hllhc_final/pre_gnn/run_pre_gnn.sh   # and r16638
EVAL_BASE=outputs/11_12_2026_output/hllhc_final/pre_gnn/s8 OUT=outputs/11_12_2026_output/hllhc_final/h1_vertex_finding \
  GNN_CHUNKS=outputs/11_12_2026_output/gnn_v2_freeze/infer/test_s8 bash scripts/run_h1_vertex_finding.sh
python -u -m gnn.evaluation.chain_t_metrics --infer outputs/11_12_2026_output/gnn_v2_freeze/infer/test_s8 --gnn v2e527 \
  --hs-min-tracks 2 --out outputs/11_12_2026_output/hllhc_final/chain/metrics/test_s8_v2e527_hsmin2.npz
python -u scripts/hllhc_final_chain.py --metrics .../test_s8_v2e527_hsmin2.npz --vertex-min-tracks 2 --out .../chain
python -u scripts/hllhc_final_chain_extras.py ...; scripts/hllhc_final_chain_extras_{summary,plots,trackdiag}.py   # see F/chain_extras/README.md
python -u scripts/hllhc_final_supplementary.py ...; scripts/hllhc_final_supplementary_plots.py   # see F/supplementary/README.md
python -u scripts/resolution_vs_mu.py {select,infer,analyze,plot} --out F/resolution_vs_mu   # see its README
bash F/cpu_timing/run_interleaved.sh; python -u scripts/hllhc_cpu_timing_summary.py   # see F/cpu_timing/README.md
python scripts/eb_numbers.py --out presentations/mattia/11_13_2026_EB/numbers.tex && bash presentations/mattia/11_13_2026_EB/build.sh
```

Every result JSON records the git hash, checkpoint sha256s and
`get_regime("hllhc").to_dict()`. The code was committed while the runs were
in progress, so the JSONs carry several hashes (f7e8a67 ... 6c2cbe2) with a
dirty flag; no working-point code changed in between.
