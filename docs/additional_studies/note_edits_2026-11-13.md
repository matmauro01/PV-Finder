# Proposed internal-note edits, 13 November 2026

Note: ANA-IDTR-2026-01-INT1, branch `mm-edits-2026-11-13` (from `origin/main`
91b6f49), Draft MR into `main`:
https://gitlab.cern.ch/atlas-physics-office/IDTR/ANA-IDTR-2026-01/ANA-IDTR-2026-01-INT1/-/merge_requests/1
(commits be5bf7d, 08ceda2).
Built PDF (current branch head): `presentations/mattia/11_13_2026_paper/note_with_MM_edits_2026-11-13.pdf`.

**Convention.** Added text is red, `[MM: ...]` (macro `\mmedit`). Text proposed
for removal is kept, struck through in red (macro `\mmdel`), followed by its
replacement in `\mmedit`. Both macros are in `ANA-IDTR-2026-01-INT1-defs.sty`.
Professors' comments (`\lat`, `\rbg`, `\mm`) are untouched. New figures are
added next to the old ones (old figures kept, captions marked); no file moved.

Page and line numbers below are those of the PDF built from the base commit
91b6f49 (72 pages; draft line numbers as printed in the margin). "Caption"
means an unnumbered caption on that page.

## Revision 2: Mattia's comments (13 Nov 2026, commit 08ceda2)

Page and line numbers in this section are those of the current PDF (75 pages).
The tables further down keep the base-PDF numbers of the first round.

1. Close-pair fraction recomputed on CPU from the r16443 PU200 truth list
   (`fair_r16443/cache_r16443.npz`): 12.8% of adjacent truth pairs are closer
   than 0.217 mm (13.45% at 0.229 mm and 17.2% at 0.30 mm reproduce the old
   values). The intro keeps "about 13%" with 0.217 mm and no annotation;
   Sec. 4.3.1 says 12.8%.
2. Table 2.2 simplified: dataset rows removed, "idem" where the low pile-up
   value equals HL-LHC.
3. and 4. Dataset-role annotations in Sec. 3.2 and Sec. 4.1 removed.
5. "Vertex-finding model" instead of "density model" (the only instance was in
   my added text; the original text does not call the network a density model).
6. Figures replaced in place by s8 versions (old files kept in `figures/`):
   - Fig. 4.1(b) `mm_hs_efficiency_stored_density_s8.pdf`, replotted on CPU
     from the s8 peaks of all 25,000 events (24,968 HS found, 99.872%; the
     PU200 subset gives the headline 1,918). Panel (a) is truth only, unchanged.
   - Fig. 4.2 `mm_fig6_vertices_vs_mu_r16443.pdf`.
   - Fig. 4.3 `mm_deltaz_pu200.png` (EB deck): 0.217 ± 0.003 vs 0.305 ± 0.006 mm.
   - Fig. 4.4 `mm_fig7b_residuals_r16443.pdf`: 55.58 ± 0.24 vs 56.45 ± 0.24 um,
     as in the text.
   - Fig. 4.6 `mm_window_scan_r16443.pdf` (ATLAS style, efficiency panels only;
     fake numbers in the text).
   - Fig. 6.1 (a) `mm_fig8a_run2mc_deltaz.pdf`, (b) `mm_mc_category_counts_e400.pdf`
     (new CPU replot, `scripts/note_lowpu_category_counts.py`, 373 events at
     55 <= mu <= 65), (c) `mm_fig8b_run2mc_yield_vs_mu.pdf`.
   - Fig. 6.2 panels now the e400 versions with Run 2 MC overlaid (the separate
     Fig. 6.3 of round 1 is gone).
   - Fig. 7.2 now the s8 / t = 0.94 bar chart with the z0 baselines (TikZ kept
     as comments; the round-1 Fig. 7.3 is gone).
   - Fig. 7.4 (b)-(d) replotted at t = 0.94 on CPU from the saved low pile-up
     records (`scripts/lowpu_gnn_t_options.py` with t = 0.94,
     `outputs/11_13_2026_output/note_edits/lowpu_t094/`); (a) unchanged (same GNN).
   - Checked and kept: Fig. 2.2 (same file as in the EB deck), Fig. 2.3
     (resolution model, still used), Fig. 2.4-2.6 and 7.1 (schematics), Fig. 5.1
     (earlier-model GPU timing, labelled as such), Fig. 5.2-5.3 (Athena),
     Fig. 7.3 (HS resolution, t = 0.94).
7. Sec. 4.1.2 updated with existing s8 numbers (`fair_r16443/results_r16443.json`):
   margin +3.0 to +3.3 pt for 0.10-0.20 mm; at 0.5 mm +0.75 ± 0.05 (merged) and
   +2.90 ± 0.06 (one-to-one); fake+split fewer below about 0.63 mm; matched
   multiplicity (height 0.035, 97.87 per event): +2.74 ± 0.06 pt and
   3.13 ± 0.09 fewer fake+split; chance matches 18.7 / 18.4% at 0.17 mm and
   45.6 / 46.7% at 0.5 mm. The fake+split column of Table 4.1 is restored
   (19.20 vs 19.84, -0.63 ± 0.09) and the abstract, intro and conclusions use it.
8. CPU projection: the AMVF comparison and the packed/padded caveat removed
   (Sec. 5.1.2 and the conclusions).
9. and 10. Sec. 6 checked against `lowpu_final_numbers_2026-11-12.md`: every
   number agrees; the bootstrap count is 2,000 (struck 10,000).

### Still waiting (current PDF)

On the final recompute with the 13 Nov s8 cut (integral 0.186, no height):
- p. 1 l. 16-17 (abstract), p. 14 Table 2.2 caption, p. 19 l. 399 (Sec. 4.1),
  p. 48 l. 988 (conclusions): every pre-GNN HL-LHC number in Chapter 4
  (Table 4.1, Figs. 4.1b-4.6, Sec. 4.1.1-4.1.2).
- p. 52 l. 1050 (Appendix A settings matrix and commands, earlier configuration).
- p. 64 l. 1400 (Appendix D, earlier model; not updated).

On the GNN v3 retrain:
- p. 36 l. 676: the sentence announcing it; all of Chapter 7 (pp. 36-47) and
  the GNN numbers of the conclusions, p. 48 l. 1004.
- p. 29-30 l. 587-590: GNN CPU time (98 ms) in the CPU projection.
- p. 41 l. 832: GNN median dominant-track fraction (100%, GNN v4).
- p. 46 l. 933: Sec. 7.7 (GNN v4 augmentation study), kept as a record.

Done in round 3 (below): p. 41 l. 829 and p. 44 l. 891-907.

Left for Mattia:
- p. 26 l. 490-495: Sec. 4.3.2 height-cut and GBT studies, earlier model; with
  no height cut in the frozen configuration they could be cut or moved to the
  appendix.

Other:
- p. 28 Fig. 5.1 and p. 29 l. 581: timing of the earlier model; the frozen
  chain has no Athena timing yet (the CPU number is a projection).

## Round 3: the two CPU recomputations (13 Nov 2026)

Current PDF page/line. CPU only, from saved inputs; scripts committed.

- **p. 41 l. 829, truth vertices without a one-to-one peak** (s8, r16443 PU200,
  greedy closest-first within 0.17 mm, `scripts/note_unmatched_truth_ntrk.py`):
  56,944 of 213,738 unmatched (one-to-one 73.36%, equal to the finder cap);
  median n_trk 3 (unchanged) against 10 for matched (was 9); 33.5% have exactly
  two tracks (was 34.8%). ROOT truth equals the cache in all 1,920 events.
- **p. 44 l. 892-907, low pile-up Truth+GNN** (mu60 v2 e115 on the July truth
  graphs `data/mu60/ttva_graphs/mu60_truth_test_fixedheights.pt`,
  `gnn.evaluation.lowpu_chain -d -1 --min-tracks 2`, t = 0.94): 67,545
  vertices; Clean / Merged / Split / Fake 83.42 ± 0.20 / 12.01 ± 0.16 /
  4.57 ± 0.09 / 0.003%; Clean/truth 78.06 ± 0.22% (was 89.2% at t = 0.98 with
  >= 1 track). Regression: the same graphs with >= 1 track at t = 0.98 give 72,544 vertices, 88.76 / 7.40 / 3.83%, Clean/truth 89.20%, the old row exactly.
- **p. 44 l. 896, oracle on the frozen peaks** (`scripts/note_lowpu_oracle.py`,
  e400 integral 0.50 peaks, each labelled track to the peak nearest its truth
  vertex within 0.17 mm, >= 2 tracks): Clean/truth 73.90 ± 0.19% (was 81.9% at
  1 mm on the integral-0.2 peaks, >= 1 track). The GNN (70.57%) is 3.3 pt below.
  Sanity: below the one-to-one finder match (76.24%).
- **p. 44 l. 901**: Clean fraction of reconstructed vertices 85.1% (chain)
  against 83.4% (truth graphs), so the sentence "slightly higher for the
  deployed chain" still holds.

## Numbers: what is final and what is flagged

- **GNN-dependent numbers** (all of Chapter 7, the conclusions, the CPU GNN
  time) are written as final text with the frozen GNN v2 e527, t = 0.94,
  vertices with >= 2 tracks for both algorithms. One red sentence where the GNN
  is introduced (Section 7.1) says an updated GNN is being trained and the
  numbers will be updated.
- **Pre-GNN HL-LHC numbers** (s8) come from
  `docs/evaluation/hllhc_final_numbers_2026-11-12.md`, computed with the 12 Nov
  cuts (integral 0.184, height 0.0184). The frozen cuts are now integral 0.186,
  no height cut; the consolidation study finds no measurable change (efficiency
  -0.00 ± 0.00 pt, fakes +0.01 ± 0.01 per event). These are flagged "to confirm
  with the final recompute" once each: in the abstract, at the start of
  Section 4.1, in the caption of the frozen-configuration table and in the
  conclusions.
- **Low pile-up numbers** are final (`lowpu_final_numbers_2026-11-12.md`). The
  chain numbers at t = 0.94 are option B of
  `outputs/11_12_2026_output/lowpu_final/gnn_mc_ntrk2/t_options.md` (the doc
  tables are at the old t = 0.97). The Run 3 association at t = 0.94 did not
  exist; it was run for this edit (CPU, see "Run 3 at t = 0.94" below).
- **Not recomputed, marked in red as such** (no number invented): the median n_trk of unmatched truth
  vertices, the GNN median dominant fraction (100%, GNN v4), the low pile-up
  Truth+GNN row and the 81.9% oracle, and all of Section 7.7 (GNN v4
  augmentation study) and Appendices A and D (earlier model, one red note each).

## Audit of the note against the docs (before editing)

The note at 91b6f49 is one model and one window behind the docs:

| Item | Note | Docs (frozen) |
|---|---|---|
| HL-LHC vertex-finding model | v6, 3.55M parameters, 280 channels, 4 latent | s8, 439,593, 80 channels, 1 latent |
| HL-LHC peak cuts | integral 0.30, height 0.03; 0.50 for the resolution | integral 0.186, no height cut, one threshold |
| Window | 0.1694 mm (HL-LHC); 0.416 mm and per-sample sigma (low pile-up) | 0.17 mm everywhere; HS 0.1 mm |
| HL-LHC efficiency | 83.16 vs 79.80% (+3.36) | 83.06 vs 79.85% (+3.20 ± 0.06) |
| sigma_vtx-vtx | 0.229 mm (at 0.50) | 0.217 ± 0.003 mm |
| 68% residual | 55.05 um | 55.58 ± 0.24 um |
| Low pile-up model | Run 2 reproduction (KDE pretraining), ep300 | e400, two-phase, no KDE |
| Low pile-up MC | 91.59% at 0.416 mm, AMVF sigma on other events | 81.55 vs 75.74% at 0.17 mm, same events |
| GNN (HL-LHC) | v4, t = 0.98, >= 1 track, 0.5 mm one-to-one | v2 e527, t = 0.94, >= 2 tracks, 0.17 mm |
| Clean/truth PU200 | 68.4% | 50.6% (AMVF 30.9%) |
| HS selection | 94.4 vs 80.7% | 95.05 vs 92.92% (AMVF stored primary) |
| Low pile-up chain | t = 0.99, integral 0.2, 77.43% | t = 0.94, integral 0.50, 70.57% |
| AMVF fit weights | "not stored in the ntuples" | stored in Trey's supplementary ntuple |
| Timing | "faster at all pile-up" (CPU and GPU figure) | faster on GPU only; CPU is a projection |

All these are corrected in red where they appear in the main chapters. The
lower Clean/truth (68.4 -> 50.6%) is the >= 2-track rule, t and the 0.17 mm
window, not a regression; the earlier 68.4% was with >= 1-track vertices.

## The edits

Old text is abbreviated with "...". Locations: page, line in the base PDF.

### 1. 0.17 mm window everywhere, justification plot, HS at 0.1 mm

| Where | Old | New |
|---|---|---|
| Abstract, p. 1, l. 11-14 | 0.1694 mm; 83.16 / 79.80%; 0.60 fewer fake+split; 0.229 mm | 0.17 mm; 83.06 / 79.85%; 0.63 ± 0.09 fewer fake+split; 0.217 ± 0.003 mm; flag |
| Sec. 1, p. 6, l. 118-124 | 3.36 ± 0.06 at 0.1694 mm; 0.60 fewer fake+split; 0.229 mm | 3.20 ± 0.06 at 0.17 mm; 0.63 fewer fake+split; 0.217 mm ("about 13%" stays: 12.8% at 0.217 mm); one sentence on s8 |
| Sec. 2.5.2, p. 16, l. 298 | (window paragraph) | added: 0.17 mm for every vertex-matching result in both regimes, 3 x 68% at PU200, Fig. 4.5; "For consistency with ITk performance studies, which quote only the hard-scatter vertex, we also give the hard-scatter result with a 0.1 mm window." |
| Sec. 4.1, p. 19, l. 364 | "0.1694 mm, or about 0.17 mm" | 0.17 mm = 3 x 55.6 / 56.5 um = 0.167 / 0.169 mm, rounded |
| Sec. 4.1, p. 19, l. 367-370 | 83.16 / 79.80 / 3.36; 0.60 fewer fake+split | 83.06 / 79.85 / 3.20 ± 0.06; 0.63 ± 0.09 fewer fake+split (19.20 vs 19.84); separately fakes 18.64 vs 19.65 (-1.01 ± 0.09), fake fraction 18.48 vs 20.08%, splits 0.56 vs 0.18; r16638 83.22 vs 80.13% (+3.08); flag |
| Sec. 4.1, p. 19, l. 376 | HS 0.1 mm paragraph | added the ITk-consistency sentence |
| Table 4.1, p. 20, caption | 0.1694 mm; 83.16 / 79.80; fake+split 19.27 / 19.87 | 0.17 mm; 83.06 / 79.85; fakes and splits separately |
| Sec. 4.1.1, p. 21, l. 393 | 0.229 ± 0.003 mm, fit at integral 0.50 | 0.217 ± 0.003 mm on the same peaks; AMVF fit at bound c = 0 (0.298 weighted, 0.282 unbounded) |
| Sec. 4.1.1, p. 21, l. 407 | 55.05 ± 0.23 / 56.45 ± 0.26 um | 55.58 ± 0.24 / 56.45 ± 0.24 um |
| Sec. 4.1.1, p. 22, l. 413 | "rounded to four decimal places" | rounded to 0.17 mm; 1 mm radius must stay wider than the window; low pile-up 0.17 mm = about 2.4 x 68% |
| New Fig. 4.5 (after Fig. 4.4, p. 23) | | `mm_window_justification_r16443.pdf`: single-vertex residual, 68%, 3 x 68%, 0.17 mm marked |
| Figs. 4.2, 4.3, 4.4 captions, p. 21-23 | | marked "earlier model; to be replaced by the s8 version" (s8 plots exist in `F/h1_vertex_finding/plots/`) |
| Sec. 4.1.2, p. 22, l. 421 | one-to-one 73.43 / 70.07, +3.37 | 73.36 / 70.10, +3.26 ± 0.06 |
| Sec. 4.1.2, p. 22, l. 424 | window scan (v6) | note: rest of section is v6 at 0.1694; s8 margin +3.0 to +3.3 pt for 0.10-0.20 mm |
| Sec. 6, p. 30, l. 546 | per-sample windows | local centroid, 0.17 mm = about 2.4 x 68% (72.5 / 66.8 um) |
| Sec. 6.1, p. 30, l. 554 | 0.416 mm sigma, 91.59%, 0.419 fakes, 0.416 mm window | 0.392 ± 0.009 mm, 81.55 ± 0.14%, 2.363 ± 0.040, 0.17 mm |
| Sec. 6.2, p. 30-32, l. 565-576 | 0.368 / 0.348 mm; 95.31 / 94.78%; 1.821 / 1.959 | 0.356 / 0.333 mm (AMVF 0.808 / 0.813); 94.39 / 94.03%; 3.006 / 3.192 |
| Table 6.1, p. 32 | ep300 values | e400 at 0.17 mm, plus a Run 2 simulation row with AMVF as reference and AMVF truth efficiency |
| Sec. 7.5, p. 38, l. 718-722 | one-to-one and truth-guided at 0.5 mm, 78.1% | 0.17 mm, 73.36% (AMVF 70.10%) |

### 2. Split-inclusive efficiency

| Where | New (one sentence) |
|---|---|
| Sec. 2.5.2, p. 16, l. 297 | "Split vertices are included in this efficiency: a truth vertex inside the window of a split vertex counts as found. This changes no number, because every such truth vertex already has its own one-to-one match, so Eq. (2.5) equals the fraction of truth vertices with at least one reconstructed vertex within w." |

### 3. HS longitudinal resolution (paragraph + figure)

| Where | New |
|---|---|
| Sec. 7.5, p. 40, after l. 742 | Paragraph: 3D adaptive vertex fit with beam-spot constraint on the GNN-assigned tracks of the HS candidate, seeded at the peak; ITk definition (HS within 0.1 mm, iterative 3-RMS of z_reco - z_HS). 9.52 ± 0.26 um vs AMVF 10.24 ± 0.27 um on 1,915 common PU200 events (ITk ~10 um); flat in mu and density. |
| New Fig. 7.3 | `mm_hs_resolution_vs_mu.pdf` (t = 0.94) |

### 4. Timing claim restricted to GPU, CPU projection

| Where | Old | New |
|---|---|---|
| Sec. 5.1.2, p. 28, l. 534 | "PV-Finder is faster at all pile up ranges, and roughly 9 times faster at mu = 200." | On the GPU faster at all pile-up and ~9x at mu = 200; on the CPU the timed (v6, padded) model is slower than AMVF at PU200 (2.4 s vs ~0.65 s); claim holds on GPU only. |
| After Table 5.1, p. 29 | | CPU projection paragraph: s8 packed ONNX 174 ms (10.9x the timed model), GNN 98 ms, measured chain 277 ms (EPYC 7402, 1 thread); projected Athena chain 413 ms (336-456). (The AMVF comparison and the packed/padded caveat were removed on Mattia's request.) |
| Sec. 8, p. 45, l. 840 | "complete production chain still needs to be timed" | GPU ~9x; CPU timed model slower; CPU projection 413 ms |

### 5. Simulation on the Fig. 6.2 panels, AMVF-reference bias

| Where | New |
|---|---|
| Sec. 6.2, p. 32, after l. 581 | Paragraph "AMVF as reference": data numbers are agreement with AMVF; Run 2 simulation with AMVF as reference gives 93.09 ± 0.12% and 2.998 ± 0.056 per event, against 94.4 / 94.0% and 3.0 / 3.2 in data and 81.55% / 2.363 against truth. The gap is the choice of reference, not data/MC. AMVF reference counts a vertex PV-Finder finds and AMVF misses against PV-Finder. Pairwise-dz shapes agree (chi2/ndf 202/200, 197/200). No Run 3 MC. |
| Fig. 6.2 caption, p. 33 | points to the replacement |
| Fig. 6.2 (replaced in place) | `mm_fig10{a,b,c,d}_*.pdf`: Run 2 / Run 3 data with Run 2 simulation overlaid, data/sim ratio, pile-up distributions |
| Fig. 6.1 caption, p. 31 | marked "earlier model and 0.416 mm window; to be replaced" |

### 6. z0 baselines in Fig. 7.2

| Where | Old | New |
|---|---|---|
| Sec. 7.5, p. 38, l. 726 | truth-guided 73.7%, GNN 68.4% at t = 0.98; steps 21.9 / 4.4 / 5.2 | 68.55%, 50.57% at t = 0.94; steps 26.6 / 4.8 / 18.0; baselines: nearest z0 31.3%, within 3 sigma(z0) 32.3%, significance 32.3% (32.7% at s_max = 2), AMVF 30.9% (= nearest z0 in every mu bin within 0.4 pt), GNN without cut 39.4%; gain = learned assignment (+8.5) + score cut (+11) |
| Fig. 7.2 (replaced in place, TikZ kept as comments) | GNN v4 / 0.5 mm bars | `mm_clean_yield_stages_pu200.pdf` |

### 7. AMVF HS selection 92.9% and the HS numbers

| Where | Old | New |
|---|---|---|
| Sec. 7.5, p. 40, l. 742 | "best result at t = 0.95 ... 94.4% of events, compared with 80.7% for AMVF" | ITk definitions. HS reconstructed (0.1 mm): 99.90 / 99.79%. Reconstructed and selected: PV-Finder+GNN 95.05 ± 0.53%, AMVF stored primary 92.92 ± 0.56%, +2.1 ± 0.5 pt; stored primary = largest sum-pT^2 vertex of AMVF fit tracks in 25,000 / 25,000 events; ITk ~92-93%; inclusive 97.1 vs 95.1%; same rule on TTVA lists 89.1% (not used). HS tracks on HS vertex 87.9 vs 74.1%; HS purity 82.3 vs 88.4%; Clean HS 88.3 ± 0.8 vs 92.5 ± 0.6% (-4.2 ± 0.6); 94.6% at t = 0.97. |
| Sec. 8, p. 45, l. 843 | 94.4 vs 80.7% | 95.1 vs 92.9%, resolution 9.5 vs 10.2 um, Clean HS 88.3 vs 92.5% |

### 8. AMVF fit weights

| Where | Old | New |
|---|---|---|
| Sec. 7.4, p. 37, l. 666 | "these weights ... are not stored in the ntuples used here" | Headline uses AMVF TTVA links (Nonprompt_Hard_MaxWeight), what analyses use. Supplementary ntuple has fit tracks and weights: 23.8% of tracks in no AMVF fit, TTVA attaches 99.5%. With its own fit tracks and weights AMVF has 46.3 Clean per event instead of 34.4, Clean/truth ~41.6% instead of 30.9%; still ~9 pt below PV-Finder+GNN (50.6%; 50.7% with GNN-score weights). Headline keeps TTVA. |
| Sec. 8, p. 45, l. 843 | | adds "41.6% for AMVF with its own fit weights" |

### 9. Frozen configuration table and datasets

| Where | New |
|---|---|
| New Table 2.2 (Sec. 2.2.4, after Table 2.1, p. 13) | One column per regime, "idem" where equal. Vertex-finding model (s8: U-Net v2, 80 ch, MLP 128 x 5, 439,593 params; e400: U-Net v1, 64 ch, MLP 100 x 5, 358,713 params); peak finding 0.01 / width 3 / integral 0.186 (0.50) / no height / centroid ±3; 0.17 mm, HS 0.1 mm; GNN v2 e527 (mu60 v2 e115); t = 0.94; >= 2 tracks. Caption flags the HL-LHC cut change. Dataset rows removed on Mattia's request. |
| Sec. 2.2.4, p. 12, l. 239 | Old: integral 0.30 + height 0.03, 0.50 for the resolution (struck). New: one configuration per regime; regimes differ only in model and integral; no height cut; sigma on the same peaks; cut values not portable between models; HL-LHC cut set at v6's fake rate, 0.50 the Run 2 convention; 13 Nov consolidation (0.186 alone = 0.184 + 0.0184 within 0.01 pt and 0.02 fakes per event in every mu bin). |
| Table 2.1 caption, p. 13 | "Superseded by Table 2.2" |
| Sec. 2.3, p. 13, l. 254, Table 2.3 | s8 replaces v6: architecture, 439,593 parameters, 1 + 1 epochs (step 450,000 of 465,624); s8 minus v6 -0.15 ± 0.04 / -0.08 ± 0.04 pt |
| Sec. 4.1, p. 19, l. 371-374 | integral 0.30 / height 0.03 / 0.50 for resolution -> frozen configuration, same peaks |
| Sec. 4.3.2, p. 25, l. 450 | height 0.03 -> no height cut; s8 consolidation result |
| Sec. 6, p. 30, l. 540 | Run 2 reproduction model with KDE pretraining -> e400, two-phase, no KDE, for every low pile-up result |
| Sec. 7.4, p. 38, l. 699 | "vertices with no assigned tracks are removed" -> fewer than two tracks, both algorithms, tracks unassigned |
| Table 7.1, p. 36 | PU200 column: v4 (180,000 all-hadronic, 108 epochs, 6 passes) -> v2 (149,820 flat-mu ttbar, 7,500 validation, 30 files, 540 epochs, checkpoint 527, 18 passes) |
| Appendix A, p. 49, l. 891; Appendix D, p. 61, l. 1238 | one red note each: earlier configuration, not updated |

### GNN chapter numbers (t = 0.94, >= 2 tracks)

| Where | Old | New |
|---|---|---|
| Sec. 7.1, p. 34, l. 597 | | "An updated GNN, trained on PV-Finder candidates instead of truth vertices, is being trained. The GNN results in this chapter use the frozen GNN v2 (epoch 527, t = 0.94) and will be updated with the new model." |
| Sec. 7.5, p. 38, l. 708-710 | 193,945 peaks, t = 0.98, 83.16% | ~101 per event, t = 0.94, 83.06% |
| Table 7.2, p. 40 | AMVF 187,960 / 66,127; GNN 175,491 / 146,269 / 83.3% / 68.4% | AMVF 187,908 / 66,096 / 35.2% / 30.9%; GNN 152,795 / 108,086 / 70.7% / 50.6% |
| Sec. 7.5, p. 40, l. 735 | "AMVF produces slightly more vertices" | "23% more" |
| Sec. 7.5, p. 40, l. 735 | | Merged 24.6 vs 42.6%, Split 4.6 vs 22.1%; track eff 62.9 vs 68.3%, purity 79.5 vs 68.7%, F1 70.2 vs 68.5% |
| Sec. 7.5, p. 40, l. 739 | 303 fakes of 175,491 (0.17%); 9.67% with empty peaks | 21.3 candidates/event with < 2 tracks removed (17.8 with one); 69 of 152,795 (0.05%) vs AMVF 135 of 187,908 (0.07%) |
| Sec. 7.6, p. 40, l. 748-757 | t = 0.99, 55,897 Clean, 77.4 vs 60.9%; 61,845 vertices, 13 fakes, 4,627 empty; Merged 7.4 vs 20.2% | t = 0.94 (HL-LHC value; best HS-ID, best F1 within 0.01, -1.4 pt from best Clean/truth 71.94% at 0.985); 50,946 Clean, 70.57 ± 0.25 vs 60.47 ± 0.28%; per mu bin 87.4 / 79.1 / 70.3 / 63.8 vs 84.2 / 71.6 / 59.4 / 52.0%; 59,874 of 61,102 peaks kept, 1 fake, 1,228 removed (883 one track, 345 none); Merged 13.1 vs 20.4% |
| Sec. 7.6, p. 41, l. 769 | AMVF 398 empty, 56,931 left | 398 empty and 593 one-track removed, 56,338 left |
| Table 7.3, p. 41 | 61,845 / 90.38 / 7.42 / 2.18 / 0.021 / 77.43; AMVF 56,931 / 77.23 / 20.19 / 2.53 / 0.054 / 60.91 | 59,874 / 85.09 / 13.12 / 1.79 / 0.002 / 70.57; AMVF 56,338 / 77.48 / 20.40 / 2.11 / 0.004 / 60.47; 1,000 bootstrap samples |
| Sec. 7.6, p. 41, l. 772 | F1 0.874 at t = 0.95 vs 0.849 | 87.02 ± 0.11 vs 84.86 ± 0.12% (eff 85.6 vs 83.9, purity 88.5 vs 85.9) |
| Sec. 7.6, p. 41, l. 777 | HS-ID 98.98 ± 0.20 vs 98.75 ± 0.22 | 99.10 ± 0.18 vs 98.75 ± 0.21; AMVF stored primary 98.31% |
| Sec. 7.6.1, p. 41, l. 781-784 | 2,000 events, t = 0.98, 1 mm: 83.4% / 95.7% / 83.7% / 88.7% | the same 2,500 events and peaks as Ch. 6, t = 0.94, >= 2 tracks: GNN assigns 95.8% (AMVF 94.8%); agreement of both-assigned tracks 94.5% at 0.17 mm (95.5% at 1 mm), 92.9% (93.9%) of AMVF-assigned; leading vertex 91.3%; 28.0 vs 26.9 vertices per event |
| Sec. 7.7, p. 43, l. 789 | | note: GNN v4 study, kept as record |
| Sec. 8, p. 45, l. 831-843 | 0.42 vs 0.79 mm; 83.16 etc.; 68.4 vs 30.9%; 73.7% at 0.5 mm | 0.39 vs 0.79 mm and 81.6 vs 75.7%; 83.06 etc.; 50.6 vs 30.9% (41.6% AMVF fit weights); 68.5% at 0.17 mm |

## Run 3 at t = 0.94

The low pile-up doc gives the Run 3 association at t = 0.95, 0.97, 0.98 and
0.99 only. It was rerun with the production code (`gnn.evaluation.run3_assoc_lowpu`,
t grid extended by 0.94 through a wrapper, CPU, 4 threads, 20 min, tmux),
same inputs as `lowpu_final/run3_assoc_ntrk2`. Gates passed (peaks and AMVF
lists equal the R3 pickle in 2,500 / 2,500 events, AMVF self-check 1). The
t = 0.97 and 0.95 rows reproduce the GPU production numbers exactly
(933,361 / 942,705 agreeing tracks, 980,612 both-assigned at 0.97).

| t | GNN assigned | AMVF assigned | agree / both (0.17 / 1 mm) | agree / AMVF (0.17 / 1 mm) | leading vertex | 1-track GNN vertices dropped / event |
|---|---:|---:|---:|---:|---:|---:|
| 0.94 | 95.78% | 94.83% | 94.47 / 95.46% | 92.94 / 93.92% | 91.28% | 0.41 |
| 0.97 (check) | 92.23% | 94.83% | 95.18 / 96.13% | 91.29 / 92.21% | 91.20% | 0.63 |

## Files

- Note branch: `sections/*.tex`, `ANA-IDTR-2026-01-INT1-metadata.tex`,
  `ANA-IDTR-2026-01-INT1-defs.sty`, seven new figures `figures/mm_*.pdf`
  (copied from `outputs/11_12_2026_output/hllhc_final/` and `lowpu_final/`).
- Run 3 rerun: `outputs/11_13_2026_output/note_edits/run3_assoc_ntrk2_t094/summary.json`
  and `run3_assoc_t094.log` (not in git).
