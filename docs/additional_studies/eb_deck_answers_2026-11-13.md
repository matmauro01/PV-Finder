# EB deck: answers to Mattia's questions (13 Nov 2026)

Items 1, 2, 3, 4, 10, 11 and 15 of
[RESUME_eb_deck_2026-11-12.md](RESUME_eb_deck_2026-11-12.md). Frozen
configuration (`working_points.py`): HL-LHC s8 at 0.184 / 0.0184, GNN v2 e527;
low pile-up e400 at 0.50 / no height cut, mu60 v2 e115; **t = 0.94** in both
regimes; >= 2 tracks per vertex after the GNN for both algorithms; window
0.17 mm (HS 0.1 mm).

Most answers reuse existing results. New outputs are in
`outputs/11_13_2026_output/eb_questions/` (`O/` below). There are three new
checks: item 1 (a low pile-up rerun with the HL-LHC cuts), item 2 (efficiency
against distance to the interval edge) and item 15 (how AMVF picks its
primary vertex). HL-LHC numbers after the GNN are from
`outputs/11_12_2026_output/hllhc_final/` (`F/` below), at t = 0.94.

---

## 1. Peak cuts: HL-LHC 0.184 / 0.0184 against low pile-up 0.50 / none

**Answer.** Integral and height are in units of each model's output, and the
two are different networks:
- s8 (U-Net v2) and e400 (U-Net v1) were trained on different detectors, with
  different target-width presets. ITk z0 resolution is about 2x better, so the
  HL-LHC target Gaussians are narrower.
- The numbers do not carry over between models. What carries over is the
  method: each regime's cuts fix a fake rate.
  - The HL-LHC cuts were matched to v6's fake rate.
  - 0.50 is the historical Run 2 convention.

The low pile-up rerun with the HL-LHC cuts is not "very similar", so one set
for both regimes is not recommended. The rerun used Run 2 MC, the 2,550 test
events, e400, with the cuts overridden and recorded (`O/item1_lowpu_hllhc_cuts/`).

| Run 2 MC, 0.17 mm | e400 at 0.50 (frozen) | e400 at 0.184 / 0.0184 | AMVF |
|---|---:|---:|---:|
| Efficiency, Eq. (1) | 81.55 % | 83.32 % | 75.74 % |
| Reco vertices | 61,102 | 66,731 | 57,328 |
| Fakes / event | 2.363 | **4.049** (+71 %) | 2.487 |
| Fake fraction | 9.86 % | 15.47 % | 11.06 % |
| σ_vtx-vtx [mm] | 0.392 ± 0.009 | 0.361 ± 0.006 | 0.789 |

With the HL-LHC cuts, efficiency rises 1.8 points, but fakes rise by 1.7 per
event. PV-Finder would then make more fakes than AMVF (4.05 against 2.49 per
event) and lose the "more efficient and fewer fakes" claim at low pile-up.

- Slide bullet: *Cuts are set per model at a matched fake rate. Applying the
  HL-LHC cuts at low pile-up gives +1.8 pt efficiency but +71 % fakes
  (4.0 against 2.4 per event; AMVF 2.5).*
- Slide bullet: *Integral and height are in units of the model output, so the
  numbers are not portable between models; the method is.*

## 2. The 40 mm intervals

**Why we split.**
- The density network has a fixed input and output: 1000 bins × 0.04 mm =
  40 mm.
- The per-track MLP writes a 1000-bin vector for every track. A full 480 mm
  event in one pass would need a 12,000-bin output per track, 12x the memory
  (about 1,300 tracks per event at PU200).
- The U-Net is convolutional, so it only needs local context.
- The 12 intervals run as one batch of 12.

**How boundaries are handled.**
- Intervals are [-240 + 40k, -240 + 40(k+1)) mm and do not overlap. Each track
  goes to the interval that contains its z0.
- Each track carries its interval's z_start and z_end as input features (the
  "incbounds" features).
- The target is built on the full 12,000-bin grid and then cut, so a vertex
  next to an edge also puts target content in the neighbouring interval.
- At inference, the 12 outputs are concatenated in z order. Peak finding runs
  on the **stitched** 12,000-bin histogram, so a peak that crosses an edge is
  one region.
- The GNN sees the whole event, with no intervals.

**Test** (new, from the frozen pre-GNN caches; `scripts/eb_q2_boundary_efficiency.py`,
plot `O/item2_boundary/boundary_efficiency.png`). The sample is PU200, s8 at
the frozen cuts, with AMVF as the control because it has no intervals. The
script reproduces the headline 83.06 % / 18.48 % exactly.

| distance of truth vertex to nearest edge | 0-0.1 mm | 0.1-0.2 | 0.2-0.5 | 0.5-1 | 1-2 | 10-20 mm |
|---|---:|---:|---:|---:|---:|---:|
| PVF eff, r16443 [%] | 82.0 ± 1.2 | 82.0 ± 1.2 | 81.9 ± 0.7 | 83.4 ± 0.5 | 83.0 ± 0.4 | 83.1 ± 0.1 |
| PVF eff, r16638 [%] | 84.8 ± 1.1 | 81.0 ± 1.2 | 83.4 ± 0.7 | 83.2 ± 0.5 | 83.5 ± 0.4 | 83.0 ± 0.1 |
| AMVF eff, r16443 [%] | 76.5 ± 1.3 | 78.7 ± 1.3 | 79.9 ± 0.7 | 80.7 ± 0.5 | 79.7 ± 0.4 | 79.9 ± 0.1 |

- **Efficiency is flat to the edge**, within about 1σ, in both files.
- **One small, real artefact:** reconstructed vertices within 0.1 mm of an
  edge are fake 32-34 % of the time, against 18 % elsewhere and 22-23 % for
  AMVF there. This excess is about 0.2 fake per event, or 1 % of the
  18.6 fakes per event.
- The pairwise-Δz study already gave the same ≤ 1 % size
  (`docs/research/pairwise_dz_bump.md`; tile stitching). Earlier, fakes were
  found no more likely than real peaks to lie within 2 mm of an edge
  (JOURNAL, H4).

- Slide bullet: *Fixed 1000-bin (40 mm) network input; the per-track output
  would be 12x larger for a whole event. Peaks are found on the stitched
  12,000-bin histogram, and the GNN sees the whole event.*
- Slide bullet: *Efficiency is flat to within 0.1 mm of an interval edge
  (82-85 % against 83.1 %). A narrow excess of fakes at the edges is worth
  0.2 per event (1 % of fakes).*

## 3. GNN size (39,521 parameters)

**Answer.** **No larger GNN was tried.**
- Every GNN config in `configs/gnn/` (v1-v4, full-mu v1/v2, mu60) uses
  hidden_dim 32, 4 heads and 2 HeteroConv layers.
- There is no width or depth scan in JOURNAL, docs or memory. Capacity was
  only ever scanned for the density network.

What the evidence says:
- The v2 learning curve has plateaued (-0.07 % over the last pass, LR at its
  floor).
- Training and validation loss are equal (0.2552 / 0.2559), so there is no
  overfitting.
- More data helped: v1 → v2 used 2.5x the data and gained 0.015 in loss and
  about 1 point of track F1 at every cut (`docs/evaluation/gnn_v2_freeze_2026-11-12.md`).

Equal training and validation loss on 150k events means the model is not
memorising. That fits a model limited either by capacity or by the
information in its inputs, and only a wider run can tell them apart. There is
room left: the truth-guided ceiling is 68.6 % Clean/truth at PU200, against
50.6 % for the GNN.

Cost of a test: a hidden_dim 64 run is about 4x the parameters and fits in one
3-6 h GNN v2 training. The GNN is now 98 ms of the 277 ms CPU chain.

- Slide bullet: *Not scanned yet. Training loss equals validation loss and
  has plateaued (no overfitting). More data improved it (v1 → v2), so width
  is the next knob; a hidden_dim 64 run is planned.*

## 4. Density model trained only at PU200

**Why.**
- The PU200 pool is 5,173,600 events (62.1 M intervals). The flat-mu
  files hold only about 200k events, and they are needed untouched for
  evaluation (r16443 / r16638 entries 0-24,999) and for GNN training.
- The network works per 40 mm interval, so it sees local track density, not
  mu. A PU200 event already covers a wide density range: per interval it holds
  32 truth vertices at the centre, 5 at |z| ≈ 100 mm and fewer than 1 beyond
  140 mm. That is the range of a central interval from mu ≈ 30 down to
  mu < 10.

**Does it generalise?** Yes for finding vertices
(`F/h1_vertex_finding/plots/eff_vs_mu_r16443`):
- The Eq. (1) lead over AMVF is positive in every 10-mu bin and largest at low
  mu: +4.6 ± 0.4 pt at mu 0-10, against +3.2 pt at 190-200.
- The fake fraction equals AMVF's at mu < 10 (31.4 against 31.7 %).
- Fakes per event at mu < 30 are 0.07 above AMVF (about 2σ).

The one low-mu weakness is position. The 68 % residual is 0.9-1.6 µm worse
than AMVF's at mu < 20 and better above mu = 30
(`F/resolution_vs_mu/resolution_vs_mu_p68`).

**Would uniform-mu training help?** Possibly for the low-mu position, but not
for efficiency. It would need a new flat-mu production, because the existing
flat files are spoken for. This has not been tried.

Deck fixes:
- GNN v2: 149,820 events at all mu.
- PU200 = **185 ≤ mu ≤ 215**. The deck's "≤ 200" is wrong.

- Slide bullet: *The density network sees one 40 mm interval at a time. PU200
  events already contain low-density intervals (|z| > 100 mm).*
- Slide bullet: *Generalises: the efficiency gain over AMVF is largest at low
  mu (+4.6 pt at mu < 10); position is 1-1.6 µm worse than AMVF at mu < 20.*

## 10. Categories, baselines, and why AMVF ≈ nearest z0 at PU200

### Definitions
Track-purity taxonomy, unweighted tracks (`hs_metrics.classify_vertices`):
- The **dominant** truth vertex of a reco vertex is the one that contributes
  the most of its tracks.
- **Clean:** the dominant vertex provides ≥ 70 % of the tracks.
- **Merged:** the dominant vertex is a real truth vertex with < 70 %.
- **Fake:** the dominant group is tracks without truth.
- **Split:** among the reco vertices that share a dominant truth vertex, all
  except the one with the largest Σp_T². The kept one stays Clean or Merged.
- The rule is ATLAS's 70 % rule, with track counts instead of fit weights
  (`docs/evaluation/atlas_fig25_crosscheck.md`).
- Only vertices with ≥ 2 tracks are classified. The denominator of
  Clean/truth is truth vertices with n_trk ≥ 2.

Baselines (`F/chain_extras/README.md`):
- **Nearest z0:** each track goes to the nearest PV-Finder peak.
- **z0 within 3σ:** the nearest peak if |Δz| < 3σ(z0), where
  σ(z0) = `RecoTrack_ErrZ0`, the track's own fitted z0 uncertainty.
- **Eq. (4):** the same with sqrt(σ(z0)² + σ_PV²), where σ_PV is the peak width.
- **One-to-one / finder cap:** greedy closest-first one-to-one matching of all
  peaks (or AMVF vertices with ≥ 2 tracks) to truth within 0.17 mm, divided by
  the truth count. This is the maximum Clean/truth any track assignment could
  reach on these candidates.
- **Truth-guided:** each truth-labelled track goes to the peak nearest its own
  truth vertex, if that peak is within 0.17 mm. This is an oracle on the real
  peaks, MC only.

### PU200 numbers (t = 0.94, ≥ 2 tracks)
| | Clean/truth | track eff / purity | F1 |
|---|---:|---|---:|
| Finder cap, PVF / AMVF | 73.4 / 70.1 % | | |
| Truth-guided | 68.6 % | 0.879 / 0.943 | 0.910 |
| PVF + GNN, t = 0.94 | **50.6 %** | 0.629 / 0.795 | 0.702 |
| GNN, no cut | 39.4 % | 0.689 / 0.696 | 0.693 |
| Nearest z0 | 31.3 % | 0.687 / 0.687 | 0.687 |
| AMVF (TTVA links) | 30.9 % | 0.683 / 0.686 | 0.685 |

### Why AMVF is no better than nearest z0: diagnosis
1. **The same at every mu, not just at PU200.** AMVF and nearest z0 agree
   within 0.4 pt in every mu bin: 83.4 / 83.9 (mu 0-10), 64.4 / 64.3 (30-40),
   33.2 / 33.5 (150-200). AMVF's track association quality equals assigning
   every track to the nearest vertex position. Both assign about 100 % of
   tracks and both get about 31 % of them wrong at PU200. With 31 % wrong
   tracks, most vertices fall below 70 % purity and become Merged (42.6 %) or
   Split (22.1 %).
2. **AMVF finds the vertices; the association loses them.**
   - Positionally, AMVF finds 79.9 % of truth vertices and has only 0.18
     positional splits per event.
   - In track content, 21.7 of its 97.9 vertices per event are Split: they sit
     on a truth vertex but are dominated by tracks of a neighbour.
3. **The TTVA links are not the fit** (Trey's ntuple,
   `F/supplementary/classes.json`, PU200).
   - 23.8 % of tracks are in no AMVF fit, but TTVA (Nonprompt_Hard_MaxWeight)
     attaches 99.5 % of tracks.
   - 23.4 % of the TTVA track-vertex links are not fit links.
   - Classified with the fit tracks and fit weights, AMVF has 46.3 Clean
     vertices per event against 34.4 with the TTVA links (+34 %). That gives
     about **41.6 % Clean/truth, against 30.9 % with TTVA**:
     - the fit list with unit weights adds +7.3 per event (37.5 %);
     - the weights add +4.6 more.
   - Split barely moves (21.7 → 19.6 per event). Splits are intrinsic to
     AMVF at this density.
4. **Where the GNN gains.**
   - With no score cut, the GNN already gives 39.4 % (+8.5 pt over AMVF): a
     learned assignment using pT, η, d0 and context.
   - The score cut adds +11 pt by leaving the ambiguous 21 % of tracks
     unassigned instead of making them contaminate vertices.
   - AMVF/TTVA is designed to assign every track. It optimises vertex finding,
     HS selection and pile-up track rejection, not the purity of pile-up
     vertices.

### Consistent with ATLAS?
- **Run 2:** yes. Our AMVF classification of TTVA links on Run 2 MC at mu = 60
  gives All / Matched / Merged = 32.2 / 23.8 / 7.8 per event, against ATLAS
  about 32.5 / 23.5 / 8.5 (IDTR-2021-01 Fig. 25a, the same taxonomy as
  ATL-PHYS-PUB-2019-015; `docs/evaluation/atlas_fig25_crosscheck.md`).
- **HL-LHC:** the ITk paper (arXiv:2412.15090, Sec. 5.5) publishes **no**
  per-category counts at mu = 200. It mentions merging and splitting only
  qualitatively, so there is no direct anchor.
- The HS anchors it does publish are reproduced: HS reconstructed and
  selected about 92 % (ours 92.9 %), HS resolution about 10 µm (ours 10.2 µm).
- **"Not tuned for HL-LHC":** the ITk paper says only that ITk uses the same
  AMVF as Run 3, with ITk track selection. It does not say AMVF is untuned. We
  have no evidence of mis-tuning and should not claim it.
- The honest statement: our headline uses AMVF's TTVA output, which is what
  analyses consume. With AMVF's own fit weights, AMVF Clean/truth is about
  41.6 %, still 9 pt below PVF + GNN.
- Our Fake category is structurally about 0 (99.8 % of tracks are
  truth-labelled), so it is not comparable to ATLAS Fake.

- Slide bullet: *AMVF's track association at PU200 is as good as nearest-z0
  in every mu bin (within 0.4 pt). Both attach about 100 % of tracks, 31 % of
  them wrongly, so most vertices fail the 70 % rule (Merged 43 %, Split 22 %).*
- Slide bullet: *With AMVF's own fit tracks and weights, AMVF Clean/truth is
  41.6 % (TTVA links: 30.9 %). Run 2 classification matches ATLAS (mu = 60:
  23.8 against about 23.5 Matched per event); ITk publishes no HL-LHC
  categories. GNN: +8.5 pt from learned assignment, +11 pt from declining
  ambiguous tracks.*
- Plots: `F/chain_extras/plots/clean_yield_stages_pu200`, `F/supplementary/plots/backup/classes_per_event_pu200_all_variants_min2`.

## 11. Why PV-Finder + GNN is weaker against AMVF at low mu (HL-LHC chain)

Clean/truth, t = 0.94, ≥ 2 tracks, test r16443 (`F/chain_extras/chain_extras_summary.json`):

| mu | finder cap PVF / AMVF | truth-guided | GNN no cut | **GNN t = 0.94** | nearest z0 | **AMVF** | GNN − AMVF | GNN track eff t / no cut |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 0-10 | 88.5 / 83.9 | 88.4 | 86.4 | **81.0** | 83.9 | **83.4** | **−2.5 ± 0.6** | 0.851 / 0.960 |
| 10-20 | 84.4 / 80.0 | 83.9 | 78.9 | 79.0 | 74.6 | 74.6 | +4.4 ± 0.4 | 0.852 / 0.921 |
| 20-30 | 83.7 / 78.9 | 83.0 | 74.4 | 76.9 | 69.1 | 68.8 | +8.1 ± 0.3 | 0.853 / 0.892 |
| PU200 | 73.4 / 70.1 | 68.6 | 39.4 | 50.6 | 31.3 | 30.9 | +19.6 ± 0.1 | 0.629 / 0.689 |

**Answer.**
- The finder is not the problem: its cap is +4.6 pt above AMVF at mu < 10, and
  without a score cut the GNN beats AMVF by +3.0 pt there.
- The one PU200-tuned cut t = 0.94 costs 5.5 pt at mu < 10.
  - It removes 11 % of the GNN's correct tracks (track efficiency
    0.960 → 0.851) and buys only 1.6 pt of purity (0.969 → 0.985), because
    almost nothing is wrong to purify at low mu.
  - Under the ≥ 2-track rule, a small vertex that loses tracks drops out.
- The GNN is also less confident at low mu: 47 % of the correct tracks of
  5-10-track vertices score < 0.99 at mu < 10, against 25 % at mu 20-50. Low
  mu (< 20) supplies only about 2 % of the training tracks in a flat-mu sample
  (`gnn_v2_freeze_2026-11-12.md` §6; the cause is plausible, not proven).
- Satellite peaks are not the cause.
  - Pre-GNN fake fractions are equal at mu < 10 (31.4 / 31.7 %).
  - Low-mu splits are mostly 1-track HS satellites, removed by the ≥ 2 rule.
  - After the GNN, Split is 5.9 % of vertices for PVF against 28.8 % for AMVF
    at mu 0-10.
- The gap reverses at 10-20 and grows to +19.6 pt at PU200.
- **Low pile-up regime** (Run 2 MC, mu60 GNN, t = 0.94): PVF + GNN beats AMVF
  in every mu bin (87.4 / 79.1 / 70.3 / 63.8 against 84.2 / 71.6 / 59.4 / 52.0
  for mu 0-20 / 20-40 / 40-60 / 60-80).
- **Error bars:** they exist. Every bin has 200-replica event-bootstrap errors
  and paired errors on the difference from AMVF (`bins[...]['arms']['min2'][q]['err']`,
  `diff_amvf_err`). They are 0.06-0.6 pt, so they are already drawn but hidden
  by the markers. Say "statistical errors smaller than the markers" on the
  slide, or draw the difference panel.

- Slide bullet: *At mu < 10, PV-Finder + GNN is 2.5 ± 0.6 pt below AMVF: one
  PU200-tuned cut t removes 11 % of correctly assigned tracks where there is
  almost nothing to purify. Without the cut the GNN is +3 pt above AMVF.*
- Slide bullet: *The lead is +4 pt from mu = 10 and +20 pt at PU200; a
  mu-dependent t would recover low mu. Statistical errors (bootstrap) are
  smaller than the markers.*
- Plot: `F/chain_extras/plots/clean_yield_stages_vs_mu` (t = 0.94, regenerated 22:05).

## 15. The HS selection claim

**Old claim** "96.3 % against 92.9 %": GNN v4, t = 0.98, plain Σp_T². This
is superseded.

**Current** (t = 0.94, PU200, 1,920 events, `F/chain/chain_headline.json`):

| | HS reconstructed and selected |
|---|---:|
| PV-Finder + GNN, Athena 24.0 rule | **95.05 ± 0.53 %** (1,825) |
| AMVF stored primary | **92.92 ± 0.56 %** (1,784) |
| AMVF TTVA lists re-ranked with the same rule | 89.11 ± 0.69 % (1,711) |
| GNN − AMVF stored (paired) | **+2.1 ± 0.5 pt** |
| Inclusive mu 0-200 | 97.11 / 95.14 (stored) / 93.36 (rule) % |

**Which AMVF number is fair: the stored primary.** A new check (Trey's
ntuple, `scripts/eb_q15_amvf_primary_from_fit_tracks.py`, `O/item15_hs_selection/`)
compares the stored primary with the largest-Σp_T² vertex computed from
different track lists:
- with AMVF's `fitTracks`, the vertex track links that Athena's sorting tool
  actually ranks, the largest-Σp_T² vertex **equals the stored primary in
  25,000 / 25,000 events (1,920 / 1,920 at PU200)**;
- with the TTVA `assocTracks` it agrees in only 77.7 % at PU200;
- with the fit tracks restricted to weight > 0.5 it agrees in 97.0 %.

So the stored primary *is* the Athena Σp_T² rule applied to AMVF's own
inputs. The 89.1 % applies the rule to TTVA lists that Athena never ranks,
which penalises AMVF unfairly; keep it as backup at most.

**Our definition against the ITk paper** (arXiv:2412.15090, Sec. 5.5).
Theirs: "the fraction of events in which the hard-scatter vertex is
reconstructed and selected as the highest Σp_T² primary vertex". Reconstructed
means within **0.1 mm in z (longitudinal only)** of the true HS. The
HS-matched vertex is the one with the most tracks matched to HS particles.
The paper quotes about 92 % at mu = 200.

Ours:
- the same 0.1 mm z window;
- the same "most HS tracks" match;
- the same Σp_T² selection.

Differences:
- tracks are matched through the stored truth-vertex association, not
  R_match > 0.5;
- tracks are counted unweighted;
- our PU200 range is 185-215, theirs 190-210. Restricting to 190-200 moves
  AMVF to 93.6 %.
- For the GNN, Σp_T² runs over the GNN-assigned tracks that are χ²-compatible
  (18.42) with the 3D-fit vertex, the analogue of Athena's link list.
- AMVF 92.9 % against ITk about 92 % shows we follow their definition closely.

**Proposed conclusion wording:**
> *Hard-scatter vertex reconstructed (within 0.1 mm) and selected as the
> highest-Σp_T² vertex at ⟨μ⟩ ≈ 200: 95.1 ± 0.5 % for PV-Finder + GNN against
> 92.9 ± 0.6 % for AMVF (ITk definition; AMVF's own primary, consistent with
> the ITk paper's ≈ 92 %).*

- Slide bullet: *HS selection (ITk definition, 0.1 mm, highest Σp_T²): 95.1 %
  against 92.9 % for AMVF at PU200 (+2.1 ± 0.5 pt); inclusive 97.1 against
  95.1 %.*
- Slide bullet (footnote): *AMVF's stored primary is exactly the Σp_T² leader
  of its fit tracks (25,000 / 25,000 events); re-ranking its TTVA lists
  instead gives 89.1 % (backup).*

---

## Flags for the deck

1. **Clean HS vertex and HS purity are now below AMVF at t = 0.94:** 88.3
   against 92.5 % and 82.3 against 88.4 % (they were above at 0.97). The HS
   track fraction is still higher (87.9 against 74.1 %). Do not claim a
   Clean-HS win.
2. **mu 0-10: Clean/truth is below AMVF** by 2.5 ± 0.6 pt. The
   `working_points.py` note "beats AMVF in Clean/truth in every mu range" is
   true only for the coarse mu < 20 bin (+2.5 pt on test) used on validation.
   The 10-unit plot shows the dip.
3. **AMVF ≈ nearest z0 depends on the TTVA input.** With AMVF's fit weights,
   AMVF reaches about 41.6 % Clean/truth. The deck should say that the
   headline uses the TTVA links.
4. Interval edges: about 0.2 extra fake per event within 0.1 mm of an edge.
5. HS selection: quote 95.1 against 92.9 %, not 96.3 against 92.9 %.
6. PU200 is 185-215, not 185-200 (samples slide).

## Reproduction
```bash
# item 1 (tmux, about 7 min on one A100)
python -u src/pv_finder/evaluation/vertex_finding/run_eval_pvf.py --regime lowpu --device 1 \
  --indices configs/test_main_indices_2550evt.p --integral-threshold 0.184 \
  --integral-threshold-res 0.184 --min-height 0.0184 \
  --output-dir outputs/11_13_2026_output/eb_questions/item1_lowpu_hllhc_cuts --dataset-name 'Run 2 MC (HL-LHC cuts)'
# item 2 (seconds)
F=outputs/11_12_2026_output/hllhc_final/h1_vertex_finding
python -u scripts/eb_q2_boundary_efficiency.py --cache $F/fair_r16443/cache_r16443.npz \
  $F/fair_r16638/cache_r16638.npz --out outputs/11_13_2026_output/eb_questions/item2_boundary
# item 15 (about 5 min, CPU)
python -u scripts/eb_q15_amvf_primary_from_fit_tracks.py --n 25000 \
  --out outputs/11_13_2026_output/eb_questions/item15_hs_selection
```
