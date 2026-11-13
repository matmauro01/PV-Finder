# Paper draft of 11 November 2026: answers, frozen configuration, datasets

Draft: `final_paper/current_paper/paper/PVFinderPaper_11_11_2026.pdf`
(32 pages). The line numbers below are the draft's own line numbers.

Decisions referred to as D# are in
[meeting_2026-11-11_plan.md](meeting_2026-11-11_plan.md). Repo paths are
relative to the repo root. "Note" means the internal note, GitLab main
91b6f49.

Every answer uses the frozen configuration
(`src/pv_finder/utils/working_points.py`, Part 2) and the final numbers of
[hllhc_final_numbers_2026-11-12.md](../evaluation/hllhc_final_numbers_2026-11-12.md)
(HL-LHC) and
[lowpu_final_numbers_2026-11-12.md](../evaluation/lowpu_final_numbers_2026-11-12.md)
(low pile-up). The HL-LHC t may still change after the rescan under the
>= 2-track rule; every HL-LHC number that depends on t is marked
`<!-- t-dependent -->` in the source of this page. Output paths:
`F/` = `outputs/11_12_2026_output/hllhc_final/`, `lowpu_final/` =
`outputs/11_12_2026_output/lowpu_final/`.

Status keys:
- **Answered**: settled by code, a checkpoint, a log or a final recompute.
- **D#**: settled by a decision.
- **New result**: needs a study that has not been done.

---

## Part 1. The 29 questions

### Q1 (l. 190). Do the 40 mm intervals overlap?
**Answered.**
- No overlap. Interval k covers [-240 + 40k, -240 + 40(k+1)) mm, k = 0..11
  (`src/pv_finder/data/root_to_h5.py:34-36,108-110`).
- Each track goes to the one interval that contains its z0. The lower edge
  is included (`searchsorted(..., side="left")`, `root_to_h5.py:137-140`).
- Tracks outside [-240, 240) mm are dropped (`root_to_h5.py:128`). The
  ntuple selection already requires |z0| <= 200 mm
  (`docs/data/run_4_all_etas.md`).
- Boundary effect worth one sentence in the paper: the target histogram is
  built on the full 12,000-bin grid and then cut into twelve pieces
  (`_build_truth_histogram`, `root_to_h5.py:67-104`). A truth vertex near a
  boundary can therefore put target content into the neighbouring interval,
  which holds none of that vertex's tracks.

### Q2 (Fig. 3 caption). Is every element of Fig. 3 right, and does it cover Run 2/3 too?
**Answered.** Fig. 3 must describe the frozen HL-LHC model s8, with the
corrections below. For Run 2/3 it needs three changes.

HL-LHC corrections:
- **Latent size.** s8 has a 1 x 1000 latent representation, not 4 x 1000.
  The same holds for l. 212 and l. 219. The "4 x 1000", the 3.55 x 10^6
  parameters (l. 222, Table 1) and the 280 channels of Table B.1 of the note
  describe v6, which is not used for any result. s8 has 80 channels and
  439,593 parameters (Q4). Table B.1 must be changed to s8.
- **Input features.** The density network uses seven features per track:
  d0, z0, sigma(d0), sigma(z0), cov(d0, z0), interval start and interval end
  (`root_to_h5.py:37`). Sec. 4.1.1 l. 183 says helix parameters
  (d0, z0, phi, theta, q/p) with covariance; that is wrong for the density
  network. theta, phi and pT enter only the GNN.
- **GNN update order.** In each GAT layer, the track-to-vertex and
  vertex-to-track updates run in parallel from the previous layer's
  embeddings (one `HeteroConv`, `src/gnn/models/ttva_gat.py:67-90,113-121`).
  Sec. 4.2.2 l. 307-308 says vertices are updated first and tracks second;
  that is not what the code does.

Run 2/3 differences:
- MLP 7 -> 100 x 5 -> 1000 instead of 128 x 5 (the latent is 1 x 1000 in
  both regimes);
- the U-Net v1 design with 64 channels (Q10);
- fully connected graphs instead of k = 20.

### Q3 (l. 205). Target widths come from AMVF. Has anything been done to remove this?
**Answered: not yet.**
- Widths: sigma(n) = A n^-B + C, floor 2 um. Vertices with fewer than two
  tracks get one bin, 0.04 mm (`root_to_h5.py:82-88`).
- A, B, C (preset `hllhc_alleta`) are a background-corrected fit of
  AMVF-truth residuals (`src/pv_finder/data/resolution_presets.py`).
- What exists: a per-vertex, track-only precision (inverse-variance z0 of
  the truth-associated tracks), computed for diagnostics in
  `docs/evaluation/run2_vs_run4_resolution_2026-11-09.md`.
- What removing AMVF would take: a new preset fitted on that quantity, a
  rebuild of the 5.17 M-event HDF5 pool, and a retrain (s8 trained in
  13.7 h, Q7). Not started.

### Q4 (l. 223). U-Net v2 details.
**Answered** (`src/pv_finder/models/unet_v2.py`, `build_model` in
`src/pv_finder/training/hllhc_helpers.py:55-87`,
`configs/vertex_finding/config_hllhc_alleta_v6_small_s8.yml`).

Yes, s8 uses the Appendix E design, at 80 channels and one latent channel:
- **Channels:** 80 at every stage (`n_unet_channels: 80`). Input: 1 latent
  channel.
- **Encoder:**
  - ConvBNReLU(1 -> 80, kernel 15) at 1000 bins;
  - residual block (kernel 5) at 1000;
  - max-pool 2, residual block (kernel 5) at 500;
  - max-pool 2, residual block (kernel 5) at 250. That is the bottleneck:
    two down-sampling stages, factor 4.
- **Decoder:** nearest-neighbour upsample x2 + Conv(kernel 3) + BN + ReLU,
  then an additive skip and a residual block (kernel 5). Done twice:
  250 -> 500 (skip from the 500-bin encoder), then 500 -> 1000 (skip from
  the second 1000-bin encoder block).
- **Head:** ConvBNReLU(80 -> 80, k 5), ConvBNReLU(80 -> 40, k 3),
  Conv(40 -> 1, k 1), softplus output.
- **Residual block:** Conv + BN + ReLU + Dropout, plus the identity.
- **Normalisation and activations:** BatchNorm after every convolution
  except the last 1x1. ReLU throughout the U-Net.
- **Dropout:** p = 0.25 after the ReLU of every convolution block (encoder,
  decoder, upsampling, head). None in the MLP.
- **MLP:** 7 -> 128 x 5 -> 1000, LeakyReLU slope 0.01, softplus output.
  Padded tracks are masked, the result is summed over tracks and scaled by
  0.001 (`autoencoder_models.py:214-352`).
- **Parameters:** 439,593 = MLP 196,072 + U-Net 243,521 (counted from the
  checkpoint; the training log gives the same total).

### Q5 (l. 232). Concatenation or smoothing? One batch?
**Answered.**
- Plain concatenation in z order. No smoothing, normalisation or boundary
  treatment (`run_eval_pvf_run3.py:195-198`).
- The twelve intervals run as one batch of 12, padded to the widest
  interval (`run_inference`, l. 201-215).
- In Athena the tensor is (12, 7, N_trk) (note Fig. athena-diagram), also a
  single batch.

### Q6 (l. 244). Phase 1: U-Net absent, or present and frozen?
**Answered: both are true. The paper should say "absent".**
- Absent from the forward pass. `phase1_forward` runs only the MLP, and its
  1000-bin output is compared with the target (`hllhc_helpers.py:185-195`;
  `train_mlp_hist_then_e2e.py:106-115`).
- Its parameters are also frozen (`requires_grad = False`,
  `train_hllhc_e2e.py:81-90`) and are switched back on for Phase 2
  (l. 128).
- The low-pile-up model e400 is trained the same way (Q10).

### Q7 (l. 250, Table 1). Datasets, checkpoint choice, warm-up, time, hardware.
**Answered** (`configs/vertex_finding/config_hllhc_alleta_v6_small_s8.yml`;
logs `outputs/logs/hllhc_alleta_v6_s8_1p1ep_20261109_211919.log` and
`..._20261110_211930.log`).
- **Training sample:** the six PU200 files of Part 3, 5,173,600 events, the
  same pool and split as v6.
- **Split:** seed-42 random permutation of the 62,083,200 *intervals*:
  59,599,872 train / 1,862,496 validation / 620,832 test
  (`collectdata_poca_KDE.py:88-104`). Twelve intervals of one event can
  land in different roles. The test intervals are not used for any result:
  results come from the independent files r16443 and r16638.
- **Checkpoint:** `hllhc_alleta_v6_s8_1p1ep_phase2_step_450000_fullstate.pth`,
  Phase-2 step 450,000 of 465,624 (97% of the epoch). The run stopped on a
  host-memory error after this step, before the end-of-epoch checkpoint and
  validation. The learning rate at this step is 1.28e-6, against a floor of
  1e-6, so the schedule had essentially finished.
  The model was frozen on its held-out performance
  ([model_freeze_candidates_2026-11-12.md](../evaluation/model_freeze_candidates_2026-11-12.md)).
- **Learning rate (Table 1 must change from 2 + 2 to 1 + 1 epochs):**
  - Phase 1: constant 1e-3, 1 epoch.
  - Phase 2: per-step linear warm-up from 0.01 x 1e-4 to 1e-4 over 3,000
    steps, then per-step cosine decay to 1e-6 over 1 epoch
    (`build_phase2_scheduler`, `hllhc_helpers.py:100-160`).
- **Other settings:** Adam, betas (0.9, 0.999), batch 128, gradient-norm
  clip 1.0, MSE loss, dropout 0.25.
- **Time and hardware:** one NVIDIA A100-PCIE-40GB. Phase 1: 9.1 h
  (2026-11-09 21:19 to 2026-11-10 06:27). Phase 2: 4.6 h for 450,000 steps
  (2026-11-10 21:19 to 2026-11-11 01:53), restarted from the Phase-1
  checkpoint after a cuDNN initialisation crash. 465,624 steps per epoch.

### Q8 (l. 262). Peak width, sigma_PV, height.
**Answered** (`efficiency_res_optimized_atlas.py:16-190`).
- **Width:** sqrt of the amplitude-weighted variance of the bin index over
  the whole above-threshold region (up to the split point), times 0.04 mm.
- **sigma_PV:** this width is used directly as sigma_PV in the first edge
  feature, (z0 - z_PV) / sqrt(sigma(z0)^2 + sigma_PV^2)
  (`graph_construction.py:105-146,291-360`;
  `pu200_chain_graphs.py:201-217`). At PU200 its median is 0.072 mm on
  matched peaks.
- **Height:** the maximum bin value of the region, the raw predicted density
  with no normalisation (`peakvals`). It is the second vertex-node feature.
- **Training graphs differ:** they use truth z, sigma_PV from the resolution
  model (`hllhc_alleta`) and a height from the same Gaussian-in-bin recipe
  as the targets (`root_to_graphs.py:129`,
  `graph_construction.py:149-168`). This is the training/inference mismatch
  the note discusses.

### Q9 (l. 266-273). Final peak-finding configurations; is the local centroid also used at low pile-up?
**D1. Answered.** Two configurations, one per regime (Part 2):

| | Amplitude | Min width | Integral | Min height | Position |
|---|---|---|---|---|---|
| HL-LHC (s8) | 0.01 | 3 bins | 0.184 | 0.0184 | local centroid ±3 bins |
| Run 2/3 (e400) | 0.01 | 3 bins | 0.50 | none | local centroid ±3 bins |

- The local centroid (amplitude-weighted mean of the bins within three bins
  of the maximum) is used in both regimes.
- l. 267-268 must change: 0.30 / 0.03 were v6's cuts. The s8 cuts
  0.184 / 0.0184 give v6's fake rate (two-file mean of the matched values
  0.182 and 0.186).
- There is no separate threshold for the resolution: sigma_vtx-vtx is fitted
  on the counted peaks. The note's "0.50 for the HL-LHC resolution fit" and
  the low-pile-up GNN's 0.2 no longer apply to any result.

### Q10 (l. 281). Run 2/3 density model.
**Answered.** One density model, PVF-e400, for every low-pile-up result
(Run 2 MC, Run 2 data, Run 3 data, the GNN chain).
- **Checkpoint:**
  `model_weights/lowpu_e400/e2e_mlpHist50_e2e400_1latent_mse_phase2_epoch_400_fullstate.pth`
  (sha1-identical copy of `atlas_pvfinder/model_weights/e2e_attempt4_mlp_hist/`).
- **Class:** `trackstoHists_UNet_1000` (v1), `run_eval_pvf.py:95-108`.
- **MLP:** 7 -> 100 x 5 -> 1000, LeakyReLU 0.01, softplus, 1 latent
  channel.
- **U-Net v1, 64 channels:**
  - encoder ConvBNReLU kernels 25, 7, 5, 5, with max-pool between them,
    giving 1000 -> 500 -> 250 -> 125 (bottleneck 125 bins);
  - decoder: three ConvTranspose(2, 2) + ConvBNReLU(kernel 5) blocks with
    concatenated skips;
  - output Conv(128 -> 64, k 5), Conv(64 -> 1, k 5), softplus;
  - dropout 0.25 in every conv block (`autoencoder_models.py:801-1000`).
  - So yes: the original design.
- **Parameters:** 358,713 trainable = MLP 142,200 + U-Net 216,513 (training
  log). The 359,616 in `working_points.py` includes the batch-norm running
  statistics.
- **Inputs:** the same seven features as HL-LHC.
- **Training:** the same two-phase procedure as HL-LHC, with no KDE at any
  step (`configs/vertex_finding/config_mlp_hist_e2e.yml`):
  - Phase 1: 50 epochs, MLP only, compared with the target histogram;
  - Phase 2: 400 epochs end to end;
  - Adam, constant lr 1e-3 in both phases, batch 128, MSE, dropout 0.25;
  - checkpoint: the last epoch (400);
  - 11.7 h in total (2025-11-12 17:21 to 2025-11-13 05:03).
- **Data:** 51,000 Run 2 MC events (`/share/lazy/qibinlei/..._Target_Y_split.h5`),
  contiguous split 0.70 / 0.25 / 0.05: events 0-35,699 train,
  35,700-48,449 validation, 48,450-50,999 test (the 2,550 test events of
  every Run 2 MC result; index files in
  `atlas_pvfinder/mattia_finder/end_to_end_training_random_init/attempt_4/`).
- **Target preset:** the targets are Qi Bin's file. The file carries no
  attributes. Our registry records the upstream constants as preset `run3`
  (A = 0.23817, B = 0.49491, C = -0.00079 mm). Only Qi Bin can confirm what
  was used to build that file.
- l. 276-280 must change: the Run 2/3 model is not trained in two separate
  stages on a KDE, and no analytic KDE is used anywhere in the paper.

### Q11 (l. 314). GNN details.
**Answered** (`src/gnn/models/ttva_gat.py`).
- **Layers:** PyG `GATConv` (the original GAT attention, not GATv2), 4
  heads, `concat=False` (heads averaged), `edge_dim=3` (edge features enter
  the attention), no self-loops. Two `HeteroConv` layers (aggregation "max";
  only one relation per destination type, so it has no effect).
- **Activations:** LeakyReLU(0.01) after the two linear encoders and the two
  hidden edge-MLP layers. ReLU after each residual sum. The score is the
  sigmoid of the logit.
- **Normalisation:** none.
- **Dropout:** none is applied. `dropout=0.25` is stored but never used in
  `forward` (l. 54, 104-137), so the "dropout 0.25" in the GNN docs and
  configs does not describe the trained models.
- **Parameters:** 39,521 in both frozen checkpoints (HL-LHC v2 epoch 527,
  low pile-up mu60 v2 epoch 115).
- **d0:** used as stored in the ntuple. In the HL-LHC simulation the beam
  spot is at (0, 0, 0) (`BeamPosX/Y/Z` = 0, sigma_xy 12 um), so "relative
  to the beam spot" and "relative to the origin" coincide. For collision
  data the ntuple's d0 reference was not verified; Rocky would know.
- **Tracks:** the same selected tracks as the density network. The chain
  builds the PV-Finder input and the graph from the same z0 and d0 arrays
  (`pu200_chain_graphs.py:189-217`).
- **Track features:** d0, z0, sigma(d0), sigma(z0), cov, theta, phi,
  pT / 50 GeV (`constants.py:10`).

### Q12 (l. 361). Does Eq. (1) equal the note's efficiency? Fake rate as a fraction of reco?
**Answered.**
- Eq. (1) ("truth with at least one reco within w") equals the note's
  clean + merged truth count exactly, by construction (§3.3a in
  `docs/evaluation/metric_definitions.md`). The final recompute checks this
  identity in every event.
- **Fake fraction** before the GNN, 0.17 mm, r16443 PU200 (1,920 events):

  | | Fake / reco | Fakes / event | Splits / event |
  |---|---|---|---|
  | PV-Finder | 18.48 ± 0.09% | 18.64 ± 0.10 | 0.560 ± 0.016 |
  | AMVF | 20.08 ± 0.09% | 19.65 ± 0.10 | 0.184 ± 0.010 |

  Difference: -1.60 ± 0.08 points. r16638 gives 18.50 vs 20.04%.

### Q13 (l. 367). One window for everything?
**D2. Answered.**
- 0.17 mm for every vertex-matching result, both regimes. 0.1 mm for
  hard-scatter results only, as in the ITk paper.
- HL-LHC: 3 x the larger 68% single-vertex residual at PU200. With s8 it is
  55.58 um for PV-Finder and 56.45 um for AMVF, so 3 x 68% is 0.167 /
  0.169 mm, which rounds to 0.17 mm.
- Low pile-up: the same 0.17 mm is about 2.4 times the 68% residual on
  Run 2 MC (2.35 x PV-Finder's 72.5 um, 2.54 x AMVF's 66.8 um).
- Not 0.1 mm for all vertices: most pile-up vertices have a 68% residual of
  55-56 um (matching_window_audit_2026-11-09.md).
- The 1 mm used for single-vertex residuals is an assignment radius for
  measuring the residuals that set the window, not a matching window. It
  must stay wider than the window, or the derivation is circular.

### Q14 (l. 392). HS vertex resolution: include it? Definition?
**Answered.** Include it.
- **Definition (ITk paper Fig. 28, Athena `iterRMS_convergence`):** for
  events with a reconstructed vertex within 0.1 mm of the true HS, take
  z_reco - z_HS for the closest such vertex. The resolution is the iterative
  3-RMS width.
- **PV-Finder + GNN position:** a 3D adaptive vertex fit with a beam-spot
  constraint, on the tracks the GNN assigns at t to the HS candidate.
- **Result** on the 1,915 PU200 events where both algorithms find the HS:
  PV-Finder + GNN 9.48 ± 0.28 um, AMVF 10.24 ± 0.27 um, difference
  -0.77 ± 0.24 um. <!-- t-dependent -->
- It is flat in mu and in local density. Inclusive (mu 0-200): 9.56 vs
  9.93 um. <!-- t-dependent -->

### Q15 (l. 396). Local pile-up density results?
**Answered** (plots listed in
[hllhc_final_numbers_2026-11-12.md](../evaluation/hllhc_final_numbers_2026-11-12.md)).
- **HS metrics vs density** (ITk definition: truth vertices within ±2 mm of
  the true HS, excluding it, divided by 4 mm): HS reconstruction, HS
  selection and HS resolution, `F/supplementary/plots/main/hs_*_vs_density_itk_min2`.
- **All vertices vs density** (other truth vertices within ±2 mm of each
  truth vertex, divided by 4 mm): Eq. (1) efficiency before the GNN and the
  Clean fraction after it, `F/chain_extras/plots/eff_vs_density_*`.
- Trend: PV-Finder's Eq. (1) efficiency is flat in density while AMVF's
  falls; the HS resolution is flat.

### Q16 (l. 428). F1 inputs: exact definitions.
**Answered** (`src/gnn/evaluation/threshold_scan.py:149-164,240-252`;
`chain_scan.py:96-130`).
- **Correct track:** a track assigned to a kept reconstructed vertex whose
  dominant truth vertex is the track's own truth vertex.
- **Efficiency:** correct tracks / all tracks with a truth vertex. Tracks
  left unassigned, including those on vertices dropped by the >= 2-track
  rule, count in the denominator.
- **Purity:** correct tracks / assigned tracks that have a truth vertex.
- **Tracks without truth association** are excluded from both; they enter
  only the assigned fraction.

### Q17 (l. 434). Wrong-assignment and unassigned-track metrics.
**Answered** (`F/chain_extras/chain_extras_summary.json`, PU200, >= 2 tracks).
- Wrong-assignment rate = 1 - purity: 14.4% (GNN) vs 31.4% (AMVF).
  <!-- t-dependent -->
- Truth-matched tracks left unassigned: 34.6% vs 0.5%. <!-- t-dependent -->
- Tracks without truth correctly left unassigned: 75.1% vs 4.6%.
  <!-- t-dependent -->
- Low pile-up (Run 2 MC): wrong-assignment rate 10.1% vs 14.1% (purity
  89.91 vs 85.88%).

### Q18 (l. 445). Significance-based association, Eq. (4).
**Answered** (`F/chain_extras/`, PU200, >= 2 tracks).
- Eq. (4) uses sqrt(sigma(z0)^2 + sigma_PV^2) on the same k = 20 candidates
  as the GNN. The draft's sigma_PV^2 in the denominator is consistent with
  the code (`graph_construction.py:128`).
- Clean/truth: Eq. (4) 32.26% at s_max = 3 (32.67% at the best s_max = 2),
  z0 within 3 sigma 32.34%, nearest z0 31.31%, AMVF 30.92%.
- PV-Finder + GNN gives 53.89%, so the gain is the learned assignment plus the
  score cut, not the significance. <!-- t-dependent -->
- l. 441 must change: sigma_PV is the peak width (Q8), not the Appendix B
  parameterisation. The two agree at n_trk = 2 and differ by up to 2x above
  10 tracks.

### Q19 (l. 453). Why 25,000 events and the 1,920 PU200 events?
**Answered: compute budget, not physics.**
- The 25,000 are the first entries of the flat-mu file r16443
  (100,000 events, mu 0-200, about 500 per integer mu, mean 99.6).
- The 1,920 are those with 185 <= mu <= 215: actual range 185-200,
  mean 192.5. The whole file has 7,680 events in that window.
- Before the GNN, the first 25,000 entries of r16638 (1,888 PU200 events)
  give a second independent sample.
- After the GNN only r16443 entries 0-24,999 are used: GNN v2 trains on
  r16443 entries >= 25,000 and on r16638 (Part 3). The only further unseen
  events are the reserved 7,500 r16443 and 9,980 r16638 entries, kept for one
  evaluation.

### Q20 (l. 470). Chance-match check, window scan; efficiency vs mu, n_trk, separation.
**Answered** (`F/h1_vertex_finding/`, `F/chain_extras/`).
- Window scan: auxiliary figure `window_scan_{tag}`, which also shows the
  3 mm-shift chance-match curve. The margin is +3.0 to +3.3 points for
  windows of 0.10-0.20 mm (+3.20 at 0.17 mm). Keep the chance-match
  discussion in the note.
- Efficiency vs mu: `eff_vs_mu_{tag}`.
- Efficiency vs truth n_trk (PUB-note style): `eff_vs_n_trk_pregnn_*`
  before the GNN, `eff_vs_n_trk_postgnn_*` after it. It rises to 100%.
- Efficiency vs separation: `eff_vs_dz_nearest_pregnn_*` (distance to the
  nearest truth vertex), backup.

### Q21 (l. 481). sigma_vtx-vtx threshold; AMVF value and fit.
**Answered.**
- One threshold: sigma is fitted on the counted peaks (s8 cuts, Q9).
- r16443 PU200: PV-Finder 0.217 ± 0.003 mm, AMVF 0.305 ± 0.006 mm (same fit,
  300 bins over ±6 mm, same 1,920 events, AMVF vertices with n_trk >= 2).
  r16638: 0.218 vs 0.299 mm.
- Caveat for AMVF: its production fit (unweighted, c >= 0) sits on the
  boundary c = 0. A sqrt(N)-weighted fit gives 0.298 mm and an unbounded fit
  0.282 mm; PV-Finder is stable. The AMVF / PV-Finder ratio is 1.30-1.40
  depending on the fit.
- Resolution vs mu in the paper is the 68% residual, not sigma_vtx-vtx.
  sigma_vtx-vtx is quoted only at PU200, or per bin for mu >= 100
  ([resolution_vs_mu_2026-11-11.md](../evaluation/resolution_vs_mu_2026-11-11.md);
  s8 table in `F/resolution_vs_mu/table.md`).

### Q22 (l. 500). Merged/Split after the GNN at mu ~ 200; drop candidates without tracks?
**Answered.**
- A reconstructed vertex counts only if it has >= 2 tracks, for both
  algorithms (PV-Finder: GNN-assigned tracks at t; AMVF: its associated
  tracks). Candidates below that are removed from the output, and their
  tracks count as unassigned.
- r16443 PU200 at t = 0.97:

  <!-- t-dependent -->
  | | Vertices | Clean | Merged | Split | Fake |
  |---|---:|---:|---:|---:|---:|
  | PV-Finder + GNN | 143,528 | 115,192 | 24,491 | 3,791 | 54 |
  | AMVF | 187,908 | 66,096 | 80,104 | 41,573 | 135 |

- Per event: 60.00 / 12.76 / 1.97 / 0.03 vs 34.42 / 41.72 / 21.65 / 0.07.
  <!-- t-dependent -->
- Clean/truth: 53.89 ± 0.11% vs 30.92 ± 0.11%. <!-- t-dependent -->

### Q23 (l. 514). GNN thresholds at mu ~ 200; one working point?
**D1, D4. Answered: one working point, t = 0.97, GNN v2 epoch 527.** <!-- t-dependent -->
- The same t is used for vertex classification, HS selection, HS resolution
  and the track metrics. <!-- t-dependent -->
- Chosen by Mattia from options derived on the GNN validation events
  ([gnn_v2_freeze_2026-11-12.md](../evaluation/gnn_v2_freeze_2026-11-12.md)):
  Clean/truth within 0.6 points of the maximum (t = 0.98), the most even
  across mu, and better track F1 and HS resolution. <!-- t-dependent -->
- HS selection at PU200, the same Athena 24.0 rule for both: 95.00 ± 0.52%
  vs AMVF 89.11 ± 0.69%. AMVF's stored primary gives 92.92 ± 0.56%.
  <!-- t-dependent -->

### Q24 (l. 528). AMVF Run 2 sigma on a different subset; AMVF efficiency.
**Answered.** Both on the same 2,550 Run 2 MC test events, the same code and
the frozen low-pile-up configuration:

| | sigma_vtx-vtx | Efficiency, Eq. (1), 0.17 mm | Fakes / event | 68% residual |
|---|---|---|---|---|
| PV-Finder | 0.392 ± 0.009 mm | 81.55 ± 0.14% | 2.363 ± 0.040 | 72.5 um |
| AMVF | 0.789 ± 0.015 mm | 75.74 ± 0.17% | 2.487 ± 0.044 | 66.8 um |

Source: [lowpu_final_numbers_2026-11-12.md](../evaluation/lowpu_final_numbers_2026-11-12.md), R1.

### Q25 (l. 535). Low-pile-up GNN uses integral 0.2, vertex finding 0.50.
**D1. Answered.** Both now use the same peaks: e400 at integral 0.50, local
centroid. The GNN graphs are built from the vertex-finding peaks of the same
2,550 events. Run 2 MC chain at t = 0.97, >= 2 tracks:

| | PV-Finder + GNN | AMVF |
|---|---|---|
| Vertices | 59,151 | 56,338 |
| Clean / Merged / Split / Fake | 87.24 / 11.31 / 1.45 / 0.003% | 77.48 / 20.40 / 2.11 / 0.004% |
| Clean / truth | 71.48 ± 0.24% | 60.47 ± 0.28% |
| Track F1 | 86.88 ± 0.11% | 84.86 ± 0.12% |
| HS identification | 99.06 ± 0.19% | 98.75 ± 0.21% |

Table 5 of the draft (77.43%, integral 0.2, t = 0.99) is replaced by these.

### Q26 (l. 548). GNN thresholds in Run 2 simulation.
**D1. Answered: one t = 0.97 for every low-pile-up use**, the same as HL-LHC.
- Rule: keep 0.97 unless another t is better in Clean/truth, HS
  identification and track F1 together, each by more than two paired
  standard errors. No t is.
- 0.97 is within 0.5 points of the best Clean/truth (71.94% at 0.985) and
  within 0.15 points of the best track F1 (87.03% at 0.95).
- The July HS identification of 98.98% was at t = 0.99, not at the t = 0.5
  that the JOURNAL names. At 0.97 it is 99.06%.

### Q27 (l. 574). Run 3 data association: t, peak threshold, event subset.
**Answered.**
- Settings: the frozen low-pile-up configuration. e400 at integral 0.50, no
  height cut, local centroid, t = 0.97; GNN vertices need >= 2 tracks, AMVF
  vertices nTracks >= 2. Agreement windows 0.17 mm and 1 mm.
- **Events:** the same 2,500 Run 3 events and the same peaks as the Run 3
  vertex finding (the first 2,500 loader-accepted entries in 0-3,158 of
  `file_3.root`).
- Results: GNN assigns 92.23% of tracks (AMVF 94.83%). Of the tracks both
  assign, 95.18% (0.17 mm) and 96.13% (1 mm) go to vertices that agree. The
  leading vertex agrees in 91.20% of events.
- These are agreement numbers, not efficiencies.

### Q28 (Fig. 10 caption). "integral=0.2" in panel titles vs 0.5 in the note.
**Answered: the threshold is 0.50.**
- The old panels were made at 0.5; "integral=0.2" was a stale default label
  of the April 2026 plotting code.
- The regenerated panels (e400, frozen configuration) state "PV-Finder
  integral threshold 0.50": `lowpu_final/summary/figures/fig10{a,b,c,d}_*`.
  The Run 3 panels say the MC is Run 2 MC.

### Q29 (l. 652). Remove the AMVF dependence; which sample for `hllhc_alleta`; Run 2/3 parameters.
**Answered.**
- AMVF dependence: see Q3.
- `hllhc_alleta` was fitted on the **full-|eta| (|eta| < 4)** sample:
  601229 r16633, 99,800 events, 2.31 M truth vertices with n_trk >= 2
  (`resolution_presets.py`; `outputs/07_31_2026_output/amvf_resolution_alleta/`).
- Run 2/3 parameters: A = 0.23817443 mm, B = 0.49491396,
  C = -0.000787436 mm (preset `run3`), with the caveat of Q10.

---

## Part 2. The frozen configuration

Two configurations, defined in `src/pv_finder/utils/working_points.py`
(`get_regime("hllhc")`, `get_regime("lowpu")`). Every result is made with
one of them and records it; no override is used in any production run.

| | HL-LHC (`hllhc`) | Run 2/3 (`lowpu`) |
|---|---|---|
| Samples | ITk simulation, 0 <= mu <= 200 | Run 2 MC, Run 2 data, Run 3 data |
| Density model | s8: U-Net v2, 80 channels, 1 x 1000 latent, MLP 128 x 5, 439,593 parameters; Phase-2 step 450,000 of 465,624 | e400: U-Net v1, 64 channels, 1 x 1000 latent, MLP 100 x 5, 358,713 trainable parameters; Phase-2 epoch 400 |
| Amplitude, min width | 0.01, 3 bins | 0.01, 3 bins |
| Integral, min height | 0.184, 0.0184 | 0.50, none |
| Position | local centroid ±3 bins | local centroid ±3 bins |
| sigma_vtx-vtx | counted peaks, 300 bins over ±6 mm | same |
| Vertex window | 0.17 mm | 0.17 mm |
| HS window | 0.1 mm | 0.1 mm |
| Residual assignment radius | 1 mm (measures the residuals, not a window) | 1 mm |
| GNN | full-mu v2 epoch 527 | mu60 v2 epoch 115 |
| GNN t | 0.97 <!-- t-dependent --> | 0.97 |
| Vertex rule after the GNN | >= 2 tracks, both algorithms | >= 2 tracks, both algorithms |

Other fixed definitions, not working points: Clean = >= 70% track purity;
local density over ±2 mm; chance-match check by a 3 mm shift; Run 3 track
agreement also given at 1 mm.

### Recompute status

| Item | Content | Status |
|---|---|---|
| H1 | HL-LHC vertex finding at 0.17 mm, PV-Finder and AMVF: Table 3, Fig. 6, window scan, fake fraction | Done: [hllhc_final_numbers](../evaluation/hllhc_final_numbers_2026-11-12.md) |
| H2 | sigma_vtx-vtx on the counted peaks; Fig. 7(a) caption | Done: same |
| H3 | Finder cap, truth-guided, oracle at 0.17 mm; Fig. 7.2, Table 4 | Done: same (chain extras) |
| H4 | Post-GNN categories at PU200 and vs mu | Done: same <!-- t-dependent --> |
| H5 | HS selection, clean-HS and HS-track fractions | Done: same <!-- t-dependent --> |
| H6 | HS resolution, 3D adaptive fit | Done: same <!-- t-dependent --> |
| H7 | Efficiency vs mu, n_trk, density; wrong/unassigned rates; Eq. (4) | Done: same |
| H8 | Everything with the frozen s8 | Done: same |
| R1 | Run 2 MC vertex finding at 0.17 mm | Done: [lowpu_final_numbers](../evaluation/lowpu_final_numbers_2026-11-12.md) |
| R2 | Run 2 MC GNN chain on e400 peaks, t = 0.97 | Done: same |
| R3 | Run 2 and Run 3 data vertex finding; Fig. 10 panels | Done: same |
| R4 | Run 3 data association on the R3 events | Done: same |

Open items:
- The HL-LHC t may change after the rescan under the >= 2-track rule
  (`outputs/11_12_2026_output/hllhc_final/t_rescan_min2/`). The marked
  numbers then need a refresh.
- Not recomputed at low pile-up: the truth-guided / oracle bounds of the
  July chain (81.9%, quoted in the paper text); the HS selection with the
  Athena rule and the 3D-fit HS resolution, which exist only for HL-LHC.

---

## Part 3. Datasets

### HL-LHC

Pile-up = `ActualNumOfInt` read from every ROOT file. Script:
`scripts/hllhc_dataset_inventory.py`. Outputs in
`outputs/11_11_2026_output/datasets/`: `hllhc_dataset_inventory.json` and
`mu_{training,validation,test,held-out_*}.{png,pdf}`. Samples: mc21 14 TeV
PhPy8EG A14 ttbar (hdamp 258.75), e8481_s4494, ITk, in-time pile-up only.
Track selection follows the ITk paper Table 6
(`data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/README`).

| File | DSID / r-tag | Events | Pile-up | Role |
|---|---|---:|---|---|
| SingleLep | 601229 r16438 | 99,800 | 190-210, mean 200.0 | PV-Finder training pool |
| SingleLep | 601229 r16633 | 99,800 | 190-210, mean 200.0 | PV-Finder training pool |
| all-had part 1 | 601237 r16633 | 1,421,400 | 190-210, mean 200.0 | PV-Finder training pool |
| all-had part 2 | 601237 r16633 | 1,498,600 | 190-210, mean 200.0 | PV-Finder training pool |
| all-had part 3 | 601237 r16633 | 1,200,000 | 190-210, mean 200.0 | PV-Finder training pool |
| all-had part 4 | 601237 r16633 | 854,000 | 190-210, mean 200.0 | PV-Finder training pool |
| **PU200 pool** | | **5,173,600** | integers 190-210 | |
| SingleLep | 601229 r16443 | 100,000 | flat 0-200, mean 99.6 | entries 0-24,999: evaluation of every HL-LHC result; entries >= 25,000: GNN training, validation and reserved test |
| SingleLep | 601229 r16638 | 99,800 | flat 0-200, mean 99.6 | entries 0-24,999: second evaluation sample before the GNN; all entries: GNN v2 training (89,820) and reserved test (9,980) |

**Pile-up shape of the PU200 pool.** Uniform over the 21 integer values
190-210, about 237k events each in the training role. Two production
features: mu = 196 is doubled (440k) and mu = 197 nearly empty (32k);
mu = 203 is high (308k) and mu = 204 low (165k).

**Pile-up shape of the flat-mu files.** About 500 events per integer mu
from 0 to 200; some values hold twice that and a neighbour none. In 10-unit
bins it is flat to ±10%.

**PV-Finder split (s8)** (seed-42 permutation of intervals, 96 / 3 / 1%;
event equivalents = intervals / 12):

| Role | Intervals | Event equivalents | Pile-up |
|---|---:|---:|---|
| Training | 59,599,872 | 4,966,656 | 190-210, mean 200.0 |
| Validation | 1,862,496 | 155,208 | 190-210, mean 200.0 |
| Test (unused) | 620,832 | 51,736 | 190-210, mean 200.0 |
| Evaluation before the GNN | r16443 and r16638 entries 0-24,999 each (PU200: 1,920 and 1,888 events with 185 <= mu <= 215) | 25,000 per file | flat 0-200 |
| Evaluation after the GNN | r16443 entries 0-24,999 (PU200: 1,920 events) | 25,000 | flat 0-200 |

The validation and test roles have the same pile-up shape as training,
because the split is a random permutation of a single PU200 pool. The density
model never saw r16443 or r16638.

**GNN samples.** The full-mu GNNs train on truth-candidate graphs (truth
vertices as candidates; no PV-Finder peaks in the training graphs).

| Model | Training | Validation | Test |
|---|---|---|---|
| full-mu v2 (frozen) | 149,820 events, mu 0-200: v1's 60,000 r16443 entries >= 25,000 (about 3,000 per 10-mu bin) + 89,820 r16638 entries (about 4,500 per 10-mu bin) | v1's 7,500 r16443 entries: the 1,500-event monitor selects the checkpoint, all 7,500 set t | r16443 entries 0-24,999; reserved, untouched: v1's 7,500 r16443 test entries and 9,980 r16638 entries (about 500 per 10-mu bin) |
| full-mu v1 (superseded) | 60,000 r16443 entries >= 25,000, stratified in 10-mu bins | 7,500 (1,500 monitor) | 7,500 reserved |

- The v2 split is in
  `outputs/11_11_2026_output/gnn_fullmu_v2/split_manifest.json` (split seed
  20261111; it keeps v1's split, recorded in
  `outputs/09_24_2026_output/gnn_fullmu_v1/split_manifest.json`, seed
  20260924).
- Checks in the manifest and in `gnn_v2_freeze/hygiene.json`: no r16443
  truth-z or hard-scatter fingerprint occurs in r16638, and the GNN never
  saw r16443 entries 0-24,999.
- The note's earlier GNN v4 (180,000 events of 601237 part 1) is not used for
  any result.

### Low pile-up

There is no Run 3 MC. The simulation is called Run 2 MC, and the Run 3 data
plots compare with it.

| Sample | Source | Events used | Pile-up | Role |
|---|---|---|---|---|
| Run 2 MC | `/share/lazy/qibinlei/recoTrackNPV_jets_pubindices_1000bins_incbounds_Target_Y_split.h5`, 51,000 events | events 0-35,699 | | e400 training; GNN mu60 v2 training (35,700 graphs) |
| | same | events 35,700-48,449 | | e400 validation; GNN mu60 v2 validation (12,750 graphs) |
| | same (GNN graphs from `recoTracks_incamvfassoc.h5`) | events 48,450-50,999 (2,550; 72,189 truth vertices with >= 2 tracks) | mean 38.9 | test: every Run 2 MC result (R1, R2) |
| Run 2 data | 2018, 13 TeV ZeroBias, run 364076, file `_000002` | entries 0-2,499 (2,500) | mean 42.5 | vertex finding (R3) |
| Run 3 data | `data/run3/file_3.root` | first 2,500 loader-accepted entries in 0-3,158 | mean 50.0 | vertex finding (R3) and association (R4), same events |

Neither the density model nor the GNN saw the 2,550 Run 2 MC test events.
There is no separate validation sample for t at low pile-up; t is one
parameter on 72k vertices, and the even and odd halves of the test events
give the same choice.
