# Paper draft of 11 November 2026: answers, threshold audit, datasets

Draft: `final_paper/current_paper/paper/PVFinderPaper_11_11_2026.pdf`
(32 pages). The line numbers below are the draft's own line numbers.

Decisions referred to as D# are in
[meeting_2026-11-11_plan.md](meeting_2026-11-11_plan.md). Repo paths are
relative to the repo root. "Note" means the internal note, GitLab main
91b6f49.

Status keys:
- **Answered**: settled by code, a checkpoint or a log.
- **D#**: settled by a decision; may also need a recompute.
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
**Answered.** It is right for HL-LHC (latent 4 x 1000, k = 20 graphs) with
the corrections below; for Run 2/3 it needs changes.

HL-LHC corrections:
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
- latent 1 x 1000 (`run_eval_pvf.py:95-108`);
- the U-Net v1 design (Q10);
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
  rebuild of the 5.17 M-event HDF5 pool, and a retrain (v6 took 31.3 h).
  Not started.

### Q4 (l. 223). U-Net v2 details.
**Answered** (`src/pv_finder/models/unet_v2.py`, `build_model` in
`src/pv_finder/training/hllhc_helpers.py:55-87`, v6 config).

Yes, it is the Appendix E design. Specification:
- **Channels:** 280 at every stage (`n_unet_channels: 280`). Input:
  4 latent channels.
- **Encoder:**
  - ConvBNReLU(4 -> 280, kernel 15) at 1000 bins;
  - residual block (kernel 5) at 1000;
  - max-pool 2, residual block (kernel 5) at 500;
  - max-pool 2, residual block (kernel 5) at 250. That is the bottleneck:
    two down-sampling stages, factor 4.
- **Decoder:** nearest-neighbour upsample x2 + Conv(kernel 3) + BN + ReLU,
  then an additive skip and a residual block (kernel 5). Done twice:
  250 -> 500 (skip from the 500-bin encoder), then 500 -> 1000 (skip from
  the second 1000-bin encoder block).
- **Head:** ConvBNReLU(280 -> 280, k 5), ConvBNReLU(280 -> 140, k 3),
  Conv(140 -> 1, k 1), softplus output.
- **Residual block:** Conv + BN + ReLU + Dropout, plus the identity.
- **Normalisation and activations:** BatchNorm after every convolution
  except the last 1x1. ReLU throughout the U-Net.
- **Dropout:** p = 0.25 after the ReLU of every convolution block (encoder,
  decoder, upsampling, head). None in the MLP.
- **MLP:** 7 -> 128 x 5 -> 4000, LeakyReLU slope 0.01, softplus output.
  Padded tracks are masked, the result is summed over tracks and scaled by
  0.001 (`autoencoder_models.py:214-352`).
- **Parameters:** 3,547,993 = MLP 583,072 + U-Net 2,964,921 (counted from
  the v6 build).

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
- Absent from the forward pass. `phase1_forward` runs only the MLP; the 4
  latent channels are summed into one 1000-bin histogram and compared with
  the target (`hllhc_helpers.py:185-195`;
  `train_mlp_hist_then_e2e.py:106-115`).
- Its parameters are also frozen (`requires_grad = False`,
  `train_hllhc_e2e.py:81-90`) and are switched back on for Phase 2
  (l. 128).

### Q7 (l. 250). Datasets, checkpoint choice, warm-up, time, hardware.
**Answered.**
- **Training sample:** the six PU200 files of Part 3, 5,173,600 events
  (`configs/vertex_finding/config_hllhc_alleta_v6_mse.yml`).
- **Split:** seed-42 random permutation of the 62,083,200 *intervals*:
  59,599,872 train / 1,862,496 validation / 620,832 test
  (`collectdata_poca_KDE.py:88-104`; v6 log). Twelve intervals of one event
  can land in different roles. The test intervals are not used for any
  result: results come from the independent files r16443 and r16638.
- **Checkpoint:** last epoch, Phase 2 epoch 2
  (`hllhc_alleta_v6_mse_2ep_phase2_epoch_2_fullstate.pth`). It is also the
  lowest validation loss: 0.015755 against 0.016638 after epoch 1
  (`outputs/logs/hllhc_alleta_v6_mse_2ep_20260731_111446.log`).
- **Learning rate:**
  - Phase 1: constant 1e-3, 2 epochs.
  - Phase 2: per-step linear warm-up from 0.01 x 1e-4 to 1e-4 over 3,000
    steps, then per-step cosine decay to 1e-6 over 2 epochs
    (`build_phase2_scheduler`, `hllhc_helpers.py:100-160`).
- **Other settings:** Adam, betas (0.9, 0.999), batch 128, gradient-norm
  clip 1.0, MSE loss.
- **Time and hardware:** 31.3 h on one NVIDIA A100-PCIE-40GB
  (2026-07-31 11:14 to 2026-08-01 18:31). Phase 1: 8.3 h. Phase 2: 23.0 h.
  465,624 steps per epoch.

### Q8 (l. 262). Peak width, sigma_PV, height.
**Answered** (`efficiency_res_optimized_atlas.py:16-190`).
- **Width:** sqrt of the amplitude-weighted variance of the bin index over
  the whole above-threshold region (up to the split point), times 0.04 mm.
- **sigma_PV:** this width is used directly as sigma_PV in the first edge
  feature, (z0 - z_PV) / sqrt(sigma(z0)^2 + sigma_PV^2)
  (`graph_construction.py:105-146,291-360`;
  `pu200_chain_graphs.py:201-217`).
- **Height:** the maximum bin value of the region, the raw predicted density
  with no normalisation (`peakvals`). It is the second vertex-node feature.
- **Training graphs differ:** they use truth z, sigma_PV from the resolution
  model (`hllhc_alleta`) and a height from the same Gaussian-in-bin recipe
  as the targets (`root_to_graphs.py:129`,
  `graph_construction.py:149-168`). This is the training/inference mismatch
  the note discusses.

### Q9 (l. 269). Final peak-finding configurations; is the local centroid also used at low pile-up?
**D1.** See Part 2.
- Currently the local centroid is used only for HL-LHC. Every low-pile-up
  result used the full-region weighted mean.
- The "0.50 for the HL-LHC resolution fit" in the note is not what produced
  the quoted 0.229 mm: the evaluation at integral 0.30 gives
  0.2290 ± 0.0032 mm (Q21).

### Q10 (l. 281). Run 2/3 density model.
**Answered for the vertex-finding model. But three different low-pile-up
density models are used across the paper** (Part 2, item L1).

Vertex-finding model:
`model_weights/03_24_2026/reproduction_T2HIST_400ep_T2KDE100_K2H150_epoch_300_fullstate.pth`.
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
- **Parameters:** 358,713 (from the checkpoint).
- **Inputs:** the same seven features as HL-LHC.
- **Training:** 300 of a planned 400 end-to-end epochs (checkpoint epoch
  300). Constant lr 1e-3, no warm-up or schedule, batch 128, MSE, dropout
  0.25 (`configs/vertex_finding/config_T2HIST_400ep_03_24_2026.yml`;
  JOURNAL 2026-03-24, 2026-04-15).
- **Data:** 51,000 events (`/share/lazy/qibinlei/..._Target_Y_split.h5`),
  split 0.70 / 0.25 / 0.05. The 2,550 test events are the 0.05.
- **Analytic KDE: yes, still used.** The end-to-end model was initialised
  from an MLP trained to reproduce the analytic KDE
  (`reproduction_KDE_A_z_400ep_epoch_100`) and a U-Net trained KDE ->
  histogram (`reproduction_KDE2HIST_400ep_epoch_150`).
- **Target preset:** the targets are Qi Bin's file. The file carries no
  attributes. Our registry records the upstream constants as preset `run3`
  (A = 0.23817, B = 0.49491, C = -0.00079 mm). Only Qi Bin can confirm what
  was used to build that file.
- **Why not the two-phase procedure:** historical. This model is the
  reproduction of the earlier work. A two-phase, KDE-free low-pile-up model
  already exists: `e2e_mlpHist50_e2e400_1latent_mse` (50 epochs MLP ->
  histogram, then 400 end-to-end, in `model_weights/`). It is the "PVF-e400"
  behind the current low-pile-up GNN numbers. Freezing it, or retraining
  one, would answer this reviewer point (see L1).

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
  configs does not describe the trained model.
- **Parameters:** 39,521 (both checkpoints).
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
  `docs/evaluation/metric_definitions.md`; tested in
  `tests/test_amvf_fair_matching.py`). No recalculation is needed for the
  definition.
- **Fake fraction** at 0.1694 mm on the 1,920 events
  (`outputs/11_10_2026_output/small_model_eval/v6/r16443/eval_results.pkl`):

  | | Fake / reco | (Fake + split) / reco |
  |---|---|---|
  | PV-Finder | 18.55% (35,971 / 193,945) | 19.07% |
  | AMVF | 20.11% (37,797 / 187,908) | 20.30% |

- These will move slightly at 0.17 mm (D2).

### Q13 (l. 367). One window for everything?
**D2.**
- 0.17 mm for every vertex-matching result. 0.1 mm for hard-scatter
  results only, as in the ITk paper.
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
  constraint, on the tracks the GNN assigns to the HS candidate.
- **Results** on the 1,915 events where both algorithms find the HS:

  | GNN tracks used | PV-Finder + GNN | AMVF |
  |---|---|---|
  | All assigned | 9.27 ± 0.25 um | 10.25 ± 0.30 um |
  | t >= 0.98 | 9.72 ± 0.26 um | 10.25 ± 0.30 um |

  (`docs/evaluation/itk_hs_resolution_2026-11-09.md`, section 2026-11-11).
- With the frozen GNN and the single t (D1, D4), the quoted number is the
  t = 0.98 one.

### Q15 (l. 396). Local pile-up density results?
**Partly answered.**
- HS resolution vs local density exists
  (`outputs/11_09_2026_output/itk_hs_resolution/hs_resolution_vs_density.png`).
- HS reconstruction vs density: `hs_reconstruction_2026-09-17.md`.
- The density counts stored truth vertices within ±2 mm.
- **New result (D6):** vertex-finding efficiency and post-GNN categories vs
  local density.

### Q16 (l. 428). F1 inputs: exact definitions.
**Answered** (`src/gnn/evaluation/threshold_scan.py:149-164,240-252`;
`chain_scan.py:96-130`).
- **Correct track:** a track assigned to a reconstructed vertex whose
  dominant truth vertex is the track's own truth vertex.
- **Efficiency:** correct tracks / all tracks with a truth vertex. Tracks
  left unassigned count in the denominator.
- **Purity:** correct tracks / assigned tracks that have a truth vertex.
- **Tracks without truth association** are excluded from both; they enter
  only the assigned fraction.

### Q17 (l. 434). Wrong-assignment and unassigned-track metrics.
**New result (D6).**
- The counts exist (`n_assigned`, `n_assigned_truth`, `n_correct`).
- Wrong-assignment rate = (n_assigned_truth - n_correct) / n_assigned_truth
  = 1 - purity.
- Unassigned truth fraction = 1 - n_assigned_truth / n_truth_tracks.
- "Correctly left unassigned" (tracks without truth, or whose vertex was not
  found, that stay unassigned) is not computed anywhere yet.

### Q18 (l. 445). Significance-based association, Eq. (4).
**New result.**
- The existing baseline, "z0 within 3 sigma" (33.06% clean/truth),
  uses |z0 - z_peak| < 3 sigma(z0) only, with no sigma_PV
  (`src/gnn/diagnostics/hs_chain_study.py:105`).
- Eq. (4) with sqrt(sigma(z0)^2 + sigma_PV^2), the same combination as the
  GNN edge feature, must be run for the paper.
- The draft's change to sigma_PV^2 in the denominator is consistent with
  the code (`graph_construction.py:128`).

### Q19 (l. 453). Why 25,000 events and the 1,920 PU200 events?
**Answered: compute budget, not physics.**
- The 25,000 are the first entries of the flat-mu file r16443
  (100,000 events, mu 0-200, about 500 per integer mu, mean 99.6).
- The 1,920 are those with 185 <= mu <= 215: actual range 185-200,
  mean 192.5.
- The whole file has 7,680 events in that window; r16638 is a second
  independent file.
- The full-mu GNN uses r16443 entries >= 25,000 for training (disjoint, see
  Part 3), so extending the evaluation beyond 25,000 r16443 entries would
  overlap the GNN training. Use r16638 for more statistics.

### Q20 (l. 470). Chance-match check, window scan; efficiency vs mu, n_trk, separation.
**D6 / new result.**
- Recommendation: show the window scan as an auxiliary figure (the margin
  is +3.1 to +3.4 points for 0.10-0.20 mm). Keep the 3 mm-shift check in
  the note only.
- Efficiency vs mu: derivable from the saved per-event results (new plot).
- Efficiency vs truth n_trk and vs vertex separation: not available;
  new result (D6).

### Q21 (l. 481). sigma_vtx-vtx threshold; AMVF value and fit.
**Answered.**
- At the final operating point (integral 0.30, height 0.03, local centroid,
  300 bins over ±6 mm), PV-Finder sigma_vtx-vtx = 0.2290 ± 0.0032 mm
  (`outputs/11_10_2026_output/small_model_eval/v6/r16443`; its
  `INVOCATION.txt` shows `--integral-threshold 0.30`). The same value was
  found on 2026-08-05 (`outputs/08_05_2026_output/eval_v6_mu_window`).
- So the note's "0.50 for the resolution plot" is not what produced
  0.229 mm, and no separate threshold is needed.
- AMVF: 0.305 ± 0.006 mm, same fit code and binning, same 1,920 events,
  vertices with n_trk >= 2 (`run2_vs_run4_resolution_2026-11-09.md`).

### Q22 (l. 500). Merged/Split after the GNN at mu ~ 200; drop candidates without tracks?
**New result (D4/D6)**, recomputed with the frozen GNN.
- Saved so far at t = 0.98: Clean 146,269, populated vertices 175,492,
  Fake 303. Merged and Split at mu ~ 200 are not in the saved summaries.
- Recommendation: remove candidates without tracks from the output, not
  only from the rates. A vertex with no tracks cannot be written as an
  xAOD vertex with tracks, and the note already uses this "drop-empty"
  convention.

### Q23 (l. 514). GNN thresholds at mu ~ 200; one working point?
**D1: one working point, t = 0.98.**
- Current: t = 0.98 for vertex classification and t = 0.95 for HS
  identification.
- At t = 0.98, reconstructed-and-selected HS is 96.30% against 94.27% at
  0.95, so a single t = 0.98 loses nothing
  (`hs_chain_studies_2026-11-09.md`).
- To re-tune on the validation set once the GNN is retrained (D4).

### Q24 (l. 528). AMVF Run 2 sigma on a different subset; AMVF efficiency.
**Answered.** Both are now on the same 2,550 events with the same code:

| | sigma_vtx-vtx | Efficiency at 0.416 mm | at 0.1694 mm |
|---|---|---|---|
| PV-Finder | 0.410 ± 0.008 mm (0.416 published) | 91.59 ± 0.10% | 80.85 ± 0.14% |
| AMVF | 0.789 ± 0.015 mm | 88.18 ± 0.12% | 75.67% |

Sources: `run2_vs_run4_resolution_2026-11-09.md`,
`data_mc_validation_2026-11-09.md`. To recompute at 0.17 mm (D2).

### Q25 (l. 535). Low-pile-up GNN uses integral 0.2, vertex finding 0.50.
**D1; must be recalculated.**
- The difference is more than the threshold. The low-pile-up GNN numbers
  (77.43%, t = 0.99) were made on peaks from a different density model,
  "PVF-e400" (`e2e_mlpHist50_e2e400_1latent_mse_phase2_epoch_400`),
  with integral 0.2 and the full-region mean
  (`docs/evaluation/vertex_association.md:240-376`; JOURNAL 2026-07-12/14).
- The low-pile-up vertex-finding numbers use the reproduction ep300 model
  at integral 0.5.
- The GNN results must be regenerated on the frozen low-pile-up model and
  configuration.

### Q26 (l. 548). GNN thresholds in Run 2 simulation.
**Answered; D1 for the final value.**
- t = 0.99: vertex classification.
- t = 0.95: best track F1, 0.874. A diagnostic, not a working point.
- t = 0.5: HS identification (98.98% vs AMVF 98.75%; JOURNAL 2026-07-14,
  "Working points: t=0.99 (mu60 vertices), t=0.98 (PU200), t=0.5 (HS-ID)").
- With one working point, the HS identification is recomputed at t = 0.99.

### Q27 (l. 574). Run 3 data association: t, peak threshold, event subset.
**Answered.**
- Settings: t = 0.98, PV-Finder integral 0.5, no height cut, 1 mm agreement
  window (`outputs/07_14_2026_ttva_mu60v2/run3_eval/run3_summary.json`).
- **Density model:** the two-stage KDE model (T2KDE ep130 + K2H ep190), a
  third low-pile-up density model.
- **Events:** the 2,000 events are a seed-42 random sample of `file_3.root`
  (`data/run3/cache_file3_2000ev_seed42.npz`, entries up to 936,586). They
  are **not** a subset of the 2,500 vertex-finding events (entries
  0-3,158): only 8 events are in both.
- To recompute with the frozen model at t = 0.99 (D1).

### Q28 (Fig. 10 caption). "integral=0.2" in panel titles vs 0.5 in the note.
**Answered: 0.5 was used. The titles are stale.**
- Re-running the ep300 checkpoint on 25 of the Run 2 MC test events
  reproduces the stored peak lists in 23/25 events at integral 0.5 and in
  7/25 at 0.2. The remaining 2 differ only by CPU/GPU rounding.
- The stored pair lists equal all pairs of the stored peaks, so counts and
  sigma fit use the same peaks (all three samples:
  `outputs/04_16_2026_output/reproduction_ep300_{mc,run2,run3}`).
- The "integral=0.2" title comes from the April 2026 plotting code, which
  carried the then-default label. Regenerate the panels.

### Q29 (l. 652). Remove the AMVF dependence; which sample for `hllhc_alleta`; Run 2/3 parameters.
**Answered.**
- AMVF dependence: see Q3.
- `hllhc_alleta` was fitted on the **full-|eta| (|eta| < 4)** sample:
  601229 r16633, 99,800 events, 2.31 M truth vertices with n_trk >= 2
  (`resolution_presets.py`; `outputs/07_31_2026_output/amvf_resolution_alleta/`).
- Run 2/3 parameters: A = 0.23817443 mm, B = 0.49491396,
  C = -0.000787436 mm (preset `run3`), with the caveat of Q10.

---

## Part 2. Threshold inventory and proposal

### Current thresholds

| Quantity | Result | Value now |
|---|---|---|
| Amplitude tau | all | 0.01 |
| Minimum width | all | 3 bins |
| Integral | HL-LHC vertex finding, sigma_vtx-vtx, GNN graphs (hs_chain metadata) | 0.30 |
| Integral | HL-LHC sigma fit, as stated in note §4.1 / D.112, D.173, D.192 | "0.50" (not what produced 0.229 mm, Q21) |
| Integral | `pu200_chain_graphs.py` default | 0.40 (stale; the note's runs used 0.30) |
| Integral | low-pile-up vertex finding (MC, Run 2, Run 3 data), Run 3 GNN | 0.5 |
| Integral | low-pile-up GNN chain (MC) | 0.2 |
| Integral | satellite / bump study (note D.388) | 0.20 |
| Minimum height | HL-LHC | 0.03 |
| Minimum height | low pile-up | none |
| Position estimator | HL-LHC | local centroid ±3 bins |
| Position estimator | all low pile-up | full-region weighted mean |
| Density model | low-pile-up vertex finding | reproduction ep300 |
| Density model | low-pile-up GNN | PVF-e400 |
| Density model | Run 3 GNN | T2KDE130 + K2H190 |
| GNN t | HL-LHC vertex categories | 0.98 |
| GNN t | HL-LHC HS identification | 0.95 |
| GNN t | HS adaptive fit | none (9.27 um) or 0.98 (9.72 um) |
| GNN t | low-pile-up vertex categories | 0.99 |
| GNN t | low-pile-up best F1 | 0.95 |
| GNN t | low-pile-up HS identification | 0.5 |
| GNN t | Run 3 data | 0.98 |
| Window | HL-LHC vertex finding | 0.1694 mm |
| Window | HS reconstruction, selection, resolution | 0.1 mm |
| Window | full-chain one-to-one, finder cap, truth-guided, current Fig. 7.2 | 0.5 mm |
| Window | oracle | 0.5 mm in the note, 1.0 mm code default |
| Window | single-vertex residual assignment | 1 mm |
| Window | Run 2 MC | fitted 0.416 mm |
| Window | Run 2 / Run 3 data, AMVF reference | own fitted sigma, 0.368 / 0.348 mm |
| Window | Run 3 GNN agreement | 1 mm |
| Other | local density | ±2 mm (definition, not a window) |
| Other | chance-match check | 3 mm shift |
| Other | Clean | >= 70% track purity |

### Proposed configuration: two sets in total

| | HL-LHC (mu ~ 200) | Run 2/3 (low pile-up) |
|---|---|---|
| Density model | frozen HL-LHC model (D3; v6 unless replaced) | **one** model for every low-pile-up result (L1) |
| Amplitude, width | 0.01, 3 bins | 0.01, 3 bins |
| Integral | 0.30 | 0.50 |
| Minimum height | 0.03 | none |
| Position | local centroid ±3 bins | local centroid ±3 bins (L2) |
| GNN t (all uses) | 0.98 (re-tune once on validation after D4) | 0.99 |
| Vertex window | 0.17 mm | 0.17 mm (L3) |
| HS window | 0.1 mm | 0.1 mm |
| Residual assignment | 1 mm (assignment radius for the residual measurement only) | 1 mm |

**Why the HL-LHC set:**
- 0.30 / 0.03 / centroid is the operating point of every HL-LHC efficiency
  and GNN result. It already gives sigma_vtx-vtx = 0.229 mm (Q21), so the
  0.50 can go.
- One t = 0.98 is at least as good as 0.95 for the HS (Q23).

**Why the low-pile-up set:**
- 0.50 is what three of the four low-pile-up studies used: MC and data
  vertex finding, Run 3 GNN.
- The integral scan of 2026-04-15 (JOURNAL) shows that 0.2 buys
  +3.2 points of efficiency on Run 2 MC at +1.4 fakes per event. If maximum
  efficiency matters more than continuity, 0.2 is the alternative.

**Decisions for Mattia:**
- **L1.** Choose one low-pile-up density model.
  - The ep300 reproduction model: KDE-initialised; behind all low-pile-up
    vertex-finding and data results.
  - Or PVF-e400: two-phase and KDE-free, which answers Q10; behind the
    low-pile-up GNN.
  - Recommendation: run both through the low-pile-up benchmark (2,550 MC
    events, Run 2 and Run 3 data; minutes each). Pick e400 if it is equal
    or better, since it removes the KDE from the paper. Otherwise ep300.
- **L2.** Use the local centroid at low pile-up too: one estimator in the
  paper. Alternative: keep the full-region mean there and explain two.
- **L3.** The same derivation as at HL-LHC gives 3 x p68 = 0.208 mm on Run 2
  MC (p68 69.5 / 66.8 um). D2 says 0.17 mm everywhere. Using 0.17 mm keeps
  one number; the paper should then say it is the HL-LHC-derived value.

### Results to recompute

**HL-LHC:**
- H1. All vertex-finding numbers at 0.17 mm, PV-Finder and AMVF: Table 3,
  Fig. 6, window scan. Includes the fake fraction of Q12.
- H2. sigma_vtx-vtx: no recompute. Fix the text and the Fig. 7(a) caption
  to 0.30.
- H3. Full-chain one-to-one, finder cap, truth-guided and oracle at
  0.17 mm (now 0.5 / 1.0 mm). Fig. 7.2 and Table 4.
- H4. Post-GNN categories, including Merged and Split, at mu ~ 200 and vs
  mu, with the frozen GNN and t = 0.98 (Q22).
- H5. HS selection, clean-HS fraction and HS-track fraction at t = 0.98
  with the frozen GNN. The 0.95 rows are dropped.
- H6. HS resolution with the adaptive fit at t = 0.98, frozen GNN.
- H7. New for D6: efficiency vs mu, vs truth n_trk (PUB-note style) and vs
  local density; track-level wrong/unassigned rates (Q17); the Eq. (4)
  significance baseline (Q18).
- H8. All of H1-H7 again if D3 freezes a model other than v6.

**Low pile-up:**
- R1. Run 2 MC vertex finding (PV-Finder and AMVF, same 2,550 events) at
  0.17 mm with the chosen model (L1) and estimator (L2).
- R2. Low-pile-up GNN chain: rebuild the inference graphs from the frozen
  model at 0.50 and rerun at t = 0.99: categories, F1, HS identification
  (Q25, Q26).
- R3. Run 2 and Run 3 data vertex finding, AMVF-reference recovery at
  0.17 mm, and the Fig. 10 panels with correct titles (Q28).
- R4. Run 3 data association with the frozen model at t = 0.99, on the same
  2,500 events as the vertex finding (Q27).

---

## Part 3. HL-LHC datasets

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
| all-had part 1 | 601237 r16633 | 1,421,400 | 190-210, mean 200.0 | PV-Finder pool; GNN v4 training (entries 0-180k) |
| all-had part 2 | 601237 r16633 | 1,498,600 | 190-210, mean 200.0 | PV-Finder pool; GNN v4 validation (5k graphs) |
| all-had part 3 | 601237 r16633 | 1,200,000 | 190-210, mean 200.0 | PV-Finder training pool |
| all-had part 4 | 601237 r16633 | 854,000 | 190-210, mean 200.0 | PV-Finder training pool |
| **PU200 pool** | | **5,173,600** | integers 190-210 | |
| SingleLep | 601229 r16443 | 100,000 | flat 0-200, mean 99.6 | held-out evaluation (entries 0-24,999); full-mu GNN train/val/test (entries >= 25,000) |
| SingleLep | 601229 r16638 | 99,800 | flat 0-200, mean 99.6 | held-out evaluation |

**Pile-up shape of the PU200 pool.** Uniform over the 21 integer values
190-210, about 237k events each in the training role. Two production
features: mu = 196 is doubled (440k) and mu = 197 nearly empty (32k);
mu = 203 is high (308k) and mu = 204 low (165k).

**Pile-up shape of the flat-mu files.** About 500 events per integer mu
from 0 to 200, with some values at 320-1,000. In 10-unit bins it is flat
to ±10%.

**PV-Finder split** (seed-42 permutation of intervals, 96 / 3 / 1%; event
equivalents = intervals / 12):

| Role | Intervals | Event equivalents | Pile-up |
|---|---:|---:|---|
| Training | 59,599,872 | 4,966,656 | 190-210, mean 200.0 |
| Validation | 1,862,496 | 155,208 | 190-210, mean 200.0 |
| Test (unused) | 620,832 | 51,736 | 190-210, mean 200.0 |
| Held-out evaluation | r16443 entries 0-24,999 (1,920 with 185 <= mu <= 215), r16638 likewise | 25,000 per file | flat 0-200 |

The validation and test roles have the same pile-up shape as training,
because the split is a random permutation of a single PU200 pool.

**GNN samples:**

| Model | Training | Validation | Test |
|---|---|---|---|
| v4 no-aug (note) | 180,000 all-had events, 601237 part 1, mu 190-210 | 5,000 graphs, part 2 | r16443 PU200 slice |
| full-mu v1 | 60,000 events of r16443 entries >= 25,000, stratified in 10-mu bins, about 3,000 per bin, mu 0-200 | 7,500 (1,500 used for the learning curve) | 7,500 reserved |

The full-mu v1 split is recorded in
`outputs/09_24_2026_output/gnn_fullmu_v1/split_manifest.json`
(split seed 20260924). Entries are disjoint from the evaluation entries
0-24,999 of the same file.
