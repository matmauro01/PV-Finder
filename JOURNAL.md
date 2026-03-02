# Journal

Append-only log. Each entry: date, what was done, why. Never edit old entries.

---

## 2026-02-23 — Repository bootstrap

Set up clean PV-Finder repo from scratch, migrating from bloated atlas-pvfinder.

Created:
- `CLAUDE.md` — agent instructions and workflow rules
- `.gitignore` — excludes venv, data, model weights, caches
- `docs/` — wiki structure for current-truth documentation
- `JOURNAL.md` — this file
- `requirements.txt` — project dependencies

Decision: dual documentation system. `docs/` is the wiki (rewritten as things change),
`JOURNAL.md` is the historical log (append-only). This keeps docs lean while preserving
full context of what happened and why.

---

## 2026-02-23 — Source and docs structure

Created `src/pv_finder/` package with six submodules mirroring the old atlas-pvfinder
codebase: models, data, training, evaluation, diagnostics, utils. Each folder has a
matching wiki page in `docs/`.

Structure decision: organize by function (what the code does) rather than by model
(PVF vs GNN). Rationale: evaluation and diagnostics scripts work across both models,
and shared utilities (peak finding, constants) don't belong to either.

---

## 2026-02-23 — Migrated vertex finding training code

Verbatim copy from atlas-pvfinder/mattia_finder into new structure:

- `src/pv_finder/models/`: autoencoder_models.py (1628 lines, needs splitting), alt_loss_A.py
- `src/pv_finder/data/`: collectdata_poca_KDE.py, h5_dataset.py
- `src/pv_finder/training/`: 3 train scripts, training loop, weight initializer
- `src/pv_finder/utils/`: utilities.py, efficiency.py, jagged.py
- `configs/vertex_finding/`: 4 YAML configs (T2KDE, KDE2HIST, T2HIST, T2HIST combined)

Files are unmodified copies. Imports will break until paths are updated.

Confirmed: standard training uses KDE channel A_z only (out_idx=0, n_latent_channels=1).
A 4-channel variant exists in old repo's first_troubleshooting/ but is experimental.

Updated docs: models/vertex_finding.md, data/monte_carlo.md, training/vertex_finding.md.

---

## 2026-02-23 — Docs restructured into subfolders

Replaced flat doc files with subfolder structure mirroring `src/`. Each subfolder
(models, data, training, evaluation, diagnostics) contains `vertex_finding.md` and
`vertex_association.md`. Data has `monte_carlo.md` and `run_3.md` instead.

This gives a 1:1 mapping: docs topic mirrors code folder, doc file mirrors the
two main systems (PVF vertex finding and GNN vertex association).

---

## 2026-02-23 — Pre-commit hooks added

Added `.pre-commit-config.yaml` enforcing:
- Conventional commit messages (feat, fix, refactor, test, docs, chore)
- Ruff lint + auto-format
- Max 500 lines per Python file
- Large file blocker (500KB)
- Trailing whitespace and end-of-file fixes

To activate: `pip install pre-commit && pre-commit install --hook-type commit-msg --hook-type pre-commit`

---

## 2026-02-23 — Import paths and hardcoded paths fixed

Updated all 12 Python source files: `from model.X` → `from pv_finder.X` imports.
Removed `sys.path.insert` hacks (package will be installed via `pip install -e .`).

Updated all 4 YAML configs and 3 training scripts:
- MLflow URI → `PV-Finder/mlruns`
- save_folder → `PV-Finder/model_weights/`
- output paths → `PV-Finder/outputs/`
- Data file path unchanged (points to existing `/share/lazy/` H5 file)

Added mlflow, pyyaml, torch_geometric to requirements.txt and pyproject.toml.
Created local gitignored dirs: model_weights/, outputs/, mlruns/.

---

## 2026-02-23 — Added KDE-free end-to-end training (Strategy C)

Migrated `train_mlp_hist_then_e2e.py` and `config_mlp_hist_e2e.yml` from
atlas-pvfinder/mattia_finder/end_to_end_training_random_init/attempt_4/.

This is a two-phase training that avoids KDE supervision entirely:
- Phase 1 (50 epochs): MLP trained on histogram targets, UNet frozen
- Phase 2 (400 epochs): Full MLP+UNet trained end-to-end on histograms

Key insight: pure random init with histogram MSE finds degenerate solutions
(peaks at bin 0). The MLP warmup phase gives the model a reasonable spatial
mapping before the UNet co-adapts. Documented in training/vertex_finding.md.

---

## 2026-02-23 — Migrated and split feature distribution comparison tool

Copied `compare_feature_distributions.py` (1352 lines) from
atlas-pvfinder/clean_run3/ and split it into 4 files, all under 500 lines:

- `src/pv_finder/data/feature_loading.py` (389 lines): constants, data loading,
  MC track decoding, Run 3 tensor building, feature collection
- `src/pv_finder/diagnostics/feature_plots_1.py` (387 lines): Figures 1--3
  (core params, 2D correlations, tensor distributions)
- `src/pv_finder/diagnostics/feature_plots_2.py` (369 lines): Figures 4--6
  (CDF/QQ, per-subevent stats, beam spot investigation)
- `src/pv_finder/diagnostics/compare_feature_distributions.py` (356 lines):
  thin entry point with CLI, summary computation, JSON export

Also created `__init__.py` for all subpackages (data, diagnostics, models,
training, utils, evaluation) and set up data symlinks for Run 3 ROOT files.

Data files (gitignored, in `data/`):
- `run3/file_2.root` → symlink to atlas_pvfinder (18G)
- `run3/file_3.root` → symlink to atlas_pvfinder (16G)
- `run3/cache_file3_2000ev_seed42.npz` — copied (29M)
- `monte_carlo/training_data.h5` → symlink to /share/lazy/ (48G)

---

## 2026-02-23 — Added data exploration notebook

Created `src/pv_finder/scratch/data_exploration.ipynb` for interactive exploration
of MC and Run 3 feature distributions. Covers basic stats, 1D/2D distributions,
track multiplicity, beam spot analysis, and direct ROOT file browsing via uproot.

Also reviewed all three diagnostics scripts (`compare_feature_distributions.py`,
`feature_plots_1.py`, `feature_plots_2.py`). Found them functionally correct with
only cosmetic issues: duplicated utility helpers across the two plot files, and a
dead loop in `feature_plots_2.py` line 289-292.

Updated: `docs/diagnostics/vertex_finding.md`, `CLAUDE.md` project map.

---

## 2026-02-23 — KDE model vs analytical comparison tool

Created 4 new files under `src/pv_finder/diagnostics/` for comparing the T2KDE model
predictions against analytically computed KDE:

- `analytical_kde.py` (283 lines): Pure numpy port of the ACTS_KDE_generation.ipynb
  algorithm. Per-track 2D Gaussian KDE with coarse (60×60) + fine (7×7) scan grid.
  ~9 sec/event without numba.
- `kde_model_inference.py` (229 lines): Loads pickled MaskedDNN model with sys.modules
  alias fix, runs batched inference. Handles re-padding from MASK_VAL (-999999) to
  model maskVal (-240.0).
- `kde_comparison_plots.py` (449 lines): Full-event overlays, per-vertex zoom plots,
  agreement summaries, residual distributions, MC vs Run 3 comparison.
- `compare_kde_model_vs_analytical.py` (389 lines): Entry point with CLI. Orchestrates
  data loading → inference → analytical computation → metrics → plots → JSON summary.

Key findings (200 validation events each):
- Analytical KDE matches H5 truth (kde_split) to Pearson r = 1.0000 (algorithm correct)
- Model Pearson r: 0.911 ± 0.103 (MC), 0.912 ± 0.112 (Run 3)
- Peak matching rate: 93.3% MC, 77.9% Run 3
- Integral ratio: 0.916 MC, 0.908 Run 3
- Model generalises well to Run 3 on Pearson r, but misses more peaks (domain shift)

Model used: `tracks2kde_KDE_A_z_epoch180.pyt` copied to `model_weights/`.

Corrected bug: The old compare_kde_theory_vs_model.py in atlas_pvfinder used
POCA pair-based KDE (wrong algorithm) and wrong feature encoding (d0/2, theta/3,
(phi+pi)/3). This new tool uses the correct per-track Gaussian KDE and correct
feature mapping.

---

## 2026-02-24 — KDE comparison audit: old amplitude suppression was a bug

Audited the KDE comparison tool against the old atlas_pvfinder/clean_run3 results.
The ~15x amplitude suppression on Run 3 reported there was entirely caused by three
feature encoding bugs in compare_kde_theory_vs_model.py and investigate_domain_shift.py:

- Ch0: fed d0/2 instead of raw d0
- Ch2: fed theta/3 instead of d0_err
- Ch3: fed (phi+pi)/3 instead of z0_err

For MC, these cancelled out (round-trip read/re-encode), so the model appeared to work.
For Run 3, the model received angular values where it expected uncertainties, causing
output collapse. The "60% OOD d0_err" finding was comparing d0_err against theta/3 --
different physical quantities.

The new PV-Finder code uses the correct mapping (confirmed by eval_run3_v2.py) and
shows the model works similarly on both datasets (Pearson r ~0.91, integral ratio ~0.91).
No bugs found in the new code.

---

## 2026-02-24 — Per-vertex visualization: audit and rewrite

Thorough audit of `src/pv_finder/diagnostics/per_vertex_visualization/` (5 files).
Found and fixed bugs, documentation errors, and structural issues.

**Bugs fixed:**
- xlabel "z [mm]" appeared on middle residual panel instead of bottom track panel
  when 3 panels were present (vertex_plots.py)
- Redundant `import numpy as np` inside `load_run3_amvf_vertices` (peak_matching.py)

**Critical: shared peak-finding algorithm (new file)**

Created `src/pv_finder/utils/peak_finding.py` — ported `pv_locations_updated_res`
from atlas_pvfinder. Replaced scipy `find_peaks` (different algorithm) in
`peak_matching.py`. Peak finding is now consistent across evaluation and diagnostics.

Removed conjoined-peak prominence splitting after calibration on 6 events showed it
doesn't improve F1 under 1-to-1 greedy matching (F1=0.861 with splitting vs 0.827
without, but splitting adds false peaks without meaningfully reducing misses). Each
contiguous above-threshold region now yields exactly one peak.

Tuned thresholds: threshold=0.01, integral_threshold=0.5, min_width=3.

**Plot improvements (vertex_plots.py):**
- Y-axis shows raw e2e histogram values (KDE/truth rescaled to e2e peak for comparison)
- All panels use `sharex=True` for proper x-axis alignment
- Colorbar uses `fig.colorbar(ax=list(axes))` to take space equally from all panels
- Removed axis tick marks (spines.top/right=False, all tick sizes=0)
- All truth/AMVF vertices visible in zoom window are drawn (focused=black, others=grey)

**Other changes:**
- Per-event output subdirectories: `mc/event0000/`, `run3/event0000/`
- Removed unused `SUBEVENT_CENTERS` constant from inference.py
- Documented Run 3 coordinate frame approximation (track z0 in detector frame,
  AMVF vertices beam-corrected; offset ~O(1mm), within matching window)

---

## 2026-02-24 — Migrated GNN Track-to-Vertex Association (TTVA)

Migrated the GNN TTVA system from atlas_pvfinder/tracks_to_vertex/ into PV-Finder.
This adds vertex association as a post-processing step after PVF peak finding:
given candidate PVs and reconstructed tracks, a GAT predicts which tracks belong
to which vertex via binary edge classification on a bipartite graph.

New files (7):

- `src/pv_finder/utils/constants.py` (~30 lines): Centralized physics constants
  (PT_SCALE, PV resolution fit params, bin geometry, thresholds).
- `src/pv_finder/models/ttva_gnn.py` (~120 lines): TTVAGATModel — heterogeneous
  bipartite GAT with edge attributes. 2x HeteroConv(GATConv, 4 heads) + residual,
  edge prediction MLP. State dict keys preserved for existing weight compatibility
  (verified: 38/38 keys match test_GATConv_edgeattr_BCE_100.pyt).
- `src/pv_finder/data/graph_construction.py` (~470 lines): Merged h5_to_graph.py
  and pvfinder_output_to_graph.py. Shared edge attribute computation (longitudinal
  significance, horizontal significance, |dz|). Two modes: create_training_graph()
  from MC truth, create_inference_graph() from PVF peaks. Numba for compute_pv_sigma
  and norm_cdf.
- `src/pv_finder/training/training_gnn.py` (~175 lines): GNN train/validate loop.
  BCEWithLogitsLoss with dynamic pos_weight (num_neg/num_pos per batch). Imports
  gradient monitoring from existing training.py (not duplicated).
- `src/pv_finder/training/train_gnn_ttva.py` (~185 lines): CLI entry point following
  train_tracks_to_hist.py pattern. YAML config, MLflow, Adam optimizer, checkpoint
  saving (.pyt + .pth).
- `src/pv_finder/evaluation/evaluate_gnn_ttva.py` (~490 lines): Clean/Merged/Split/Fake
  vertex classification. Two evaluation methods: MaxScore (top-1 per track) and
  Threshold (all edges above cutoff). Results saved as .npy.
- `configs/vertex_association/config_gnn_ttva.yml` (~32 lines): Training config.

Key decisions:
- Stayed as close as possible to original working code. Changes limited to: type hints,
  import paths, deprecated API fixes, constant extraction.
- Fixed wrong defaults in model (track_input_size=7→8, edge_attr_dim=1→3) that were
  never actually used but were misleading.
- Fixed deprecated import: torch_geometric.loader.DataLoader (not .data.DataLoader).
- Fixed split/fake accumulation bug in source Evaluation_GNN_TTVA.py (lines 285-288
  swapped results[2]↔results[3], but results order is [clean, merged, split, fake]).
- Decoupled inference graph construction from embedded peak finding — caller provides
  pre-computed peaks.

Proofreading: All 4 code files verified against source by dedicated smart-coder agents.
Weight loading verified against existing checkpoint.

---

## 2026-02-24 — PVF vertex-finding evaluation pipeline

Added two-step PVF evaluation pipeline, ported from atlas_pvfinder/mattia_finder:

**Step 1 (Resolution):** Run PVF model inference on test events → peak finding →
collect pairwise vertex distances → fit sigmoid to extract σ_vtx-vtx.

**Step 2 (Classification):** Use σ_vtx-vtx as matching window to categorize predicted
PVs as Clean/Merged/Split/Fake/Missed against truth (from track_associations.h5,
nTracks ≥ 2).

New files (2):

- `src/pv_finder/evaluation/vertex_matching.py` (397 lines): Resolution fitting
  (sigmoid fit for σ_vtx-vtx, FWHM-based per-peak resolution), vertex categorization
  (Clean/Merged/Split/Fake), S/MT/FP matching. Fixed if/elif/else bug in original
  conjoined peak logic.
- `src/pv_finder/evaluation/evaluate_pvf.py` (462 lines): Two-step CLI with tqdm.
  Full mode (inference + resolution + classification) and classify-only mode.
  Handles old checkpoint namespace (`model` → `pv_finder.models`).

Key details:

- Test split: events 48450-50999 (2,550 events), subevent indices 581400-611999.
  Sequential, never seen during training (unlike old qibin_test_main_indices_v2.p
  which had random indices including training events).
- Peak finding: reuses `utils/peak_finding.pv_locations_updated_res` (shared code).
- Model weights: `pvf_e2e_epoch400.pyt` (epoch 400, tracks→hist UNet).
- Data: `training_data.h5` for inference, `track_associations.h5` for truth PVs.
- Constants: added N_SUBEVENTS=12, BINS_PER_SUBEVENT=1000 to `utils/constants.py`.

---

## 2026-02-24 — PVF evaluation pipeline: three bugs found and fixed

After running the initial evaluation (pvf_e2e_epoch400.pyt, 2550 events), results were
clearly wrong: 72.8% fake rate, inflated sigma, and a spurious "Missed" category with
inconsistent percentages. Three independent bugs were responsible.

**Bug 1 — Wrong truth source (72.8% fake rate)**

`evaluate_vertices` was loading truth PV positions from `pv_loc_z` in
`track_associations.h5` (raw MC generator-level vertex positions), then matching
predicted KDE peaks to those positions with a small sigma window. The model was trained
to predict KDE histograms, which spatially blur and merge nearby vertices. Predicted
peaks from the KDE systematically do not land within a small matching window of any
individual MC vertex, so almost every predicted peak was labelled fake.

Fix: use peak-finding on the saved truth KDE histograms (`pvf_truth_histograms.npy`,
same algorithm and parameters as prediction peak-finding). The model predicts KDE peaks;
truth should be defined the same way. Removed the `--track-h5` requirement entirely.

**Bug 2 — "Missed" category computed with wrong formula**

`n_missed = n_truth - n_clean_reco - n_merged_reco` conflated reco-level counts with a
truth-level quantity. Each "merged" reco PV covers multiple truth PVs, but the formula
only subtracted 1 per merged reco, so those extra truth PVs were falsely counted as
missed. The `truth_classification` array returned by `compare_res_reco` (which gives the
correct per-truth-PV classification) was discarded. Additionally, the bar chart's
percentage denominator included Missed alongside reco-side categories, so
Clean+Merged+Split+Fake percentages did not sum to 100%.

Fix: removed the "Missed" category from evaluation output. Bar chart now shows only
the four reco-side categories (Clean, Merged, Split, Fake) with percentages normalised
to total reco PVs. Note: in the original mattia_finder scripts, Missed was either not
tracked or was always zero by a latent bug (checking `tc == []` on a list that was
always non-empty after `compare_res_reco`).

**Bug 3 — Conjoined peak splitting missing from pv_locations_updated_res (inflated sigma)**

The resolution plot fits a sigmoid to the distribution of pairwise distances between all
predicted PVs. The inflection point of the sigmoid is σ_vtx-vtx. This works because
pairs of true PVs separated by less than the resolution are merged into one predicted PV
(no pair contributed to the histogram), creating a depletion ("notch") near Δz = 0.
The width of the notch determines σ.

The conjoined-peak splitting logic (from the original `efficiency_res_optimized_atlas.py`)
detects two overlapping peaks that never dip below threshold — a local minimum within an
above-threshold region where the histogram starts rising again. It splits the region into
two separate PV candidates. This logic was removed in commit a978beb ("remove prominence
splitting") based on per-vertex visualization F1 analysis. That analysis was valid for
the per-vertex diagnostics use case (1-to-1 greedy matching, F1 ~0.83–0.86) but
incorrect for the resolution use case.

Without conjoined splitting, pairs of true PVs at separations of 0.3–1 mm that produce
overlapping KDE peaks are merged into one predicted PV and their pair is never added to
the histogram. The notch extends out to the KDE overlap scale (~0.8 mm) rather than the
true vertex-vertex resolution (~0.34 mm). The fitted sigma was therefore inflated by ~2×.

Fix: restored the `peak_passed` tracking and third flush condition in
`pv_locations_updated_res` (`utils/peak_finding.py`). The return signature (4 values)
is unchanged so all callers are unaffected. The conjoined splitting is required for
correct resolution measurement; the F1 analysis should not have been applied here.

Updated: `docs/evaluation/vertex_finding.md`.

---

## 2026-02-24 — Run3 GNN track probability distribution script

Added `src/pv_finder/diagnostics/run3_track_probability.py`.

The script runs the full PVF → peak-finding → GNN pipeline on Run3 events
from the pre-extracted NPZ cache and plots the distribution of per-edge GNN
association scores (sigmoid of edge logits).

**Motivation:** Inspect how confident the GNN is about track-to-vertex
associations on real (Run3) data, without any truth labels. The score
distribution reveals whether the model produces sharp decisions (mass near 0
and 1) or diffuse uncertain scores.

**Pipeline per event:**
1. Build padded subevent tensors (100 tracks/chunk) and run the PVF model to
   obtain the 12 000-bin histogram.
2. Peak-find with `pv_locations_updated_res` to get `pred_z / pred_heights /
   pred_sigmas`.
3. Build a fully-connected bipartite inference graph with
   `create_inference_graph`.
4. Run `TTVAGATModel`; apply sigmoid to edge logits.
5. Collect all scores plus max-score-per-track.

**Outputs:** two-panel PNG/PDF (all edge scores + max-per-track, log y-axis)
and a JSON summary.

**Key design notes:**
- PVF weights (`tracks2kde_KDE_A_z_epoch180.pyt`) were pickled with the
  legacy `model.autoencoder_models` module path. The `--legacy-model-path`
  argument inserts `mattia_finder/` into `sys.path` before loading.
- GNN weights (`gnn_ttva_epoch100.pyt`) are a plain state dict; loaded into
  `TTVAGATModel` directly.
- Max-score-per-track uses `scores.reshape(n_tracks, n_pvs).max(axis=1)`,
  valid because the bipartite graph is fully connected and edges are ordered
  by the `meshgrid(indexing="ij")` in `create_inference_graph`.

Updated: `docs/diagnostics/vertex_association.md`.

---

## 2026-02-24 — Run 3 PVF evaluation script

Added `src/pv_finder/evaluation/evaluate_pvf_run3.py`.

Implements end-to-end PV-Finder evaluation on real Run 3 proton–proton collision data
loaded from a ROOT file via `uproot`, using AMVF reconstructed vertices (beam-spot-
corrected, nTracks ≥ 2) as truth.

**Key design decisions:**

- **ROOT input, on-the-fly subevent building:** Unlike the MC pipeline which reads
  pre-batched H5 subevents, this script reads raw track branches
  (`RecoTrack_z0/d0/ErrD0/ErrZ0/ErrD0Z0`) and builds the 12×100-track subevent
  tensors on the fly, matching the geometry used during training.

- **AMVF truth:** `RecoVertex_z - BeamPosZ` (nTracks ≥ 2). No truth KDE histograms
  exist for real data, so the LHCb-style efficiency metric is replaced by
  `(clean + merged) / n_amvf`.

- **Overlay resolution plot:** The pairwise Δz plot shows PVF and AMVF distributions
  side by side with separate sigmoid fits, giving σ_PVF and σ_AMVF. This lets us
  compare the model's resolution against the baseline algorithm on the same data.

- **Legacy model loading:** Checkpoints pickled from the old `model.autoencoder_models`
  namespace are handled by inserting a shim module into `sys.modules` before
  `torch.load`, matching the approach in `evaluate_pvf.py`.

Updated: `docs/evaluation/vertex_finding.md` (Run 3 section added).

---

## 2026-02-24 — Per-vertex plots: add T2KDE model overlay

Added the T2KDE (tracks-to-KDE) neural network output as a fourth curve on the
per-vertex visualization plots, alongside e2e predicted histogram, analytical KDE,
and MC truth target.

**Changes:**

- `vertex_plots.py`: New `hist_t2kde` parameter on both `plot_vertex_zoom` and
  `plot_event_overview`. Plotted as orange dash-dot line (`COL_T2KDE = "#ff7f0e"`),
  rescaled to e2e global peak like the other overlays.
- `run_per_vertex.py`: Imports T2KDE model loading/inference from
  `kde_study/kde_model_inference.py`. New `--t2kde-model-path` CLI argument
  (default: `model_weights/tracks2kde_KDE_A_z_epoch180.pyt`). Gracefully skips
  if model file not found. Passes T2KDE predictions through `_process_event` to
  both plot functions.

**Motivation:** Having the T2KDE model curve alongside the analytical KDE and e2e
prediction on the same zoom plot makes it easy to visually compare all three
representations at the vertex level — useful for diagnosing where the learned KDE
approximation agrees or disagrees with the exact analytical computation.

Updated: `docs/diagnostics/vertex_finding.md`.

---

## 2026-02-24 — Rebuilt evaluation pipeline from mattia_finder

Deleted and rewrote all three evaluation files from scratch, porting faithfully from
`atlas_pvfinder/mattia_finder/`. The previous versions produced wrong results due to
accumulated bugs from incremental refactoring.

**Files replaced:**

- `src/pv_finder/evaluation/vertex_matching.py` (755 lines) — from
  `model/efficiency_res_optimized_atlas.py`. Contains `_pv_locations_updated_res`
  (6-tuple return with conjoined flags), `filter_nans_res`, `get_reco_resolution`,
  `compare_res_reco`, `compare_res_reco2`, plus resolution fitting functions
  (`fit_sigma_vtx_vtx`, `make_resolution_plot`).

- `src/pv_finder/evaluation/evaluate_pvf.py` (387 lines) — from
  `evaluation/evaluate_model.py`. MC evaluation: loads pickled inputs/labels/outputs
  from TestModel.py, reads truth from ROOT file, classifies vertices, computes
  LHCb-style efficiency.

- `src/pv_finder/evaluation/evaluate_pvf_run3.py` (803 lines) — from
  `evaluation/run3_infer_compare_amvf.py`. Run 3 evaluation: builds subevent tensors
  from ROOT tracks, runs PVF inference on GPU, matches against AMVF truth, generates
  resolution plot (PVF + AMVF overlay) and category bar chart.

**Bugs fixed during port:**

1. `[[]]*n` shallow copy in `compare_res_reco` — created n references to the SAME
   list, causing all vertices to share classifications. Fixed to
   `[[] for _ in range(n)]`.

2. `MT_total += eff[3]` in evaluate_model.py — the efficiency tuple is
   `(S, Sp, MT, FP)`, so index 3 is FP, not MT. Fixed to `FP_total += eff[3]`.

**Peak-finding strategy:** Kept `utils/peak_finding.py` untouched (4-tuple return,
used by `diagnostics/run3_track_probability.py` and `per_vertex_visualization/`).
The 6-tuple version lives as `_pv_locations_updated_res` in `vertex_matching.py`.
Will unify after testing.

**Known issue:** `evaluate_pvf_run3.py` uses wrong feature engineering from
mattia_finder (d0/2, theta/3, (phi+pi)/3 instead of d0, d0_err, z0_err, d0_z0_cov).
Copied as-is to test matching/classification logic first; features will be fixed as
follow-up.

**Note:** Files exceed 500-line pre-commit limit (755 and 803 lines). Will be split
after validation against known-good mattia_finder results.

Updated: `docs/evaluation/vertex_finding.md`.

---

## 2026-02-25 — MC evaluation pipeline (run_eval_pvf.py)

Built `src/pv_finder/evaluation/vertex_finding/run_eval_pvf.py` from scratch to replace the broken legacy eval scripts. Supports three pipeline modes: K2H-only (analytical KDE), full T2KDE→K2H, and E2E (tracks→hist directly).

**Key findings during development:**

- **h5 ↔ ROOT index mismatch.** The h5 file uses reindexed ("pubindices") event ordering — h5 event `i` ≠ ROOT event `i`. The mapping is `configs/qibin_test_main_indices_v2.p` (copied from mattia_finder). Without this, ROOT truth matching gives ~16% efficiency instead of ~90%.

- **Pairwise Δz symmetry.** `pv_locations_updated_res` returns PVs sorted ascending, so all raw pairwise differences are negative. Fixed by adding both `+dz` and `-dz`, making the distribution symmetric for the sigmoid fit.

- **Two integral thresholds.** Use `0.5` for σ_vtx_vtx computation (matches `calculate_sigma.py`) and `0.2` for performance metrics (matches `evaluate_model.py`).

- **Pileup filter.** Summary statistics match mattia_finder only when restricted to events with 55 ≤ ActualNumOfInt ≤ 65 (loaded from ROOT). Overall efficiency across all events is also reported.

- **nTracks≥2 filter.** Without `--root`, truth from h5 has no track filter, inflating `reco_merged`. With `--root --qibin`, truth is filtered to nTracks≥2, matching mattia_finder exactly.

- **E2E checkpoint format.** mattia_finder saves full model objects (`.pyt`), not state dicts. Extracted state dict using the `pvfinder` conda env and saved as `model_weights/tracks2hist_1channel_200epochs_epoch_191_fullstate.pth`.

**Test set:** `configs/test_main_indices_2550evt.p` — h5 events 48450–50999 (last 5%, matches mattia_finder split [0.7, 0.25, 0.05]).

Updated: `docs/evaluation/vertex_finding.md`.

---

## 2026-02-27 — Add stats_histogram.png to run_eval_pvf.py

Added `plot_stats_histogram()` to `run_eval_pvf.py`. Produces `stats_histogram.png` alongside the existing `resolution_plot.png` and `performance_plot.png`.

**What it shows:** average count/event for clean, merged, split, and fake reconstructed PVs vs pileup (all events, no μ filter), with mean ± SEM error bars. X-axis is `ActualNumOfInt` (rounded) when ROOT truth is available, else N truth PVs per event.

**Why:** matches the `make_npv_plot` / category summary style from `mattia_finder/plotting/plot_tracks2hist.py` — adjacent-mu pair binning, one line per category with error bars.

**Implementation note:** to stay within the 500-line limit, several verbose print blocks and docstrings were trimmed. The file is now at exactly 500 lines.

Updated: `docs/evaluation/vertex_finding.md`.

---

## 2026-03-01 — Eval pipeline cleanup + stats histogram

**CLI simplification (Proposal B):**
- Removed `--full-pipeline`, `--t2kde-model`, `run_t2kde` — the T2KDE→K2H path was unused and added ambiguity.
- Merged `--root` + `--qibin` into `--root-truth`; qibin path is now hardcoded as `configs/qibin_test_main_indices_v2.p`.
- Added defaults for `--h5` (standard data path), `--indices` (`configs/test_main_indices_2550evt.p`), and `--root-truth` (standard ROOT path). Only the model checkpoint is now required.
- Minimal run: `python run_eval_pvf.py --e2e-model model_weights/foo.pth`

**Plotting extracted to `plots_pvf.py`** (184 lines); `run_eval_pvf.py` is 467 lines.

**New output: `stats_histogram.png`** — average clean/merged/split/fake count/event vs pileup (all events, paired μ bins, mean ± SEM), matching mattia_finder `make_npv_plot` style.

**Pileup x-axis**: performance and stats plots now use `ActualNumOfInt` (rounded) when ROOT is available, matching mattia_finder convention. Falls back to N truth PVs without ROOT.

Updated: `docs/evaluation/vertex_finding.md`.

---

## 2026-03-02 — Run 3 evaluation pipeline

Added evaluation pipeline for Run 3 real data (`data/run3/file_3.root`).

**New files:**
- `src/pv_finder/data/run3_io.py` (265 lines) — data loading from ROOT (uproot) and NPZ cache, returning `Run3Event` NamedTuple with tracks + AMVF vertices.
- `src/pv_finder/evaluation/vertex_finding/run_eval_pvf_run3.py` (470 lines) — evaluation script using AMVF vertices (nTracks >= 2) as reference "truth". Supports full pipeline (T2KDE + K2H) and E2E inference modes.

**Modified:**
- `plots_pvf.py` — added optional `title` parameter to all three plot functions for Run 3 labeling (backward compatible).

**Key finding — beam correction required:** AMVF vertex z positions are beam-corrected while PVFinder output is in the detector frame. Without subtracting BeamPosZ from AMVF z, efficiency drops from ~88% to ~26% due to systematic offset. Beam correction is now the default (`--no-correct-beam` to disable).

**Smoke test results (T2KDE+K2H, NPZ, 30 events):** Eff=87.6%, FP=3.93/evt, σ_vtx_vtx=0.30mm — comparable to MC evaluation (86.3%, 1.47/evt, 0.34mm).

Updated: `docs/evaluation/vertex_finding.md`.

---

## 2026-03-02 — Random event selection for Run 3 per-vertex visualization

**Problem:** `run_per_vertex.py` selected random MC events (via `--start-event`) but always took the first N Run 3 events from the NPZ file. Also, Run 3 event folders were named by loop counter (`event0000`) instead of actual NPZ index, so reruns would overwrite existing plots.

**Changes:**
- `feature_loading.py` — added `shuffle_seed` parameter to `load_run3_data()`. When set, shuffles iteration order so `max_events` picks random events.
- `run_per_vertex.py` — added `--seed-run3` CLI arg (random by default, like MC's `--start-event`). Fixed Run 3 event naming to use `evt["event_idx"]` (actual NPZ index) instead of loop counter.

**Generated:** 3 new random events for each dataset (MC events 10179–10181, Run 3 events 358/571/1012) in `outputs/per_vertex/`.
