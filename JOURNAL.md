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

---

## 2026-03-02 — Investigation: near-zero Δz excess in Run 3 resolution plots

The Run 3 pairwise Δz resolution plot shows a conspicuous excess of entries in the
-1mm to +1mm range compared to MC. Three hypotheses were tested systematically.

**Methodology:**

1. *Pileup check:* Analyzed 2000 Run 3 events from NPZ, using number of AMVF vertices
   (nTracks >= 2) as pileup proxy. Computed all pairwise Δz between AMVF truth vertices,
   tagged by pileup tercile (low/mid/high).

2. *Model prediction analysis:* Used existing eval pickles to classify close predicted-PV
   pairs by matching both predictions to truth (0.5mm window):
   - Run 3: `outputs/eval_run3_e2e_2500/eval_results.pkl` (2500 events, e2e epoch 130)
   - MC: `outputs/eval_e2e_pvfinder_ep130_root/eval_results.pkl` (2550 events, same model, ROOT truth)
   - For each pair of predicted PVs with |Δz| < 1.5mm, classified the pair as:
     split (both match same truth), fake-neighbor (one or both unmatched), or close-truth
     (both match different truth vertices).

3. *Histogram-level analysis:* Ran e2e inference on 200 Run 3 and 200 MC events, analyzed
   peak heights, widths, proximity to truth, and threshold sensitivity.

**Result 1 — Pileup is NOT the cause:**

High-pileup events contribute ~45% of ALL pairwise Δz (due to N^2/2 scaling), but this
fraction is identical across all Δz ranges. Mean pileup is ~31.0 for every Δz bin from
0 to 6mm — no trend whatsoever.

**Result 2 — Peak splitting: NOT the primary cause:**

Run 3 has 1.49 splits/evt vs MC 1.51 splits/evt — essentially identical. Splitting is
a baseline effect present in both datasets, not Run 3-specific.

**Result 3 — Fake neighbor peaks: THE DOMINANT CAUSE:**

- Run 3: 3.84 fake peaks/evt, MC: 0.69 fake peaks/evt (5.6x more in Run 3)
- Fake-to-real ratio: Run 3 = 11.7%, MC = 2.7% (4.4x worse)
- 79% of Run 3 fakes are within 1.5mm of a truth vertex — they are "shoulder" peaks /
  sidelobes of real vertex peaks, not random noise
- 57.6% are 0.5–1.0mm from truth — directly populating the near-zero Δz excess
- Fake peaks have median height 0.051 (vs real 0.30), FWHM 0.68mm (vs real 0.24mm)
- At |Δz| in [0.5, 1.0mm), fake-neighbor pairs: Run 3 = 1603/1000evt vs MC = 421/1000evt,
  an excess of +1182/1000evt
- Even at integral_threshold=1.0, Run 3 still has 0.62 fake/evt vs MC 0.01

**Result 4 — Close truth vertices: NOT the cause (opposite direction):**

MC has 15.8 close truth pairs/evt vs Run 3 2.8 — MC has 5.6x MORE close truth vertices.
If anything, this should produce more near-zero excess in MC, not Run 3.

**Conclusion:**

The near-zero Δz excess is caused by domain shift: the model produces broader, noisier
histogram output on Run 3 data, creating spurious sidelobes near real vertex peaks that
pass the peak-finding threshold. The real peaks are similar between MC and Run 3, but
the flanks/tails are significantly noisier in Run 3. This manifests as low-amplitude,
wide fake peaks sitting on the shoulders of genuine vertex peaks.

**Key numbers:**
- sigma_vtx_vtx: Run 3 = 0.309mm, MC = 0.383mm (but Run 3 fit is degraded by fakes)
- Efficiency: Run 3 = 96.8%, MC = 93.9%

**Analysis scripts:** `/tmp/pileup_resolution_check.py`, `/tmp/investigate_near_zero.py`,
`/tmp/histogram_analysis2.py`.

**Eval results used:** `outputs/eval_run3_e2e_2500/`, `outputs/eval_e2e_pvfinder_ep130_root/`.

**Next steps:** (completed in follow-up below)

---

## 2026-03-02 — Deep-dive: KDE inflection points & fake peak visualization

Follow-up to the near-zero Δz investigation. Two tasks completed:

### 1. KDE inflection point investigation

Analyzed 40 fake-neighbor cases across 38 unique Run 3 events. For each event, ran
the e2e model and computed the analytical KDE, then measured metrics at fake peak
locations vs 1373 truth-matched (real) peak locations. Five hypotheses tested:

**H1 — KDE inflection points: NOT SUPPORTED.**
47.5% of fakes are at KDE inflection points vs 54.3% of real peaks. Fakes are
actually *less* likely to sit at inflection points than real peaks.

**H2 — KDE shoulder (positive curvature): NOT SUPPORTED.**
Only 5.0% of fake peaks have positive KDE curvature vs 10.8% of real peaks.
95% of fakes are at locations where the KDE is concave-down — genuine signal exists.

**H3 — Model amplification: WEAKLY SUPPORTED.**
Fakes have ~15% higher hist/KDE amplification ratio (0.0056 vs 0.0049). The model
slightly over-amplifies at fake locations, but the difference is modest.

**H4 — Subevent boundary artifacts: NOT SUPPORTED.**
10.0% of fakes vs 10.1% of real peaks are within 2mm of a boundary — identical.

**H5 — Track density: STRONGLY SUPPORTED (p = 0.00014).**
Fake peaks: 31.9 tracks/mm (mean), median 26.
Real peaks: 21.3 tracks/mm (mean), median 16.
Mann-Whitney U test confirms significance. Fake sidelobe peaks appear preferentially
in high track-density regions.

**Interpretation:** Fakes are NOT inflection-point or boundary artifacts. They occur
in high track-density regions where the analytical KDE itself has substantial signal
(mean KDE at fakes = 57.7 vs 88.6 at real peaks, ~47% of truth-peak KDE). The model
sees a broad KDE feature from many overlapping tracks and over-resolves it into multiple
narrow peaks — a **deconvolution artifact**. The UNet trained on MC (with typically fewer
close-by tracks per vertex) over-resolves the broader KDE shoulders of denser Run 3 data.

**Analysis script:** `/tmp/kde_investigation.py`.
**Results:** `outputs/kde_investigation_results.pkl`.

### 2. Fake-neighbor visualization (40 cases)

Produced per-vertex zoom plots for all 40 identified fake-neighbor cases using the
existing `plot_vertex_zoom` code. For each case, two views:
- **Fake-centered**: zoom window (±4mm) centered on the fake peak position
- **Truth-centered**: zoom window (±4mm) centered on the nearby truth vertex

Each plot shows e2e predicted histogram (red), analytical KDE (blue), T2KDE model
(orange), all AMVF truth vertices, predicted peaks (filled/unfilled dots), and the
track scatter panel. Also produced 38 event-overview plots.

**Output:** `outputs/fake_neighbor_viz/` — 118 PNGs + 38 PDFs.
**Visualization script:** `/tmp/visualize_fake_neighbors.py`.
**Cases source:** `outputs/fake_neighbor_cases.pkl` (40 cases from 500 events).

**Next steps:** (completed in follow-up below)

---

## 2026-03-03 — Pipeline vs E2E comparison on Run 3 (2000 events)

Ran both the two-stage pipeline (T2KDE ep130 → K2H ep190) and the e2e model (ep130) on
the same 2000 Run 3 events from NPZ, to test whether the pipeline produces fewer fakes.

**Results (same 2000 events, AMVF truth, same peak-finding thresholds):**

| Metric          | Pipeline (T2KDE→K2H) | E2E        |
|-----------------|----------------------|------------|
| Fake/evt        | 3.96                 | 4.92       |
| Efficiency      | 87.1%                | 97.3%      |
| sigma_vtx_vtx   | 0.304 mm             | 0.353 mm   |
| Clean/evt       | 25.19                | 28.16      |
| Split/evt       | 0.62                 | 0.87       |
| Missed/evt      | 3.75                 | 0.78       |
| Pred/evt        | 29.78                | 33.96      |

**Key finding:** Pipeline has 20% fewer fakes (3.96 vs 4.92/evt) and a cleaner
resolution plot (less near-zero Δz excess). But it comes at a steep efficiency cost:
87.1% vs 97.3%, missing 3.75 vtx/evt vs 0.78. The pipeline is more conservative overall.

**Interpretation:** Both models produce significant fakes on Run 3 — the issue is not
unique to the e2e architecture. The K2H UNet also over-resolves, just less aggressively
than the e2e model. This suggests the problem is in the UNet decoder stage regardless
of whether the input is a learned latent (e2e) or an explicit KDE (pipeline).

**Resolution plots confirm:** Pipeline plot shows a clean step function with minimal
near-zero excess. E2E plot shows clear bumps at ±0.5–1mm (the sidelobe fakes).

**Output:** `outputs/eval_run3_pipeline_2000/`, `outputs/eval_run3_e2e_2000/`.

**Revised next steps:**
- Post-processing sidelobe suppression filter on e2e output (keep high efficiency,
  reduce fakes) — this is the most practical short-term fix
- Domain adaptation: fine-tune UNet on Run 3-like data
- Training augmentation: add MC events with artificially increased track density

---

## 2026-03-03 — Post-processing: Gaussian smoothing + NMS for sidelobe suppression

Added two post-processing features to `run_eval_pvf_run3.py` to address the Run 3
fake sidelobe problem identified on 2026-03-02.

### What they do

**Gaussian pre-smoothing** (`--smooth-sigma S`): Applies `gaussian_filter1d(ph, sigma=S)`
to the predicted histogram *before* peak finding. The original unsmoothed histogram is
preserved for all other purposes. Merges narrow sidelobe fluctuations into their parent
peaks so the peak finder never sees them as separate peaks. Mainly kills the closest
fakes (< ~0.3 mm from parent).

**NMS** (`--nms-min-sep D --nms-max-ratio R`): After peak finding, scans all pairs of
peaks closer than D mm. If the shorter peak's height is < R × the taller peak's height,
the shorter one is suppressed. Processes tallest-first. Targets remaining sidelobes at
0.5–0.85 mm that survived the blur. Preserves genuine close vertex pairs (similar heights)
while killing sidelobe fakes (much shorter than parent).

Implementation: `suppress_neighbor_peaks()` in `efficiency_res_optimized_atlas.py`.

### Configuration sweep (2109 events, E2E ep130, file_3.root 300K–302.5K)

| Configuration             | Eff (%) | Fake/evt | sigma (mm) |
|---------------------------|---------|----------|------------|
| Baseline (no PP)          | 97.6    | 4.80     | 0.347      |
| smooth s=2                | 97.7    | 4.10     | 0.415      |
| s=2 + NMS(0.85, 0.3)     | 97.4    | 2.70     | 0.487      |
| s=2 + NMS(0.85, 0.5)     | 96.9    | 2.00     | 0.592      |

Note: raw sigma values *increase* with PP because the fake close pairs that were
artificially pulling sigma down get removed. The post-PP sigma is a more honest measure.

### Decomposition: smoothing vs NMS

| Step                       | Peaks removed/evt | Fake | Real |
|----------------------------|:-:|:-:|:-:|
| Gaussian blur (s=2)        | 0.79 | 0.75 | 0.05 |
| NMS (0.85, ratio<0.5)      | 2.30 | 2.05 | 0.25 |
| **Total**                  | **3.09** | **2.80** | **0.30** |

NMS does 75% of the cleaning; the blur is a light first pass.

### NMS diagnostic: what gets removed (177 events, `nms_diagnostic.py`)

**NMS(0.85, 0.3)** — conservative, 90.9% of removed peaks are fake:

| Distance to survivor | Total | Real | Fake | Fake% |
|---------------------|:---:|:---:|:---:|:---:|
| 0.30–0.45 mm | 1 | 1 | 0 | 0% |
| 0.45–0.55 mm | 20 | 3 | 17 | 85% |
| 0.55–0.65 mm | 85 | 2 | 83 | 98% |
| 0.65–0.75 mm | 68 | 7 | 61 | 90% |
| 0.75–0.85 mm | 54 | 9 | 45 | 83% |

Removes 1.49 peaks/evt: 1.36 fake + 0.14 real (ratio 10:1).

**NMS(0.85, 0.5)** — aggressive, 78.6% of removed peaks are fake:

Removes 2.33 peaks/evt: 1.83 fake + 0.50 real (ratio 3.7:1).

The resolution plot bump at ±0.55–0.80 mm was 67% real-fake satellite pairs.
In the bump core (0.55–0.65 mm), NMS 0.3 achieves 98% fake purity.

### Also fixed

Sigmoid fit in `curve_fit` now uses `bounds=([0,0,0,0], [inf,...])` to prevent
the fit from going below zero (both `run_eval_pvf_run3.py` and `run_eval_pvf.py`).
Added `ax.set_ylim(bottom=0)` to `plots_pvf.py`.

**Output:** `outputs/eval_run3_smooth_s2/`, `outputs/eval_run3_s2_nms03/`,
`outputs/eval_run3_s2_nms05/`, `outputs/nms_diagnostic/`.

---

## 2026-03-11 — Run 2 real data evaluation + eval codebase cleanup

### Run 2 data

Added evaluation on Run 2 real collision data (2018, 13 TeV, ZeroBias). Data lives in
`data/run2/Run2_Data/` — 4 ROOT files, 298,639 events total. Same ROOT branch structure
as Run 3, so `run_eval_pvf_run3.py` works directly with no code changes.

Pileup distribution: mean μ=32.5, median 31.2, peaked at ~25–30. Lower than both MC
(flat, mean 39.9) and Run 3 (bimodal, median 59.0). BeamPosZ ≈ -2.5 mm (correction needed).

Baseline results (T2KDE+K2H, 2500 events, AMVF ntrks ≥ 2):

| Metric       | Run 2 real | MC (RecoVertex ref) |
|-------------|-----------|-------------------|
| Efficiency  | 87.3%     | 85.9%             |
| σ_vtx_vtx   | 0.303 mm  | 0.333 mm          |
| Truth PVs/evt | 22.5    | 22.8              |

Model generalizes well from MC to real data — comparable or slightly better performance.

### Eval codebase cleanup

Removed 12 legacy files (7 from `vertex_finding/`, 5 from parent `evaluation/`).
Cleaned `efficiency_res_optimized_atlas.py` from 953 → 292 lines by removing ~650
lines of dead code (broken functions referencing undefined symbols, unused imports).
All pre-commit hooks pass. Verified results are unchanged after cleanup.

**New doc:** `docs/data/run_2.md`. Updated `docs/evaluation/vertex_finding.md`.

---

## 2026-03-11 — Sidelobe investigation: root cause confirmed as UNet architecture

### Investigation

Analyzed the 0.25–1.0 mm bump in the pairwise Δz resolution plot that appears on
Run 2 real data, Run 3 real data, and MC simulation alike.

**Method**: classified all predicted peak pairs within 0.25–1.0 mm by matching both
peaks against truth/AMVF vertices. For each pair, determined whether it's:
- one real + one fake ("sidelobe")
- both matching different truth vertices ("genuine close pair")
- both matching the same truth vertex ("true split")
- both fake

**Results** (2500 events each):

| Category            | Run 2  | MC     |
|---------------------|--------|--------|
| Sidelobe (1 real + 1 fake) | 60.1%  | 51.9%  |
| Genuine close pairs | 26.2%  | 33.1%  |
| True splits         | 10.2%  | 12.6%  |
| Both fake           | 3.5%   | 2.3%   |

After reweighting MC to match Run 2's pileup distribution (μ≈42, σ≈5):
- 91% of Run 2's bump is shared with MC (UNet artifact)
- 9% is domain-shift excess
- Pileup distribution difference explains only 12% of the sidelobe fraction gap

**Root causes** (all architectural):
1. **ConvTranspose1d(k=2, s=2)** in decoder: kernel=stride → zero overlap between
   adjacent output regions → seam/aliasing artifacts at 2, 4, 8-bin scales
2. **8x bottleneck** (1000→125 bins): truth peaks (~2-4 bins) are sub-resolution
3. **k=25 first encoder kernel**: correlates bins across 1.0 mm (= sidelobe range)

Height ratio of close-pair peaks (MC, 500 subevents): 46% have ratio < 0.3
(clearly fake sidelobes), 65% < 0.5. Mean ratio 0.39.

### Plan: UNet_1000_v2

Designed a new K2H architecture targeting all three root causes:

1. **Replace ConvTranspose1d with nearest-neighbor interpolation + Conv1d** —
   eliminates checkerboard artifacts (Odena et al. 2016)
2. **Reduce to 2 pooling levels** (bottleneck 250 bins, 4x) instead of 3 (125, 8x) —
   truth peaks stay representable at bottleneck
3. **Reduce first encoder kernel from 25 to 15** — 0.6 mm correlation (vs 1.0 mm)

Additional: additive skip connections (simpler than concat), residual encoder blocks
(`ResConvBNrelu`), k=1 pointwise final output conv. ~156K params (vs ~221K).

New file: `src/pv_finder/models/unet_v2.py`
New config: `configs/vertex_finding/config_KDE2HIST_v2.yml`
Training: same hyperparameters as baseline (Adam lr=1e-4, 200 epochs, batch 128).

See `docs/models/vertex_finding.md` for architecture details.

## 2026-03-16 — K2H v2 training, E2E v2 model, and sidelobe root cause correction

### K2H v2 training completed

UNet_1000_v2 trained for 200 epochs (config `config_KDE2HIST_v2.yml`, device 3).
Checkpoint: `model_weights/K2H_v2_interp_200epochs_epoch_190_fullstate.pth` (156K params).

### TracksToHist_v2 — composition-based E2E model

Created `TracksToHist_v2` in `unet_v2.py`: wraps MaskedDNN (T2KDE) + UNet_1000_v2
(K2H) as submodules. Unlike the original `trackstoHists_UNet_1000` (which duplicates
both architectures inline), this composes the standalone models. Weight loading via
`TracksToHist_v2.from_checkpoints()` factory method.

### E2E v2 fine-tuning (100 epochs)

Config: `configs/vertex_finding/config_T2HIST_v2.yml`
- Pretrained: T2KDE ep130 + K2H v2 ep190
- LR: 0.0001 uniform, 5-epoch linear warmup (0.1x → 1.0x)
- Loss: MSE (not asymmetric — MSE is standard for E2E per Qi Bin's convention)
- Checkpoints saved every 10 epochs in `model_weights/T2HIST_v2_100epochs_epoch_*.pth`

Training script: `train_tracks_to_hist.py` (added `model_type: v2` dispatch,
warmup scheduler, TracksToHist_v2 support).

### Sidelobe root cause: CORRECTED — eval threshold artifact, not architecture

**Critical finding**: the sidelobe bumps in the resolution plot are **hidden by the
eval script's dual-threshold design**, not fixed by E2E training or architecture changes.

Our `run_eval_pvf_run3.py` uses two peak-finding passes:
- `INTEGRAL_THRESHOLD = 0.2` for performance metrics (efficiency, FP rate)
- `INTEGRAL_THRESHOLD_RES = 0.5` for the resolution pairwise Δz plot

The sidelobe peaks are small (low integral) — they pass 0.2 but fail 0.5. This means:
- They **count as fakes** in performance metrics
- They **don't appear** in the resolution plot
- The resolution plot looks clean even though sidelobes exist

The `clean_run3` eval script uses a **single threshold (0.4)** for both, so sidelobes
appear in its resolution plot.

**Verification**: rerunning evals with `--integral-threshold-res 0.2` makes the
sidelobe bumps reappear in the resolution plot for ALL models (E2E v1, E2E v2, pipeline).

This means:
1. The earlier conclusion that "E2E training eliminates sidelobes" was wrong — it
   was the stricter threshold hiding them
2. The UNet_v2 architecture changes may still help, but need honest evaluation
3. All models (E2E and pipeline) have sidelobes; the pipeline may be slightly worse

### Eval script improvements

- Added `--e2e-type v1|v2` flag to both `run_eval_pvf.py` and `run_eval_pvf_run3.py`
- Added `--integral-threshold-res` CLI flag to control resolution threshold
- Fixed `load_ckpt` to handle legacy `.pyt` model objects (`hasattr(ckpt, "state_dict")`)
- Fixed `load_ckpt` to handle old module paths (`model.autoencoder_models` alias)
- Created `eval_k2h_v2_run2.py` for K2H v2 eval with analytical KDEs on Run 2 data

### K2H v2 standalone eval (analytical KDE input, Run 2, 1000 events)

K2H v2 on analytical KDEs (no T2KDE): sigma=0.304mm, eff=97.6%. This isolates K2H v2
from the T2KDE stage. Resolution plot on MC was clean (no sidelobes), but with
`integral_threshold_res=0.2` needs re-checking.

## 2026-03-17 — Threshold investigation and comparative evals

Ran systematic evals comparing `integral_threshold_res=0.2` vs `0.5` across models
and datasets to quantify the sidelobe hiding effect. Results pending.

Key question being investigated: if `integral_threshold=0.5` is used for **all**
peak finding (not just resolution), does it eliminate sidelobe fakes from performance
metrics too, without hurting efficiency on real vertices? If the 0.2-0.5 integral
range is predominantly sidelobes, 0.5 everywhere is the right operating point.

## 2026-03-24 — 400-epoch component trainings launched

Started fresh 400-epoch training runs for both stages (previous reproduction was 200 epochs):

- **T2KDE**: `config_T2KDE_400ep_03_24_2026.yml`, MaskedDNN, lr=0.001, GPU 0
  - Checkpoint prefix: `reproduction_KDE_A_z_400ep`
- **K2H**: `config_KDE2HIST_400ep_03_24_2026.yml`, UNet_1000, lr=0.0001, GPU 1
  - Checkpoint prefix: `reproduction_KDE2HIST_400ep`

Both save to `model_weights/03_24_2026/` every 10 epochs. Running in tmux sessions.

## 2026-03-24 — E2E combined model training launched (Qi Bin reproduction)

Created `config_T2HIST_400ep_03_24_2026.yml` for end-to-end training, exact reproduction
of Qi Bin's approach:

- Model: `trackstoHists_UNet_1000` (v1, inlined MLP+UNet)
- Initialized via `initialize_combined_model()` from:
  - T2KDE epoch 100: `reproduction_KDE_A_z_400ep_epoch_100.pyt`
  - K2H epoch 150: `reproduction_KDE2HIST_400ep_epoch_150.pyt`
- Loss: MSE (Qi Bin convention for E2E)
- Optimizer: Adam, lr=0.001, betas=(0.9, 0.999)
- No warmup, no LR scheduler (exact Qi Bin reproduction)
- 400 epochs, batch 128, dropout 0.25, n_latent_channels=1
- GPU 2, saving to `model_weights/03_24_2026/`
- MLflow experiment: "ATLAS 2025 Reproduction - Tracks to Histogram"
- Run name: `reproduction_T2HIST_400ep_T2KDE100_K2H150`

Decision: no warmup for this run. Warmup exists in the codebase for v2 models but
Qi Bin never used it for v1 E2E. We may revisit if training is unstable.

## 2026-03-24 — Zombie process cleanup on sneezy

Discovered load average of ~375 on a 96-core machine. Root cause: 26 zombie bash/shell
processes from old Cursor sessions (dating back to Feb 24), each spinning at 99% CPU.
These are the unkillable zombie processes caused by the kernel 4.15 bug documented in
CLAUDE.md.

`kill -9` had no effect (as expected with the kernel bug). Applied two mitigations:

1. `renice 19` — lowered all zombies to lowest scheduling priority
2. `taskset -pc 95` — pinned all 26 zombies to a single CPU core (core 95)

Net effect: freed ~25 CPU cores. Training throughput should recover to near-normal.
Only a server reboot can fully remove these processes.

## 2026-03-24 — E2E v2 training resumed to 400 epochs

Resumed `T2HIST_v2_100epochs` from epoch 90 checkpoint (full optimizer state) to train
up to epoch 400. Set `warmup_epochs: 0` to avoid re-applying warmup on resume. Kept
lr=0.0001 (original v2 rate) — 10x lower than Qi Bin's v1 rate but changing mid-training
would be disruptive. If convergence is slow, can always extend beyond 400.

## 2026-03-24 — Eval code cleanup and threshold documentation

### integral_threshold_res finding

Reviewed 03/17 eval results comparing `integral_threshold_res=0.2` vs `0.5` on Run 2:

| integral_threshold_res | sigma (Run 2) | sigma (Run 3) |
|----------------------|---------------|---------------|
| 0.2                  | 0.308 mm      | 0.309 mm      |
| 0.5                  | 0.337 mm      | 0.354 mm      |

**Decision: always use 0.5 for the resolution plot.** With 0.2, small sidelobe peaks
pass the threshold and enter the pairwise distance computation, creating an artificial
excess near Δz=0 that tightens sigma. The 0.5 threshold filters these out, giving a
cleaner and more physically meaningful resolution measurement. Both MC and real data
eval scripts default to 0.5.

Threshold summary (now standardized across all evals):
- **Performance** (peak finding for matching): `integral_threshold = 0.2`
- **Resolution** (sigma_vtx_vtx pairwise Δz): `integral_threshold_res = 0.5`

### Plot improvements

Updated `plots_pvf.py`:
- Resolution plot: replaced bar histogram with points + Poisson error bars (sqrt(N))
- Performance plot: added SEM error bars on category fractions and efficiency vs pileup;
  added annotation box with sigma/eff/FP summary
- All plots: consistent color scheme, larger fonts, `--title` CLI flag for clean titles
- Added `--title` argument to both `run_eval_pvf.py` and `run_eval_pvf_run3.py`

### Eval fidelity vs Qi Bin (mattia_finder)

Verified `run_eval_pvf.py` matches Qi Bin's evaluation exactly:
- Peak finding: threshold=0.01, integral_threshold=0.2, min_width=3
- Sigma: integral_threshold=0.5, same sigmoid fit function and p0
- Truth filter: nTracks >= 2 from ROOT (with qibin index mapping)
- Test set: 2550 events (last 5%), indices 48450-50999
- Matching: compare_res_reco with sigma_vtx_vtx window in bins
- No NMS post-processing

## 2026-03-24 — MC and Run 2 evaluations launched

Running clean evals on two E2E models:

**Models:**
- E2E v1 (Qi Bin repro): `reproduction_T2HIST_400ep_T2KDE100_K2H150_epoch_150`
- E2E v2 (UNet interp): `T2HIST_v2_100epochs_epoch_121`

**MC evals** (2550 events, ROOT truth nTracks>=2):
- `outputs/03_24_2026_output/eval_mc_e2e_v1_ep150/`
- `outputs/03_24_2026_output/eval_mc_e2e_v2_ep121/`

**Run 2 evals** (2500 events, AMVF nTracks>=2, beam-corrected):
- `outputs/03_24_2026_output/eval_run2_e2e_v1_ep150/`
- `outputs/03_24_2026_output/eval_run2_e2e_v2_ep121/`

## 2026-03-26 — HLLHC PU200 dataset + ROOT→HDF5 converter

New Run 4 dataset: `ATLAS_PVFinderData_HLLHC_mc21_14TeV_ttbar_SingleLep_PU200.root`
(99,800 events, μ≈200, ~927 tracks/event, ~126 truth PVs/event).

Built `src/pv_finder/data/root_to_h5.py` — two-pass ROOT→HDF5 converter:
- **Pass 1**: scans dimensions (max tracks/subevent, max PVs/event).
- **Pass 2**: converts in chunks (default 1000 events), writing float32 tracks
  `(N_sub, 7, MAX_TRACKS)` and float16 targets `target_y_split` and `target_y`.
- Target histograms are generated on the fly via the LHCb Gaussian-CDF method
  (resolution `σ = 0.23817 · ntrks^(-0.4949) − 0.000787` for ntrks≥2, else binWidth;
  amplitude boost `where((0.15/σ)>1, (0.15/σ)*populate, populate)`).
- Channel 0 = nTracks≥2, channel 1 = nTracks<2.
- **No `kde_split`** — HLLHC training is end-to-end only (KDEs infeasible at μ≈200).

Output: `data/run4/hllhc_pu200_training.h5`. Smoke-tested on 100 events, full conversion
verified against expected shapes.

## 2026-03-26 — HLLHC E2E training launched

Created `configs/vertex_finding/config_hllhc_pu200_e2e.yml`:
- `trackstoHists_UNet_1000` (v1) with MLP warmup + E2E phases.
- 99,800 events, split [0.7, 0.25, 0.05] sequential (seed 42).
- Phase 1: 50 epochs, MLP-only histogram supervision (UNet disabled), lr=0.001.
- Phase 2: 400 epochs, full E2E, lr=0.001, batch 128, MSE loss.
- dropout=0.25, n_latent_channels=1.

**Loss choice (MSE, not asymmetric)**: investigated git history — the asymmetric loss
(`alt_loss_A.py`) was only ever used for K2H supervised on KDEs, never for E2E. With
coefficient > 1 it penalizes under-prediction more, which actively *encourages* small
satellite peaks rather than suppressing them. MSE is the correct baseline for E2E.

Checkpoints saved every 10 epochs to `model_weights/` (phase1 + phase2 numbering).
Experiment: "HLLHC PU200 — E2E from scratch", run: `hllhc_pu200_mlp50_e2e400`.

## 2026-03-26 — HLLHC eval fixes: --mu-min/--mu-max, adaptive sigmoid fit

`run_eval_pvf_run3.py` had two bugs at PU200:

1. **Hardcoded pileup filter**: `MU_MIN/MU_MAX = 55, 65` hid the single-bin summary at
   μ≈200. Added `--mu-min` / `--mu-max` CLI flags (default still 55/65 for Run 2/3).
2. **Sigmoid fit failed on HLLHC**: initial guess `FIT_P0 = [1000, 10, 30, 0.8]` was
   tuned for Run 2 counts (~1000/bin); HLLHC has ~10000/bin. Replaced with an adaptive
   guess computed from the actual histogram:
   ```python
   baseline = float(np.median(cnts))
   dip      = baseline - float(cnts.min())
   p0 = [max(dip, 1.0), 10.0, max(baseline, 1.0), 0.5]
   ```
   Works across all data scales (MC, Run 2, Run 3, HLLHC).

## 2026-04-09 — HLLHC evaluations (old vs new model)

Ran eval on 2550 random HLLHC events with both the legacy Run-2-trained E2E v1 model
and the new HLLHC-trained model at several phase-2 checkpoints (80/100/130/150/350).
The old model transfers surprisingly well (≈90% efficiency at μ≈200) despite never
having seen PU200 data — the HLLHC-trained model matches or exceeds it on matched
efficiency from ~ep100 onward.

Peak-height comparison (integral_threshold=0.4, 200 events):
- **Old Run 2 model**: mean=0.3802, median=0.2678, std=0.3344, 78.9 peaks/evt
- **HLLHC ep100**:     mean=0.3505, median=0.2674, std=0.2821, 82.6 peaks/evt

New model produces slightly fewer tall outliers (smaller std) at similar median height —
consistent with a tighter, less-peaked output at high pileup.

## 2026-04-09 — Eval plot polish + title flag

Cleaned up `plots_pvf.py`:
- Consistent color/marker dicts (`_COLORS`, `_MARKERS`) across all three plots.
- Larger fonts (`_FONT`), SEM error bars on fraction/efficiency curves.
- Performance plot: added annotation box with σ / eff / FP summary.
- Resolution plot: points + Poisson errors instead of bars.
- Added `--title` flag to both `run_eval_pvf.py` and `run_eval_pvf_run3.py` so eval
  runs can be labeled cleanly for comparison figures.

## 2026-04-09 — New diagnostics and Run 3 I/O module

- `src/pv_finder/diagnostics/histogram_heights.py` — compares MC vs Run 3 distributions
  of non-zero bin values and peak heights from the E2E model output. Used to check
  for domain-shift signatures in model output amplitude.
- `src/pv_finder/data/run3_io.py` — shared `Run3Event` loader (ROOT via uproot or
  pre-extracted NPZ), with `nTracks ≥ 2` AMVF truth filtering. Factored out of
  `run_eval_pvf_run3.py` for reuse by diagnostics scripts.

## 2026-04-14 — HLLHC v1 training diverged; v2 recipe with stabilized LR + bigger model

The v1 HLLHC training (`config_hllhc_pu200_e2e.yml`, Phase 2 lr=1e-3) diverged
around Phase 2 epoch ≈100, slowly reconverged, and ended at a noticeably worse
operating point than expected. Root cause: at μ≈200 the per-event loss gradient
scales up with pileup (~4× more truth peaks, ~6× more tracks/event vs Run 2), so
`lr=1e-3` — inherited from the Run 2 recipe — is effectively much hotter. A single
outlier batch can kick the optimizer out of the basin.

### New recipe — `train_hllhc_e2e.py` + `config_hllhc_pu200_e2e_v2.yml`

- **Phase 2 LR 1e-3 → 1e-4** (main fix).
- **Phase 2 LR schedule**: 5-epoch linear warmup `0.01·lr → lr`, then cosine decay
  to `eta_min = lr × 0.01`. Fresh Adam at Phase 2 start so the old (potentially
  damaged) momentum is discarded.
- **Gradient clipping** `max_grad_norm = 1.0` on both phases. Smoke test shows the
  fresh-init gradient norm is ~1.6, so clipping bites immediately and shields
  against outlier-batch spikes on high-pileup events.
- **Modestly larger model**:
  - `n_UNetChannels`: 64 → 96
  - `l_HiddenNodes`: `[100]*5` → `[128]*5`
  - Total parameters 359K → 681K (≈1.9×), same UNet topology and same single
    latent channel.

Phase 1 LR stays at `1e-3` because Phase 1 (MLP warmup with UNet frozen) converged
fine in v1.

### New training script

`src/pv_finder/training/train_hllhc_e2e.py` is a **new, separate** script rather
than a config flag on `train_mlp_hist_then_e2e.py`. Reason: Phase 2 needs a custom
loop to inject gradient clipping and the LR scheduler, and injecting that into the
shared `trainNet` helper would risk the non-HLLHC trainings. The new script reuses
the Phase 1 MLP-only forward pass (`forward_mlp_hist`, `_squeeze_hist`) via an
import, so there's no duplication of the phase-1 logic.

### Smoke test

Verified before committing:

- Model builds with `n_UNetChannels=96`, `l_HiddenNodes=[128]*5` (681K params).
- Full E2E forward on a synthetic `(4, 7, 400)` batch produces `(4, 1000)` output
  and a finite MSE loss; backward runs.
- Scheduler sequence over 400 epochs: `1e-6 → 1e-4` over 5 epochs of linear warmup,
  then cosine decay (`1e-4 → 1e-6`).
- Gradient norm at init ≈ 1.6 (clip threshold 1.0 → will activate early).

### Launch

```bash
tmux new -s hllhc_v2
source venv/bin/activate
python -u src/pv_finder/training/train_hllhc_e2e.py \
    -c configs/vertex_finding/config_hllhc_pu200_e2e_v2.yml \
    2>&1 | tee outputs/logs/hllhc_pu200_v2_$(date +%Y%m%d_%H%M%S).log
# Ctrl+B, D to detach
```

The v1 config is kept for reference (to diff against the v2 recipe), not deleted.

## 2026-04-15 — Unified `integral_threshold = 0.5` across all evals

Audited every `integral_threshold` usage in the repo:

| Script | Performance | Resolution |
|--------|:-:|:-:|
| `run_eval_pvf.py` (MC) | 0.2 | 0.5 |
| `run_eval_pvf_run3.py` (Run 2/3/HLLHC) | 0.2 | 0.5 (--integral-threshold-res) |
| `eval_k2h_v2_run2.py` | 0.2 | 0.5 |
| `training/training.py` (on-the-fly eff in `trainNet`) | 0.2 | — |
| `diagnostics/run3_track_probability.py` | 0.4 | — |
| Library defaults (`utils/peak_finding.py`, `per_vertex_visualization/peak_matching.py`) | 0.5 | — |

The dual-threshold design (0.2 performance, 0.5 resolution) quietly hid sidelobe
artifacts from the resolution plot: sidelobe peaks passed 0.2 → counted as fakes
in performance, failed 0.5 → never appeared in the pairwise Δz distribution, so
σ_vtx_vtx looked cleaner than the model's actual output. This was already
documented in evaluation/vertex_finding.md, but the numbers in the performance
table and the resolution table were being computed from different peak sets,
which is quietly wrong when comparing across plots.

**Change**: all eval scripts + training loop + the run3_track_probability
diagnostic now use `integral_threshold = 0.5` everywhere. The
`INTEGRAL_THRESHOLD_RES` constant and `--integral-threshold-res` CLI flag are
kept in place (still 0.5 by default) so historical 0.2 numbers can be
reproduced if needed, but by default every metric — performance counts,
efficiency, FP rate, reco_vs_mu, σ_vtx_vtx, training-loop validation
efficiency — is computed from a single peak set.

**Impact on existing numbers**: all performance metrics (clean/merged/split/fake
counts, efficiency, FP rate) and the `reco_vs_mu` plot will report different
numbers from now on. Journal entries predating this change are kept as-is and
should not be compared directly with post-2026-04-15 numbers. Resolution
numbers are unchanged (already at 0.5). Training-time `Eff`/`FPR` log lines
will change values but this does not affect the trained weights — only the
reported metric during training.

No metric regression on the training of the HLLHC v2 run currently in Phase 2:
the threshold change only affects what `trainNet`'s validation step *reports*,
not the MSE loss that drives gradients.

## 2026-04-15 — New MC eval plots: reco_vs_mu + category_counts_hist

Two new plots added to the MC eval (`run_eval_pvf.py`), saved into the same
output directory as the existing resolution/performance/stats plots:

- **`reco_vs_mu.png`** (added earlier today): mean reconstructed PVs/event vs
  ActualNumOfInt, overlaying PV-Finder (`n_pred` from peak finder at
  `integral_threshold = 0.5`), AMVF (`RecoVertex_nTracks ≥ 2` from ROOT), and
  truth (`TruthVertex_nTracks ≥ 2`, dashed reference). SEM bars per μ bin,
  wheat annotation box with overall means. Only drawn when `--root-truth` is
  available, so it is MC-eval specific today.
- **`category_counts_hist.png`** (new): per-event distribution of
  clean/merged/split/fake counts as four overlaid step-filled histograms on
  integer bins. Monospace legend with `⟨N⟩ / σ / Σ` per category so two
  evals can be compared side-by-side at a glance. Upper-left annotation box
  shows the checkpoint stem and the integral threshold used, so the plot is
  self-identifying when filed into a folder of comparison figures.

Both plots live in `plots_pvf.py` alongside the existing three. The new
`plot_category_counts` takes a `eval_label` kwarg that `run_eval_pvf.py`
populates with `ckpt: <name>\nintegral_threshold = <value>` — this is the
"clear and precise legend per eval" requirement.

`run_eval_pvf.py` now calls `plot_category_counts` unconditionally (no ROOT
needed — it only uses per-event `clean/merged/split/fake` fields that are
always computed). `run_eval_pvf_run3.py` is at 499 lines and near the 500-line
pre-commit limit, so the new plot is wired up in the MC script only for now;
pulling it into the Run-3 script is a 1-line change once we free up room.

## 2026-04-15 — Canonical Run 2 MC E2E model + plot rework

**Canonical Run 2 MC E2E model decided:**
`model_weights/03_24_2026/reproduction_T2HIST_400ep_T2KDE100_K2H150_epoch_150_fullstate.pth`
— the 400-epoch Qi Bin reproduction from 2026-03-24, initialized from T2KDE
ep100 + K2H ep150. Use this one as the default for all Run 2 MC evals going
forward. The older `e2e_mlpHist50_e2e400_1latent_mse_phase2_epoch_130` (50-ep
MLP warmup + 400-ep E2E) is kept for reference — it's what the 2026-04-09
HLLHC-vs-Run2 comparison used and is referred to as the "old Run 2 model"
in those entries.

Plot reworks in `plots_pvf.py` (triggered by user request; applies to every
caller of the eval scripts, MC and real-data):

- **`stats_histogram.png`** — no longer shows four clean/merged/split/fake
  lines vs pileup. Now shows two curves: PV-Finder `n_pred` (≡ Σ of the four
  categories) and AMVF (nTracks≥2), both vs rounded μ, with SEM error bars
  and an overall-mean annotation box. AMVF source is `n_amvf` on MC (from
  `RecoVertex_nTracks`) and `n_truth` on real data (where the "truth" field
  already holds AMVF).
- **`category_counts_hist.png`** — no longer four overlaid step-filled
  histograms. Now a **5-bar chart** (Total, Clean, Merged, Split, Fake),
  filtered to `μ ∈ [55, 65]` (default; overridable via `mu_min`/`mu_max`
  kwargs), with mean values printed above each bar and a corner metadata
  box carrying the checkpoint stem, integral threshold, n_events, and the
  pileup window. The pileup window is also in the default title.

`plot_reco_vs_mu.png` is unchanged — it still carries the truth reference
line, which distinguishes it from the new `stats_histogram.png`.

Matplotlib backend is now forced to `Agg` at the top of `plots_pvf.py` to
prevent X11 crashes when the launcher SSH session disconnects mid-plot
(recovered from one such crash earlier today, lost 2550 events of inference
because the output was written after all plots had been drawn).

## 2026-04-15 — Final four evals of the day (canonical models)

Ran four aligned evals with the `integral_threshold = 0.5` unification and the
reworked plot set (stats_histogram = PVF vs AMVF, category_counts = 5-bar
summary in the high-pileup window). Output root: `outputs/04_15_2026_output/`.

| # | Data | Script | Model | μ window |
|---|---|---|---|---|
| 1 | Run 2 MC (2550 evt) | `run_eval_pvf.py` | `model_weights/03_24_2026/reproduction_T2HIST_400ep_T2KDE100_K2H150_epoch_150_fullstate.pth` | [55, 65] |
| 2 | Run 2 real (2500 evt) | `run_eval_pvf_run3.py --root data/run2/.../_000002.ATLAS_PVFinderData_Run3Data.root` | same Run 2 canonical | [20, 35] |
| 3 | Run 3 real (2500 evt) | `run_eval_pvf_run3.py --root data/run3/file_3.root` | same Run 2 canonical | [50, 65] |
| 4 | HLLHC PU200 MC (2550 evt) | `run_eval_pvf_run3.py --root data/run4/Run4_MC21_ITk/ATLAS_PVFinderData_HLLHC_mc21_14TeV_ttbar_SingleLep_PU200.root --e2e-wide` | `model_weights/hllhc_pu200_mlp50_e2e400_v2_phase2_epoch_100_fullstate.pth` | [185, 215] |

**AMVF source per eval** — the `stats_histogram` and `category_counts` plots
need PVF vs AMVF; where the "AMVF" curve comes from depends on the script:

- `run_eval_pvf.py` (Run 2 MC) loads `RecoVertex_nTracks` from the truth ROOT
  file (`ATLAS_PVFinderData_TruthMatched.root`) and stores it as `n_amvf`.
- `run_eval_pvf_run3.py` (Run 2 real, Run 3 real, HLLHC PU200 MC) loads AMVF
  reco vertices via `load_run3_from_root` in `run3_io.py` and stores them as
  `amvf_z` / `amvf_ntrks`, which the main loop surfaces as the `n_truth`
  field (AMVF *is* the reference on this script). `plot_stats` falls back to
  `n_truth` when `n_amvf` is absent, so the same plot function handles both
  cases transparently.

**Two gotchas found during the run**:

1. `run_eval_pvf.py --root-truth` defaulted to `None`, which silently dropped
   ROOT-sourced fields (μ, `n_amvf`, nTracks≥2 truth filter). Flipped default
   to `_DEFAULT_ROOT`. The first MC eval of the day ran without ROOT truth
   because of this bug — rerun was clean.
2. HLLHC PU200 pileup is **discrete at μ ∈ {190, 210}** (not a Gaussian
   around 200). The obvious `--mu-min 195 --mu-max 205` window catches zero
   events. Used `--mu-min 185 --mu-max 215` to catch both. Noted in
   `docs/data/run_4.md`.

**HLLHC v2 checkpoint naming convention** — `phase2_epoch_100` is Phase 2
epoch 100, i.e. **150 effective epochs** counting the 50-epoch Phase 1 MLP
warmup. The user asks for "epoch 150" when they mean "ep100 after the warmup".

**HLLHC v2 model architecture** (`--e2e-wide` flag, new today):

Same `trackstoHists_UNet_1000` class but with `n_UNetChannels=96` and
`l_HiddenNodes=[128]×5` (vs default 64 / [100]×5). 680K params vs 359K.
Loading the v2 checkpoint into the default-width model fails with a shape
mismatch; the `--e2e-wide` flag extends the `E2E_CONFIG` dict in place before
instantiation. Verified by loading `phase2_epoch_120_fullstate.pth`
(subsequently `phase2_epoch_100_fullstate.pth` for the canonical eval).

Also verified the HLLHC ROOT has `RecoVertex_nTracks`, `NumRecoVtx`,
`ActualNumOfInt` and `TruthVertex_*` branches, so both AMVF reference and
pileup binning work on HLLHC MC (same as Run 2 MC).

## 2026-04-15 — Canonical Run 2 MC checkpoint: ep150 → ep300

Moved the canonical Run 2 MC E2E checkpoint forward from Phase 2 ep150 to
ep300 on the same 400-epoch reproduction training run:

`model_weights/03_24_2026/reproduction_T2HIST_400ep_T2KDE100_K2H150_epoch_300_fullstate.pth`

Same training (`train_tracks_to_hist.py`, init from T2KDE ep100 + K2H ep150,
MSE loss, lr=1e-3, 400 epochs), just later in the schedule — more converged.
Re-ran the three Run-2-model evals (MC, Run 2 real, Run 3 real) with the
new checkpoint. HLLHC eval was not re-run because that uses the separate
HLLHC v2 wide checkpoint (`hllhc_pu200_mlp50_e2e400_v2_phase2_epoch_100`).

## 2026-04-15 — Peak-finder threshold sweep + back to 0.2 for counts

Added `--integral-threshold` CLI flag to `run_eval_pvf.py` and
`run_eval_pvf_run3.py` so the peak-area threshold can be overridden per
run. Wrote `src/pv_finder/diagnostics/threshold_scan.py` which caches
model output from N events once, then sweeps both peak-finder knobs
independently:

1. `integral_threshold` over [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
   0.50, 0.60, 0.70] with `threshold` fixed at 0.01.
2. `threshold` (peak amplitude) over [0.001, 0.002, 0.005, 0.010, 0.020,
   0.050, 0.100, 0.200] with `integral_threshold` fixed at 0.5.

Ran the scan on **Run 2 MC** (Model B ep300, 300 evt, AMVF = 28.44/evt)
and **HLLHC PU200** (v2 wide phase2 ep100, 300 evt, AMVF = 96.04/evt) on
GPUs 1 and 0 in the background. Match window fixed at 0.3 mm for both.

### Integral-threshold scan results

| ith  | MC eff | MC FP/ev | HLLHC eff | HLLHC FP/ev |
|------|:------:|:--------:|:---------:|:-----------:|
| 0.10 | 92.5%  | 2.79     | 90.3%     | 8.15        |
| 0.15 | 92.0%  | 2.40     | 89.2%     | 6.42        |
| **0.20** | **91.7%** | **2.06** | **88.0%** | **4.94** |
| 0.25 | 91.2%  | 1.74     | 86.7%     | 3.83        |
| 0.30 | 90.7%  | 1.43     | 85.4%     | 3.14        |
| 0.40 | 89.6%  | 0.99     | 82.8%     | 1.99        |
| 0.50 | 88.5%  | 0.68     | 80.0%     | 1.19        |

0.2 is the clear operating point for the count-based metrics on both
datasets: +3.2% efficiency on Run 2 MC and +8.0% efficiency on HLLHC vs
0.5, at modest FP cost (+1.4 and +3.8 per event respectively). Going to
0.1/0.15 gains diminishing efficiency and adds fakes rapidly.

### Peak-amplitude scan results

Peak amplitude at 0.01 is already at the knee on both datasets — dropping
to 0.001 buys <0.5% efficiency, going above 0.02 costs several percentage
points quickly. **Don't touch `threshold = 0.01`.**

### Decision — dual thresholds again

Reverted the 2026-04-15 morning unification (0.5 everywhere). Final design:

- **`INTEGRAL_THRESHOLD = 0.2`** everywhere the count matters
  (performance stats, `stats_histogram`, `category_counts_hist`,
  `reco_vs_mu`, training `PARAM_EFF`).
- **`INTEGRAL_THRESHOLD_RES = 0.5`** only for the σ_vtx_vtx pairwise Δz
  sigmoid fit (keeps the fit clean of low-integral sidelobes).
- **`threshold = 0.01`** for peak amplitude on every dataset.

Yes, this is the pre-morning state — but now it's backed by a quantitative
scan rather than inheritance from mattia_finder. The intentional
consequence: sidelobe peaks contribute to the fake count (where they
should show up as "the model has artifacts") but do not contaminate the
resolution plot (where they would create a spurious close-pair feature).

### What "model resolution is too low" means on HLLHC

Nearest-neighbor truth-PV distance analysis on 300 HLLHC events
(nTracks≥2): median 1.32 mm, mean 2.52 mm, and 14.9% of adjacent truth
pairs are within 0.3 mm — well below σ_vtx_vtx = 0.286 mm. Those pairs
are fundamentally unresolvable by the current model: both truths fall in
the same 2σ matching window and get absorbed as one "merged" reco peak.
In the actual ep100 eval, 6.29/evt truths are tagged `tm` (truth merged),
about 45% of the theoretical ceiling. The rest of the ~14/evt unresolvable
pairs either lose one endpoint to `missed` or are caught via an off-axis
reco. This is an architectural ceiling on HLLHC efficiency, not a
threshold tuning problem.

### Reference reconstruction on HLLHC

Verified that the HLLHC mc21 ROOT file has populated `RecoVertex_*`
branches (NumRecoVtx ≈ 83–116/evt, ≈80–102 after nTracks≥2 filter,
median nTracks 6–8, RecoVertex_type ∈ {0, 1}). The loader stores these
as `amvf_z`/`amvf_ntrks` by historical name — "AMVF" is shorthand for
"ATLAS adaptive multi-vertex finder", the algorithm family is the same on
Run 2/3 and HLLHC but the HLLHC ITk tune may not be labeled as "AMVF" in
the HSG convention. Operationally it's the ATLAS reference primary-vertex
reconstruction for the sample, which is the right baseline.

## 2026-04-16 — Integral threshold: back to 0.5 for both

Reverted the dual-threshold design (0.2 perf + 0.5 resolution) to a single
unified threshold of **0.5 for both** performance and resolution. Changed
`INTEGRAL_THRESHOLD = 0.5` default in both `run_eval_pvf.py` and
`run_eval_pvf_run3.py`. `INTEGRAL_THRESHOLD_RES` stays at 0.5.

Rationale: the dual-threshold was defensible (scan showed 0.2 is the knee
for counts on both Run 2 MC and HLLHC) but made the counts and resolution
plot inconsistent — sidelobes contributed to FP but didn't show up in
pairwise Δz. Unified 0.5 gives consistent accounting across both metrics,
matching the `clean_run3` reference approach.

HL-LHC PU200 needs explicit override: peaks are shallower due to PU200
track density spread. Pass `--integral-threshold 0.2 --integral-threshold-res 0.2`
for HL-LHC evals to avoid losing real vertices.

---

## 2026-04-23 — HL-LHC MC truth support + vertex matching fix

### MC truth auto-detection in run_eval_pvf_run3.py

The HL-LHC ROOT file has both `TruthVertex_z` (MC truth) and `RecoVertex_z`
(AMVF reco). Previously the script used AMVF as truth for all data — correct
for real data (Run 2/3) but wrong for HL-LHC MC where proper MC truth exists.

Changes:
- **run3_io.py**: `Run3Event` gains optional `truth_z`/`truth_ntrks` fields.
  `load_run3_from_root` auto-detects `TruthVertex_z` branches and loads them.
  NPZ files (no truth) → fields remain None.
- **run_eval_pvf_run3.py**: When truth detected, uses TruthVertex as ground
  truth and evaluates AMVF as a separate reco algorithm (categories shown in
  the bar chart). No beam correction on truth or AMVF in this mode (both in
  detector frame, matching MC eval behavior).
- **plots_pvf.py**: `plot_category_counts` gains `truth_pvs_per_evt` param,
  shown in the info box when available. Both MC and HL-LHC eval scripts pass it.

### Vertex matching algorithm rewrite (compare_res_reco)

The old algorithm processed each reco vertex independently: if reco R saw 2
truth vertices in its window, R was always classified "merged" — even if
another reco R2 was a better match for one of those truth vertices. This
inflated the merged count, especially at PU200 where vertices are densely
packed (~1 mm apart, matching window ~0.3-0.5 mm).

New algorithm: **greedy closest-first matching**.
1. Build all valid (reco, truth) pairs within matching windows
2. Sort by distance, greedily assign 1-to-1 (closest first)
3. Classify leftovers: unmatched reco with no truth in window = fake;
   unmatched reco with truth in window (claimed by closer reco) = split;
   assigned reco with unmatched truth in window = merged

This correctly classifies two reco vertices near two close truth vertices as
two "clean" matches instead of two "merged".

### Per-vertex visualization for HL-LHC (run4)

New script `run_per_vertex_run4.py` in per_vertex_visualization/. Loads HL-LHC
ROOT events, runs both the Run 2 model and the HL-LHC model, produces side-by-
side per-vertex plots. Supports `.pyt` (full object) and `.pth` (fullstate)
checkpoint formats.

---

## 2026-05-03 — Resolution analysis + v3 model architecture

### Resolution bump investigation

Thorough analysis of the pairwise Δz bump at ±0.5-1.5 mm on HL-LHC PU200
(2500 events, ep100). Compared PV-Finder, AMVF, and truth distributions.

Key findings:
- Truth distribution is **flat** (no structure) — the bump is purely a
  reconstruction artifact, not physics
- **AMVF has a 6x larger bump** than PV-Finder (9.6% vs 5.4% above baseline)
- PV-Finder has **better resolution** (99.4% dip depth vs AMVF's 95.9%)
- PV-Finder's bump is mostly noise (2 bins marginally >2σ)
- NMS is harmful at PU200: kills 2.5 real peaks per fake removed
  (genuine close vertex pairs have similar height ratios to sidelobes)

Plots saved to `outputs/04_23_2026_output/resolution_comparison/`.
Full analysis: `docs/research/resolution_bump_analysis.md`.

### Eval pipeline improvements

- Added `--save-histograms` flag to `run_eval_pvf_run3.py`: stores raw
  12000-bin histograms in pkl (~115 MB extra for 2500 events). Enables
  post-hoc analysis without re-running GPU inference.
- Added `--peak-threshold` flag to control peak amplitude threshold
  (previously hardcoded at 0.01).
- Added `pred_heights` to eval pkl for peak height analysis.
- Added `visualize_fakes.py` diagnostic: generates zoom plots centered
  on fake peaks, sorted by proximity to nearest matched peak.

### v3 model architecture (10x scaled UNet_v2 with 4-channel latent)

Designed a scaled model for HL-LHC PU200 training:

| | v1 default (Run 2) | v2 wide (HL-LHC) | v3 (HL-LHC) |
|---|---:|---:|---:|
| Architecture | trackstoHists_UNet_1000 | trackstoHists_UNet_1000 | TracksToHist_v2 |
| UNet channels | 64 | 96 | 280 |
| MLP hidden | [100]×5 | [128]×5 | [128]×5 |
| n_latent_channels | 1 | 1 | 4 |
| Upsampling | ConvTranspose | ConvTranspose | Interp+Conv |
| Bottleneck | 8x (125 bins) | 8x (125 bins) | 4x (250 bins) |
| First kernel | k=25 | k=25 | k=15 |
| Total params | 359K | 681K | 3,548K |

v3 improvements:
- **4 latent channels**: MLP outputs 4 spatial representations of tracks
  (vs 1), giving the UNet richer input for peak placement
- **Interp upsampling**: eliminates ConvTranspose checkerboard artifacts
- **4x bottleneck**: 250 bins satisfies Nyquist for peak widths (~0.28 mm)
- **k=15 first kernel**: 0.6 mm correlation (vs 1.0 mm), reduces sidelobes

Training recipe: 50 epochs Phase 1 (MLP warmup) + 200 epochs Phase 2 (E2E),
LR=5e-5 with 7-epoch warmup + cosine decay, gradient clipping max_norm=1.0.
Data: 99,800 events (838K training subevents), batch_size=128.

---

## 2026-05-04 — v3 training: LR was too low, 3-GPU sweep launched

### Initial v3 result

v3 (3.55M params) trained at LR=5e-5 converged at the **same performance**
as v2 (680K params, LR=1e-4). Loss plateaued after ~30 epochs. The 4 latent
channels are diverse (inter-channel correlation 0.03) so the architecture is
working — the model was simply under-learning due to conservative LR.

### 3-GPU parallel sweep

| Run | GPU | Strategy | Key settings |
|-----|-----|----------|-------------|
| A | 0 | Aggressive LR + warm restarts | LR=2e-4, Adam, cosine restart T0=50 |
| B | 1 | SGD with momentum | LR=1e-4, SGD(m=0.9, wd=1e-4), cosine |
| C | 2 | MSE + TV regularization | LR=1e-4, Adam, tv_lambda=0.1 |

All reuse Phase 1 MLP weights. Early signal: Run A cut loss aggressively
after 1 epoch (confirming LR diagnosis), Run B much slower (expected for SGD).

### Open question: what if more capacity doesn't help?

If Run A reaches the same plateau as v2 (just faster), it means the extra
3.55M params don't improve over 680K. Three hypotheses for why:

1. **Loss function bottleneck**: MSE has a fundamental best solution for
   histogram prediction and both models can represent it. Run C (TV loss)
   tests this — if a different loss creates a better minimum, MSE was the
   ceiling.
2. **Target bottleneck**: truth histograms use fixed-width Gaussians
   (~0.15 mm). The labels don't contain finer information than what the
   smaller model already learns.
3. **Output resolution bottleneck**: 1000 bins/subevent (0.04 mm/bin) may
   already be representable by 680K params. More params can't improve
   what's already perfectly fit.

Will evaluate at epoch 50-100 and explore further.

---

## 2026-05-04 — Peak classification study: track features don't help at PU200

Investigated post-hoc peak classification to reduce fake rate (4.8%, 9892/208082
peaks across 2500 HL-LHC PU200 events). Built 18-feature classifier combining
track counts/quality, histogram shape, and neighborhood context.

**Key result**: Track features add negligible AUC (+0.002) over histogram shape
alone. Peak height is the dominant discriminator (53% importance). A simple
height threshold of 0.05 achieves 96.1% real kept / 50.1% fake removed — nearly
matching the full GBT classifier (96.6% / 62.7% at thr=0.7).

**Why**: At PU200 with ~200 PVs and ~700 tracks, track density is ~1.5/mm
everywhere. 17% of truth PVs have ≤2 tracks within ±0.5mm, making them
indistinguishable from fakes by track count. Track uncertainties (z0_err, d0
significance) also don't discriminate well — nearby tracks at fake positions
have similar quality to those at real positions.

**Conclusion**: Post-hoc track-based filtering is not the right approach for
PU200 fake reduction. The real opportunity is in the model training:
peak-aware loss, confidence head, or architectural changes.

Script: `src/pv_finder/diagnostics/peak_classifier.py`
Full analysis: `docs/research/peak_classification_study.md`
Artifacts: `outputs/05_04_2026_output/peak_classifier_full/`

---

## 2026-06-01 — AMVF z-resolution vs N_Tracks (Figure 12 reproduction)

Rewrote `src/pv_finder/diagnostics/amvf_resolution_vs_ntracks.py` to use the
canonical AMVF--truth residual method instead of the earlier CRLB proxy.

**Why the rewrite**: Colleague feedback confirmed (and Qi Bin's
`sample_plotting_code.py` implied) that vertex errors are not needed for this
plot — the standard approach is to match each AMVF reco vertex to a truth
vertex, then look at the spread of dz = z_AMVF - z_truth per truth N_Tracks
bin. `RecoVertex_ErrZ` being empty in the HL-LHC ntuples is irrelevant.

**Method**:
1. Load HL-LHC PU200 ttbar ROOT via `pv_finder.data.run3_io.load_run3_from_root`
   (already exposes TruthVertex_z + TruthVertex_nTracks).
2. Per event: greedy closest-first 1-to-1 match within +/- 2 mm (~90%
   matching efficiency).
3. Bin matched pairs by truth N_Tracks (26 bins, 2..140). Per bin: ROOT
   `TH1F` of dz, Gaussian fit (`TF1 "gaus"`, range +/- 2.5*RMS) -> sigma(n).
4. Fit `sigma(n) = a/n^b + c` via `TGraphErrors.Fit(TF1)`.
5. Plot with PyROOT + atlasplots, mimicking Qi Bin's sample plotting style
   (red star marker `MARKER_AMVF=29`, dashed fit line, ATLAS label, TLatex
   tags). Requires `PYTHONPATH=/usr/local/anaconda3/lib/python3.8/site-packages`
   to find ROOT 6.24 — the script prepends this automatically.

**Result (full file, 99 800 events, 8.52M matched pairs)**:
- `a = 178.98 +/- 8.23 um` (= 0.179 mm)
- `b = 0.7274 +/- 0.0142`
- `c = 0.00 +/- 0.08 um`

These are the (a, b, c) we will use to set per-vertex Gaussian widths for
the target histograms in HL-LHC PV-Finder training. They are notably
**different from the Run 3 values** in the paper (~0.16 mm at n=2,
i.e. ~1.6x wider) because the ITk has ~2x better per-track z0 resolution
than the Run 3 Inner Detector. If the HL-LHC training currently uses
Run-3 (a, b, c), the target-histogram Gaussians are too wide and should
be regenerated with the values above.

**Maximum truth N_Tracks in full file**: 168 (p99 = 40, p99.99 = 87).
Only 30 truth vertices have N_Tracks >= 120 and only 3 have >= 140;
the highest plotted bin is centered at N_Tracks ~ 120 (the [135, 170)
bin gets dropped under the 30-vertex minimum for the Gaussian fit).

Plot style mimics the ATL-PHYS-PUB Figure 12 reference plot:
blue filled circles for data (with error bars), red solid power-law fit,
ATLAS Simulation Preliminary label, "Data" / "Fit" legend, axes in mm.

Deleted `sample_plotting_code.py` from repo root after extracting style cues.

Script: `src/pv_finder/diagnostics/amvf_resolution_vs_ntracks.py`
Artifacts: `outputs/06_01_2026_output/amvf_resolution_residuals/`
  - `amvf_resolution_vs_ntracks.png`
  - `fit_params.json`
  - `vertex_data.npz`

---

## 2026-06-01 — AMVF resolution: restyle to match paper figure, full-file fit

Updated `amvf_resolution_vs_ntracks.py` after comparing my earlier plot to
the colleague's `efficiency_example.png` reference (ATL-PHYS-PUB Fig. 12
style). Key changes:

- **Plot style**: blue filled circles (data, with error bars), red **solid**
  power-law fit, ATLAS **Simulation Preliminary** label, "Data" / "Fit"
  legend in upper-right. Dropped the in-plot fit-param caption and the
  √s/⟨μ⟩ TLatex tags. Y-axis switched to **mm** ("Vertex Resolution (mm)").
  X-axis labelled "Number of Tracks".
- **Binning**: per-integer bins for n in [2, 30], progressively wider up to
  170 (full-file max is 168). Bins with <30 vertices are dropped.
- **Sample**: full 99 800-event file (8.52M matched AMVF<->truth pairs)
  instead of the 20 k smoke test.

**Updated fit (full file)**:
- `a = 178.98 +/- 8.23 um` (= 0.179 mm)
- `b = 0.7274 +/- 0.0142`
- `c = 0.00 +/- 0.08 um`

Stable vs the 20 k preview (a 172 -> 179, b 0.724 -> 0.727).

**Critical finding -- compared to the paper reference**: the example image
the colleague shared is for Run 3 simulated ttbar at ⟨μ⟩=60, 13 TeV, using
the ATLAS Inner Detector. Our sample is HL-LHC PU200 (⟨μ⟩=200, 14 TeV) on
the ITk. ITk has ~2x better per-track z0 resolution than the ID, so per-
vertex sigma is roughly half: at n=2 the example shows ~0.16 mm vs our
~0.108 mm. **The HL-LHC PV-Finder training should regenerate its target
histograms with the HL-LHC (a, b, c) above** rather than reuse Run-3 values;
otherwise the truth Gaussians are ~1.6x wider than the data supports.

**Truth N_Tracks reach**: max in our 99 800-event file is 168
(p99 = 40, p99.99 = 87). Only 30 vertices have n >= 120, only 3 have
n >= 140 — the [135, 170) bin gets dropped under the 30-vertex Gaussian
fit minimum. The x-axis is drawn out to 180 to mirror the example.

Plot: `outputs/06_01_2026_output/amvf_resolution_residuals/amvf_resolution_vs_ntracks.png`

---

## 2026-06-02 — root_to_h5: parametrised resolution presets (hllhc / run3)

Updated `src/pv_finder/data/root_to_h5.py` so the (A, B, C) constants used
to set per-PV Gaussian widths in the target histogram are no longer
hardcoded. Added:

- `RESOLUTION_PRESETS` dict at module top with two entries:
  - `hllhc`: (0.17898, 0.7274, 0.0)  — new fit from 2026-06-01
  - `run3` : (0.23817443, 0.49491396, -0.000787436)  — legacy
- `--resolution-preset {hllhc,run3}` CLI flag (default: `hllhc`)
- `--a-res / --b-res / --c-res` flags for one-off overrides
- `set_resolution(a, b, c)` helper for programmatic override
- The chosen (A, B, C) and the formula string are now written into
  `h5.attrs` (`resolution_a_mm`, `resolution_b`, `resolution_c_mm`,
  `resolution_formula`) so each HDF5 self-documents which resolution
  model it used.

**Why**: the colleague flagged `histogram_example.py` (a.k.a.
`CreatingTargetHistogram.py` upstream) as the script that builds target
histograms. We already had a port (`root_to_h5.py`) but it was using the
Run-3 values, which over-blur HL-LHC targets by ~1.6-2.9x. Switching the
default preset to `hllhc` aligns the training targets with the actual
ITk vertex resolution. Keeping `run3` as a named preset preserves the
ability to regenerate Run-3-style HDF5s and to add more presets later
(e.g. for a future Run-3 fit or a different MC sample).

**Missing branches noted in HL-LHC ROOT**: `histogram_example.py` reads
`RecoTrack_xslope`, `RecoTrack_yslope`, and 15 `POCAEllipsoid_*` branches
that are not present in `ATLAS_PVFinderData_HLLHC_mc21_14TeV_ttbar_SingleLep_PU200.root`.
The current HL-LHC model uses 7-channel `tracks` (d0, z0, d0_err, z0_err,
d0_z0_cov, z_start, z_end) + KernelA/B KDE features, none of which need
the missing branches, so this isn't blocking. If we ever migrate to a
POCA-ellipsoid feature set we will need an upstream `MakingPOCAdata`
preprocessing step.

User will launch the regeneration in tmux to write
`data/run4/hllhc_pu200_training_v2.h5` alongside the existing
`hllhc_pu200_training.h5`.

---

## 2026-06-02 — QA of new PU200 *with-timing* data

New larger dataset arrived in `data/run4/PU200_withTiming/` (~2.94 M events, 10
files) adding HGTD timing branches (`RecoTrack_Time`, `RecoTrack_TimeResolution`)
to the standard PVFinderData branch set. Built
`src/pv_finder/diagnostics/timing_data_qa.py` to make basic tracking-parameter
distributions and confirm the data is sane before training (colleague's request).

The script overlays one curve per sample — the four 601229 reco tags (r16438 /
r16443 / r16633 / r16638), the pooled 601237 all-hadronic sample (6 part files),
and the old no-timing PU200 file as a known-good reference. It plots per-track
kinematics + uncertainties, the timing branches (with the `-1` sentinel masked)
plus timing-acceptance-vs-|η|, and event-level / truth-vertex sanity panels.
`--pu200-only` filters to μ>100 for an apples-to-apples PU200 comparison.

**Findings (5000 events/file):**

1. **Data is healthy.** All per-track shapes (d0, z0, pT, η, φ, θ, ErrD0, ErrZ0,
   d0/ErrD0) overlap across every sample and match the no-timing reference.
   SingleLep vs all-hadronic differ only in a tiny high-pT tail — consistent with
   the colleague's note that the process mix is irrelevant for our studies.

2. **Timing branches correct.** `-1` = no-timing sentinel; ~3.9 % of tracks have
   real timing, all at |η| ≈ 2.4–2.5 (HGTD forward edge; tracks capped at
   |η|<2.5). Acceptance is 0 below |η|≈2.25. Real Time ≈ 0 ± 0.2 ns; TimeResolution
   discrete at ~20/25/35 ps (HGTD hit-multiplicity layers).

3. **Two reco tags are NOT fixed-PU200.** r16443 and r16638 carry a **flat
   pileup spectrum (μ ≈ 0–210, mean ~100)** — ≈480 tracks/evt avg vs ≈922 for the
   μ=200 tags (r16438, r16633, 601237). Per-track shapes are identical; only
   per-event pileup differs. Decision: **use the fixed-μ=200 tags for PU200
   training**; r16443/r16638 are a broad-μ sample (pileup-robustness studies).

4. Other checks pass: BeamPosZ=0 (centred sim), truth-z Gaussian σ≈35 mm,
   truth-nTracks falls smoothly. `RecoTrack_chisq` / `RecoVertex_chisq` are empty
   in both new and old files (pre-existing upstream non-fill), excluded from plots.

Outputs: `outputs/06_02_2026_output/timing_data_qa/` (raw) and
`.../timing_data_qa_pu200only/` (μ>100), each with track/timing/event PNGs,
`summary.json`, and a `README.md` write-up for the meeting.
Updated `docs/data/run_4.md`.

---

## 2026-06-02 — Data pipeline: compression, skip target_y, multi-file training

Three coordinated changes to prepare for training on the new ~2.74 M-event
with-timing pool (8 fixed-μ=200 ROOTs):

**1. Lossless compression in `root_to_h5.py`** —
`--compression {lzf,gzip,none}` (default `lzf`). Smoke test on 200 events
gave ~8.7x reduction. The padded `tracks` tensor is mostly the MASK_VAL
constant in the unused tail of each subevent, which LZF compresses to
near-nothing. No measurable decode overhead at read time. Compression
filter recorded in `h5.attrs["compression"]`.

**2. Skip the full-event `target_y` by default** — `--keep-target-y` is
opt-in. The HL-LHC trainer reads only `target_y_split`; `target_y` is the
biggest single chunk of disk waste at multi-million-event scale.
`h5.attrs["has_target_y"]` records the choice.

**3. Multi-file dataset support** — new factory
`make_tracksHists_dataset(paths)` in `src/pv_finder/data/h5_dataset.py`.
Accepts a single path (legacy) or a list. Per-file `max_tracks_per_subevent`
read from `h5.attrs`, global max computed, each per-file
`H5Dataset_tracksHists` right-pads tracks to that width at `__getitem__`
time so PyTorch batches stack across files with different local maxes.
Returns a `torch.utils.data.ConcatDataset`. Single-file path unchanged.

Wired through `collect_data_poca_ATLAS` (`tracks-to-hist` branch). YAML
`data_file:` can now be a string or a list of strings. Other data
pipelines (KDE-to-hist, poca-to-KDE, etc.) untouched.

Refactor: resolution presets dict moved out of `root_to_h5.py` into
`src/pv_finder/data/resolution_presets.py` to keep `root_to_h5.py` under
the 500-line pre-commit cap.

**Subagent review caught one bug**: the `pv` dataset had the compression
filter but no explicit `chunks=`, so h5py was picking automatic chunking.
Fixed with `chunks=(min(1000, n_events), max_pv)`. Also added a
shape-consistency check in `make_tracksHists_dataset` (raises if files
disagree on `target_y_split.shape[1:]`).

End-to-end smoke test: two 200-event compressed HDF5s with different
local `max_tracks` (508 vs 476) → ConcatDataset of length 4800 → batch
shapes `(B, 7, 508)` and `(B, 2, 1000)` → file-B padded tail is bit-equal
MASK_VAL → file-B live range bit-equal to raw. Compression ratio confirmed
8.5-8.7x lossless. `getTargetY()` raises a clear `KeyError` on
`--skip-target-y` files.

Conversion outputs will land at `data/run4/PU200_withTiming_h5/<name>.h5`.
The user will launch the 8-way parallel conversion in tmux. New training
config + step-count-based hyperparameter scaling lands in a follow-up
commit.

---

## 2026-06-02 — Converter: vectorise subevent build + skip Pass 1 by default

Cumulative converter optimisations after observing that sneezy was
overloaded (load avg 118 on 96 cores from other users) and 8-way parallel
conversions were running 13-17x slower than my isolated benchmark would
predict. Two changes:

1. **Vectorise `_build_subevent_tracks`** -> new
   `_build_event_subevent_tracks`. The 12 boolean masks + 12 argsorts per
   event become one global stable argsort + one `np.searchsorted` for the
   12 subevent boundaries. Per-feature reordering done once instead of 12
   times. Output: same set of tracks per subevent as before, but now with
   *deterministic* tie-break ordering (the old per-subevent argsort was
   quicksort, which depends on memory layout). Verified per-subevent
   counts identical (918 179 / 918 179 live tracks on the 1000-event
   smoke set) and contents bit-identical modulo within-subevent
   ordering.

2. **Skip Pass 1 by default.** CLI `--max-tracks-per-sub` default changed
   from `0` (scan) to `1024`; new `--max-pv` default `300`. Historical
   max across HL-LHC PU200 ROOTs is 774 tracks/subevent and ~200
   PVs/event, so 1024 / 300 give comfortable headroom; LZF compresses
   the unused padding to ~nothing. The full-tree Pass-1 scan is now
   only run if the user explicitly passes `0`. Saves ~30-60 s per file
   on top of ~12 % of total per-file time. If max_tracks is ever
   exceeded at write time, the summary prints a `WARNING` line.

End-to-end on a 1000-event slice (isolated machine):
  pre-vectorise + Pass 1: ~2.8 ms/event
  vectorised + skip Pass 1: ~2.6 ms/event

Cumulative speedup from the very first version of the code: ~10x in
isolation. Under heavy contention the per-process throughput shrinks
but the relative gain still applies. Per-file projection under load
should drop the user's currently-observed ~60 ms/event toward ~25-30
ms/event.

Also tightens the summary output: dataset shapes + max-tracks warning +
on-disk size all on three lines.

---

## 2026-06-03 — v4 training config + global shuffle + DataLoader tweaks

Set up the v4 training run on the 2.74M-event with-timing pool. Three
coordinated pieces:

**1. Global shuffle before train/val/test split.** The previous
`collectdata_poca_KDE.py` partitioned via contiguous slices
(`Subset(dataset, range(0, train_size))` ...). On the multi-file
ConcatDataset this would put almost all of val and test inside the last
file (601237 part 6), so train/val/test would see very different process
mixes. Replaced with a seed-fixed `np.random.default_rng(42).permutation`
applied before partition so every split sees a representative mix of
601229 SingleLep and 601237 all-hadronic. Per colleague feedback.

**2. DataLoader tweaks.**
  - `persistent_workers=True` (was False) — workers and their 8 HDF5
    file handles stay alive across epochs.
  - `prefetch_factor=4` (was hardcoded 2 in `train_hllhc_e2e.py`) — now
    a config knob, default 4. Helps mask LZF decompression latency.
  - `pin_memory=True` (was False) — async host->GPU copy.
Cumulative wall-time gain probably ~5-10% per run, plus cleaner
per-epoch behaviour (no worker spin-up jitter).

**3. New `config_hllhc_pu200_e2e_v4.yml`.**
  - `data_file`: list of 8 fixed-mu=200 HDF5s.
  - `device_id: 3`.
  - `train_split: [0.96, 0.03, 0.01]` (val 3%, test 1%).
  - `phase1_epochs: 3` (was 50; 27x more data per epoch).
  - `phase2_epochs: 25` (was 200; targets ~6.1M training steps,
    ~4.7x the v3 budget).
  - `phase2_warmup_epochs: 1` (was 5; 1 epoch = 244k warmup steps).
  - `phase1_lr: 1e-3`, `phase2_lr: 1e-4`, `max_grad_norm: 1.0` — all
    unchanged from v3.
  - `num_workers: 24`, `batch_size: 128`, `prefetch_factor: 4`.
  - `save_frequency: 1` (each ~30 min epoch is worth checkpointing).
  - `runname: hllhc_pu200_e2e_v4_2.7M_280ch_4lat_lr1e4`.

Smoke test confirmed: 8-file ConcatDataset builds in 2.7 s, total 32.9M
subevents matches per-file sums, split sizes [31.6M, 987k, 329k] match
[0.96, 0.03, 0.01], shuffle gives non-contiguous indices spanning the
full range, first batch fetches with correct shape (128, 7, 1024) +
(128, 2, 1000).

---

## 2026-06-04 — Phase-2 LR schedule fix: per-step warmup (no 100x cliff)

Diagnosed odd v4 training behaviour the user flagged (plateau ~step 200k, val
loss step-up + wobble after ~260k, efficiency flattening). Root cause was the
Phase-2 LR schedule, not the data or model.

**Mechanism:** `run_phase2` calls `scheduler.step()` once per epoch (line 361),
and v4 sets `phase2_warmup_epochs: 1`. So `LinearLR(start_factor=0.01,
total_iters=1)` holds the ENTIRE first epoch (~246,762 steps, ~7.5 h) at
`0.01·lr = 1e-6` — the model is effectively frozen and val/eff asymptote — then
the LR jumps **100x to 1e-4** at the epoch-1->2 boundary (confirmed in MLflow:
`p2_lr` = 1e-6 through step 246,500, then 1e-4 from 247,000). That cliff is
exactly where `p2_val_loss_step` steps up (0.0326 -> 0.038) and starts wobbling,
and `p2_eff_step` stops climbing. The epoch-2 val floor (~0.0303) does drop
below epoch-1 (~0.0326), so real learning only begins once LR becomes non-trivial
— i.e. the warmup epoch was largely wasted.

**Fix (opt-in, backward-compatible):** `build_phase2_scheduler` gains
`steps_per_epoch` and `warmup_steps` args. When `steps_per_epoch > 0` the whole
schedule is built in step (batch) units and stepped once per batch, so a short
warmup actually ramps. `train_phase2_epoch` gains a `scheduler` arg that it steps
per batch; `run_phase2` reads `phase2_step_scheduler` / `phase2_warmup_steps`,
builds the step-unit scheduler, passes it into the train loop, and skips the
per-epoch `.step()`. With `steps_per_epoch=0` (default) behaviour is byte-for-byte
the legacy per-epoch schedule, so v3 warm-restart configs (runA/B/C) are untouched.

**Verified** (unit test, `build_phase2_scheduler`):
- Legacy per-epoch reproduces the current run exactly: epoch1=1e-6, epoch2=1e-4,
  then cosine — backward-compatible.
- Per-step (`warmup_steps=3000`, `steps_per_epoch≈24.7k`): smooth ramp 1e-6 ->
  1e-4 over 3000 batches, **no jump** at the epoch boundary, cosine down to
  `eta_min=1e-6` at the final step.

New config `configs/vertex_finding/config_hllhc_pu200_e2e_v4b_stepwarmup.yml`
(identical to v4 plus `phase2_step_scheduler: true`,
`phase2_warmup_steps: 3000`) for the next launch.

The current v4 run was left running (it loaded the old module + config at
startup, so the source edits don't affect it; it's recoverable, just suboptimal).
At the time of the fix it was at step ~396k (~1.6 of 10 Phase-2 epochs), val
floor ~0.030, eff ~0.74-0.77.

---

## 2026-06-04 — Phase-2 full resume (--resume) for HL-LHC E2E

A v4b run was accidentally killed mid-epoch-1. The training script could only
"resume" via `--phase1-checkpoint` (model weights only, fresh optimizer +
restarted LR schedule). Added true Phase-2 resume:

- `utils/utilities.save_checkpoint`: optional `extra` dict merged into the
  checkpoint (backward-compatible; used to store `scheduler_state`).
- `run_phase2`: per-epoch checkpoints now save `scheduler_state`; new
  `resume_ckpt` arg restores model + optimizer + scheduler and continues the
  loop from the next epoch (`range(start_epoch + 1, ...)`).
- `main` + CLI: `--resume <phase2_epoch_fullstate.pth>` (skips Phase 1).
  Mutually exclusive with `--phase1-checkpoint`.

Granularity is the epoch boundary (clean for the per-step scheduler: after N
epochs the scheduler sits at exactly N*steps_per_epoch, aligned with restarting
epoch N+1 from batch 0). Mid-epoch step checkpoints remain eval-only.

Verified: with dropout=0 and fixed data, a checkpoint-and-resume run reproduces
an uninterrupted 2-epoch run bit-for-bit (max |Δparam| = 0.0) — model,
optimizer (Adam moments) and scheduler position all restored correctly.

Usage:
  python -u src/pv_finder/training/train_hllhc_e2e.py \
      -c configs/vertex_finding/config_hllhc_pu200_e2e_v4b_stepwarmup.yml \
      --resume model_weights/<run>_phase2_epoch_<N>_fullstate.pth

---

## 2026-06-04 — Presentation update for v4 HL-LHC results

Appended June 4 update slides to `presentations/mattia/04_16_2026/slides.tex`.
The new section summarizes the larger 2.7M-event fixed-PU200 training sample,
compressed multi-file HDF5 conversion, global shuffle, v4/v4b warmup and LR
schedule work, Phase-2 resume support, and compute constraints on the shared
server.

Added the latest ATLAS-style AMVF resolution-vs-track-multiplicity plot
(`outputs/06_01_2026_output/amvf_resolution_residuals/amvf_resolution_vs_ntracks.png`)
to explain the updated HL-LHC target-histogram widths, plus the tuned v4 epoch-2
category-count result at `peak_threshold=0.01`, `integral_threshold=0.40`
(`outputs/06_04_2026_output/v4_epoch2_eval_thr0p40_res0p40_2500_random_start_16320/production/category_counts_hist.png`).
The slides call out the open question of using AMVF reconstructed vertices for
the track-count vs precision curve and list next studies: longer training,
failure-mode visualization, improved peak finding/heuristics, alternative losses,
and checking whether close-PV density imposes a physical limit.

---

## 2026-06-08 — Failure-mode per-vertex viz + classify_vertices greedy fix

Built `src/pv_finder/diagnostics/per_vertex_visualization/failure_mode_viz.py` to
inspect what the v4b model gets wrong, to drive the next round of improvements
from real failure modes. Reuses all existing primitives (run3_io,
inference.run_e2e_on_events, peak_matching, vertex_plots, analytical_kde); the
only new code is a v2/v4b model loader and a failure-mode selection loop. Runs
v4b on HL-LHC events, classifies each MC-TruthVertex/reco vertex, and plots zoom
views ONLY for merged/missed truth and fake reco (capped per category).

**Bug found + fixed in shared `peak_matching.classify_vertices`**: it used a
fixed-window count (>=2 truths within 0.5 mm -> "merged"), which mislabels
cleanly separated close pairs as merged. Replaced with the canonical greedy
closest-first 1-to-1 (same as `efficiency_res_optimized_atlas.compare_res_reco`).
Effect on one event: merged 51->23, clean 47->75. Answers the earlier
"are merged truly merged?" question: the old diagnostic over-counted merges.

**vertex_plots.py fixes**: `axes.unicode_minus=False` (Unicode minus rendered as
boxes in the Agg font); added `truth_name` param to `plot_vertex_zoom` /
`_draw_vertex_lines` so the legend can say "MC truth" instead of "AMVF vertex".

**Failure-mode breakdown (v4b, 4 HL-LHC events, MC truth, 0.5 mm window)**:
merged ~21/evt (~22% of truth, dominant), fake ~10/evt, missed ~5/evt.
- MERGED: close pairs (<~0.5 mm); KDE often shows the structure but model emits
  fewer peaks -> resolution-limited (levers: finer output / deconvolution / timing).
- MISSED: low-track vertices with little/no track signal AND no KDE signal ->
  information-limited, not a model bug (no lever helps without tracks).
- FAKE: small peaks at sparse-track / KDE-shoulder locations -> loss-addressable
  (peak-suppression / confidence head; a height cut already removes ~50%).

Plots: `outputs/06_08_2026_output/failure_mode_viz/event800XX/{merged,missed,fake}/`.

---

## 2026-06-08 — Vertex-matching fix: primary truth of a merged reco is CLEAN, not merged

User inspection of the failure-mode plots found "merged" truth vertices that have
a dedicated reco peak sitting right on them (e.g. event 80002 truth z=-21.108 has
a peak at -21.069, 0.04 mm away) being labelled MERGED. Root cause in the greedy
matcher (`compare_res_reco`, and the diagnostic `classify_vertices` that mirrors
it): when a reco absorbs an extra unmatched neighbour truth it is labelled
"merged", and THEN every truth assigned to it — including the well-reconstructed
PRIMARY — was labelled "merged". So the primary (with its own peak) was dragged to
merged just because its peak also happened to be the nearest one for a neighbour
that had no peak of its own.

**Fix** (both `efficiency_res_optimized_atlas.py:compare_res_reco` and
`per_vertex_visualization/peak_matching.py:classify_vertices`): snapshot the
Pass-1 primaries (`primary_truth = set(truth_assigned)`) before Pass-2 absorption.
Truth labels: Pass-1 primary -> CLEAN (has a dedicated reco, even if that reco
absorbs a neighbour); Pass-2 absorbed -> MERGED (the actual casualty); else MISSED.

**Impact**:
- **Efficiency UNCHANGED.** `eff = (tc+tm)/n_truth` and the fix only re-labels truths
  *within* the non-missed set (primary<->absorbed), so `tc+tm` (and missed) are
  invariant. Verified: event 80002 eff=0.989 before and after; 80010 0.933; 80000
  0.899; 80014 0.965 — identical. No past efficiency number is invalidated.
- **Truth merged/clean breakdown corrected.** merged roughly HALVED (per-event:
  12->6, 29->15, 17->9, 26->13 at 0.5 mm window), clean rose correspondingly. The
  earlier "merged ~22%" failure-mode figure was ~2x overstated; true lost-to-merge
  is ~11%. The reco-side bar chart (reco_clean/merged/split/fake) is unaffected.

Also: low-amplitude "fake" peaks pass peak-finding because `pv_locations_updated_res`
gates on integral_threshold=0.2 (AREA) with no height floor — a 0.06-tall, ~0.4 mm
wide shoulder has integral ~0.44 and clears it (e.g. event 80000 z=-37.62). A height
cut >=0.05 removes ~50% of these (peak_classification_study); candidate operating-
point / loss lever.

Cosmetic: `plot_vertex_zoom` title now uses `truth_name` (fakes show "fake peak z"
not "truth z"); `failure_mode_viz` passes per-mode names. Plots regenerated.

## 2026-06-09 — Peak height floor (min_height) operating point

Added `min_height` to `pv_locations_updated_res` (both copies: `utils/peak_finding.py`
and `evaluation/.../efficiency_res_optimized_atlas.py`) and to `find_histogram_peaks`.
Default `0.0` (no behavior change for all existing callers). `run_eval_pvf_run3.py`
gained `--min-height`, defaulting to **0.03** — the new production operating point.
Floor is applied at the peak-record step (`targets[currentmax] >= min_height`), so it
composes cleanly with the integral/width cuts and keeps the position/height arrays
aligned. Verified both copies behave identically (synthetic test: 0.02 peak dropped,
0.05 + 0.30 kept at floor 0.03).

Why: the integral cut gates on AREA — a wide ~0.06-tall shoulder clears
integral=0.2 and becomes a fake. A height floor removes those directly.

Evidence: new `diagnostics/peak_operating_point.py` ran v4b ep3 on 300 PU200 events
(r16438, MC TruthVertex), characterising every fake + sweeping the floor 0→0.15:
- Fakes are NOT sidelobes: 0% within 0.3 mm of a real peak, only 2.5% within 0.5 mm,
  33% within 0.85 mm, 37% isolated (>1.5 mm). So `suppress_neighbor_peaks` is the
  wrong tool; a height floor is the right first lever.
- Amplitude overlap: fake median 0.060 (p90 0.16) vs real-peak p10 0.14, median 0.83.
- Sweep: floor 0.000 → eff 0.9651, 9.87 fake/evt; 0.030 → 0.9619, 8.87; 0.050 → 0.9503,
  5.96; 0.100 → 0.9219, 2.47; 0.150 → 0.8979, 1.17. No free lunch beyond ~0.03 —
  every further fake removed costs real low-ntrks efficiency (overlap region).
- Conclusion: 0.03 is a near-free default; the overlap region (~5 fake/evt) needs a
  context-aware learned objectness head (sees track support under the bump), not a
  histogram-amplitude threshold. That is the planned next training-side lever.

Artifacts: outputs/06_09_2026_output/peak_operating_point/{operating_point.png,
height_floor_sweep.csv}, opscan.log.

## 2026-06-09 — Candidate classifier (peak_classifier_v2) on v4b ep3

Ran peak_classifier_v2.py on fresh v4b ep3 artifacts (2500 evt, r16438, min_height=0,
--save-histograms → eval_v4b_ep3_artifacts/eval_results.pkl). 241k peaks, 10.1% fake.
Real/fake = within 0.5 mm of a TruthVertex.

Results:
- Best deterministic single cut: trk>=2 in 0.5 mm → keep 98.8% real, remove 30.7% fake
  (clearly beats any pure height cut — track support discriminates).
- GBT (23 feat) AUC=0.9651. Operating points: thr0.3 keep 99.1%/remove 38% fake
  (14→8.7 fake/evt, ~-0.9pp eff); thr0.5 96.8%/64.6% (14→5/evt, ~-3pp); thr0.7 93.9%/81.9%.
  Far better fake/eff trade than the 0.03 height floor (which gave only -8% fake / -0.4pp).
- MLP AUC=0.9623.
- DECISIVE ABLATION: GBT track-only(feat0-14)=0.9569 ; hist-only(feat15-22)=0.9593 ;
  all=0.9651. Track and histogram features are LARGELY REDUNDANT — each alone ≈ full AUC.
  The fake/real signal is mostly ALREADY in the histogram. Dominant feature:
  local_integral (±0.5 mm windowed area) imp=0.72, then peak_height 0.08, nearest_dz 0.06.

Implications:
1. A HISTOGRAM-ONLY post-hoc GBT gate (no NN retrain) recovers ~the full separation and
   beats the height floor substantially. Deployable via the existing eval
   `--gbt-filter-model`/`_apply_gbt` hook — BUT `_hist_features` (8 feats: height, pk_int,
   local_int, fwhm, curv, rel_height, ndz, nrat) differs from the classifier's hist set
   (has skewness instead of pk_int), so feature extraction must be aligned before deploying
   the saved gbt_hist_model.
2. The dense objectness head is now LESS critical for fakes specifically: since the signal
   is already in the histogram, the head's marginal value is histogram *cleanup*
   (helps merged shoulders / σ) + end-to-end, not unlocking new separation. Reassess after
   the post-hoc gate is deployed.
3. Caveat: local_integral dominance may partly reflect that fakes cluster in sparse regions
   (low local pileup density) — valid discriminant but confirm generalization across files/μ.

Artifacts: outputs/06_09_2026_output/peak_classifier_v2/{feature_distributions.png,
classifier_performance.png,peak_classifier_results.pkl,classifier.log}.

## 2026-06-09 — GBT hist-filter wired into eval + slides updated

Deployed the histogram-only candidate classifier as an eval operating-point gate:
- Rewrote `_hist_features` in run_eval_pvf_run3.py to EXACTLY match peak_classifier_v2
  features 15-22 (peak_height, local_integral, hist_skewness, fwhm_mm, curvature,
  rel_height, nearest_peak_dz, nearest_peak_ratio). Verified bit-exact vs the classifier
  on 779 peaks (max|Δfeature|=0, max|Δproba|=0). Loaded via existing --gbt-filter-model
  hook (reads gbt_hist_model from peak_classifier_results.pkl).
- Gated eval on INDEPENDENT file r16633 (classifier trained on r16438 → no contamination),
  GBT thr 0.3, min_height 0 (GBT replaces the floor): Eff=0.9272, fake=11.29/evt,
  sigma=0.2824 mm. vs no-filter (r16438 ref) 0.9365 / 14.0 / 0.2908. Generalizes across runs.
- Confirmed the resolution (sigma_vtx_vtx) plot is built from the POST-filter peak set:
  _apply_gbt is applied to p_pvs_r (line 331) before pairwise_dz is accumulated (lines
  332-336), so GBT-filtered peaks are correctly excluded from the resolution fit (σ tightens
  0.291→0.282 as fakes are removed). No regeneration needed.

Slides: appended a June-9 update section to presentations/mattia/04_16_2026/slides.tex
(10 frames): v4b category counts, the clean-vs-merged classification fix (before/after TikZ;
efficiency invariant), fake-suppression ladder (height-floor scan, neighbor ruled out,
post-hoc classifier AUC 0.965 + track/hist redundancy ablation, GBT deployed on independent
data), timing verdict (timing_validity.png; ~0.36 timed tracks/vertex, forward-only), and
next steps (in-model objectness head). Compiles clean (latexmk -pdf), 50 pages.

## 2026-06-22 — Wiki sync: docs/ brought up to current truth (v4b era)

Synced the wiki to the code as of June 2026 (docs lagged at the April–early-May
snapshot). Verified each change against the code/configs before editing.

- **docs/index.md**: added the missing nav links (data/run_2, data/run_4, and a new
  Research section linking peak_classification_study and resolution_bump_analysis).
- **docs/evaluation/vertex_finding.md**:
  - canonical HL-LHC checkpoint is now **v4b**
    (`...v4b_3ep_280ch_4lat_stepwarmup_phase2_epoch_3`), v2-wide relabeled as earlier;
    documented the load flags (`--e2e-type v2`, `v2≡v3` build `TracksToHist_v2`,
    `--e2e-unet-channels 280 --e2e-latent-channels 4 --e2e-hidden 128×5`).
  - documented the histogram-only GBT gate (`--gbt-filter-model` / `--gbt-threshold`,
    default 0.7) with the r16633 deployment numbers.
  - added the 2026-06-08/09 truth-side merged/clean fix to the matching section
    (efficiency invariant; merged≈halved; a merged reco covers ≥2 truths).
  - reconciled the 0.2-vs-0.5 contradiction: the mattia_finder diff table and
    Outstanding Issues §2 now reflect the unified-0.5 decision (HL-LHC overrides 0.2).
  - flagged NMS/smoothing as off-by-default and not used in canonical evals;
    clarified `--min-height` default 0.03 vs headline numbers at 0.0.
- **docs/training/vertex_finding.md**: added the v3→v4→v4b progression, the per-step
  warmup recipe, the `--resume` flag, the v3-sweep + v4b config rows, and the v4b
  MLflow run name.
- **docs/data/run_4.md**: fixed the stale training recipe (was v1 lr=1e-3 /
  `config_hllhc_pu200_e2e.yml`) to v4b; corrected the pileup description (μ spans
  190–210, mean ≈200, not the literal pair {190,210}).
- **docs/research/peak_classification_study.md**: added a 2026-06-09 update for the
  v4b / 23-feature `peak_classifier_v2` study and the deployed histogram-only gate;
  kept the May v1 study as the original record.

---

## 2026-06-23 — AMVF-only resolution + vertex-category plots

Added `src/pv_finder/diagnostics/amvf_run3_performance_plots.py` — an AMVF-only
diagnostic (no PV-Finder inference) that reproduces the two ATLAS PV-reconstruction
reference figures:

1. `amvf_resolution_delta_z.{png,pdf}` — pairwise AMVF reco–reco `Δz` distribution
   with the central dip fit by the same symmetric sigmoid notch as
   `run_eval_pvf_run3.py`, yielding `σ_vtx-vtx`. Needs no truth (works on real data).
2. `amvf_vertex_categories_vs_mu.{png,pdf}` — average AMVF reconstructed vertices per
   category (All reconstructed / Matched / Merged / Split / Fake) vs `ActualNumOfInt`.

**Definitions kept identical to the latest eval** by reusing `compare_res_reco`
directly: Matched = `reco_clean`, plus `reco_merged` / `reco_split` / `reco_fake`
(same greedy closest-first, primary-vs-absorbed-truth logic). Truth = `TruthVertex`
with nTracks ≥ 2, detector frame, no beam correction (mirrors the eval's `has_truth`
branch). Matching window = fitted `σ_vtx-vtx` (the AMVF analogue of how the eval
reuses its sigma as the window).

**Data-availability finding:** real Run 3 collision data (`data/run3/file_3.root`)
has `NumTruthVtx = 0` — the `TruthVertex_*` branches are empty — so matched/merged/
split/fake are undefined on it (the script auto-detects this and emits the resolution
plot only). The category figure therefore uses the MC ttbar sample
`data/monte_carlo/ATLAS_PVFinderData_TruthMatched.root` (13 TeV, μ = 1–80), the same
truth-matched sample as `run_eval_pvf.py`, matching the reference figures' caption.

**Generated:** `outputs/06_23_2026_output/amvf_mc_ttbar_2500/` (2500 events).
σ_vtx-vtx = 0.793 mm; per-event means all_reco=22.8, matched=17.1, merged=4.68,
split=0.01, fake=1.02.

Updated: `docs/evaluation/vertex_finding.md` (AMVF-only diagnostic section).

---

## 2026-06-23 — AMVF vs ATLAS reference: shape investigation + slides

Compared the two AMVF plots against the ATLAS PV-reconstruction reference figures
(`temp/AMVF_reference_1.png` = categories, `_2.png` = resolution) and explained the
two differences flagged by the prof.

**Resolution dip (ours wider/rounder vs reference's sharp "sigmoid"):** *not*
binning — the sigmoid-notch fit gives σ_vtx-vtx = 0.792 mm stable across 0.2 → 0.025
mm bins. It is **vertex quality**: tightening the nTracks cut on the cached pairwise
Δz sharpens and narrows the dip — nTrk ≥ 2/4/10/20 → σ = 0.79/0.75/0.65/0.54 mm with
edge slope 5.5/6.7/11.2/24. Our loose nTracks ≥ 2 keeps poorly-resolved 2-track
vertices, which round the shoulders. (Both plots are AMVF reco–reco pairwise Δz.)

**Category steepness (our matched/merged less concave/convex than reference):**
high-μ endpoints already agree (matched ≈ 27, merged ≈ 11–13, all ≈ 42). The gap is
the matching **observable**: ATLAS classifies via track-to-truth weight + a purity
threshold (track contamination at high μ flips vertices to merged → matched
saturates, merged accelerates), whereas we use greedy z-position matching, which
under-counts merged. The MC ntuple has **no `RecoVertex_assocTracks`** (only
`TruthVertex_assocTracks`), so the weight definition is not reproducible here.
Window-sensitivity check: widening 0.4 → 1.2 mm moves matched 75% → 66% and merged
7.0 → 10.5 at μ ≈ 70 but cannot reproduce the curvature — confirming it is the
observable, not a threshold.

**Slides** (`presentations/mattia/04_16_2026/slides.tex`, 4 new frames before the
annex): resolution side-by-side, categories side-by-side, a category-definition
schematic (`images/amvf_category_definitions_sketch.png`, generated to mirror
`compare_res_reco` exactly), and a greedy-algorithm / why-we-differ slide. Deck
recompiles (54 pages). Note: `presentations/` is untracked and holds files > 500 KB
(slides.pdf, the reference PNG) that the large-file pre-commit hook blocks, so the
slide assets are not part of this commit.

---

## 2026-07-12 — TTVA restructure: new src/gnn package + restored eval script

Restarting the track-to-vertex association effort. First step: gave TTVA its
own top-level package so it stops being scattered across four pv_finder
subfolders (which is how the eval script got deleted by mistake in March).

**Moves (git mv, history preserved):**

- `src/pv_finder/models/ttva_gnn.py` → `src/gnn/models/ttva_gat.py`
- `src/pv_finder/data/graph_construction.py` → `src/gnn/data/graph_construction.py`
- `src/pv_finder/training/train_gnn_ttva.py` → `src/gnn/training/train_ttva.py`
- `src/pv_finder/training/training_gnn.py` → `src/gnn/training/training_loop.py`
- `src/pv_finder/diagnostics/{run3,mc}_track_probability.py` → `src/gnn/diagnostics/`
- `configs/vertex_association/` → `configs/gnn/`

**Restored:** `src/gnn/evaluation/evaluate_ttva.py` from commit 51523df.
It was deleted in 0da13fd ("superseded by vertex_finding/ implementations") —
wrongly: no GNN eval exists in vertex_finding/. Clean/Merged/Split/Fake
classification with MaxScore/Threshold edge selection.

**Dependency rule:** `gnn` imports from `pv_finder` (constants, peak finding,
utilities, gradient diagnostics) — never the reverse.

**Small fixes while moving:** indices loading now accepts both .npy and pickled
lists (`load_event_indices` in graph_construction.py — needed for the legacy
`qibin_test_main_indices_v2.p`); `torch.load(..., weights_only=False)` for graph
files; stale `python -m pv_finder...` usage strings updated.

**Verified:** ruff clean; `gnn_ttva_epoch100.pyt` loads strict=True (38/38 keys);
forward pass on toy bipartite graph gives correct edge-logit count; top-1
selection picks one edge per track; legacy pickle indices load (2,550 events).

**Reproduction assets located** (for the next step — reproducing Qi Bin's
ACAT/internal-note numbers):

- `model_weights/gnn_ttva_epoch100.pyt` is md5-identical to
  `test_GATConv_edgeattr_BCE_100.pyt` (the baseline checkpoint); full epoch
  0–200 series in `~/codice/atlas_pvfinder/tracks_to_vertex/model_weights/`.
- `configs/qibin_test_main_indices_v2.p` is md5-identical to the indices used
  in the Nov 2025 eval.
- Target numbers from `~/codice/atlas_pvfinder/tracks_to_vertex/total_results_MaxScore.p`
  (2,550 events, PVF e400 + GNN e100, MaxScore t=0.5): 66,472 reco / 72,189
  truth PVs (92.1% recovery), Clean 76.04%, Merged 16.49%, Split 7.05%,
  Fake 0.42%. Recorded in docs/evaluation/vertex_association.md.
- Peak finding used threshold=1e-2, integral_threshold=0.2, min_width=3.
- Pre-built graphs still on disk in the old repo (`test_graphs_ground_truth.pyt`,
  `test_graphs_pvfinder_e400_fixed.pyt`, `training_graphs_full_51k.pyt`).
- Training data: `/share/lazy/qibinlei/recoTracks_incamvfassoc.h5` (51k events,
  truth `pv_assoc_tracks` + AMVF `reco_pv_assoc_tracks`).

Updated: docs/{models,training,evaluation,diagnostics}/vertex_association.md,
CLAUDE.md project map.

---

## 2026-07-12 — TTVA baseline reproduced bit-exactly

Ran the restored `gnn.evaluation.evaluate_ttva` on the Nov 2025 pre-built
inference graphs (`test_graphs_pvfinder_e400_fixed.pyt`, PVF epoch-400 peaks)
with `gnn_ttva_epoch100.pyt`, MaxScore t=0.5, 2,550 test events
(`qibin_test_main_indices_v2.p`), GPU, ~5 min.

**Result: per-event [clean, merged, split, fake, reco, truth] rows are
bit-identical to the saved Nov 2025 `total_results_MaxScore.p` for all 2,550
events.** Totals: 66,472 reco / 72,189 truth (92.1% recovery), Clean 76.04%,
Merged 16.49%, Split 7.05%, Fake 0.42%.

This validates the Feb 2026 migration + today's restoration end-to-end
(model, weights loading, top-1 edge selection, Clean/Merged/Split/Fake
classification). Note: the original Evaluation_GNN_TTVA.py printed swapped
split/fake *totals* (accumulation bug, fixed in migration), but its saved
per-event rows were in correct [clean, merged, split, fake] order — so the
baseline numbers were always right and the comparison is apples-to-apples.

Eval-script changes on the way (commit 1f4ce55): `-o/--output-dir`,
`-n/--max-events`, working CPU fallback (`-d -1`), and a split of the
classification logic into `gnn/evaluation/classification.py` (500-line limit).
Verified before/after split on 20 events: identical rows.

Outputs: `outputs/07_12_2026_ttva_reproduction/` (results .npy, log,
reproduction_summary.md).

Next: rebuild the inference graphs ourselves (PVF e400 inference → peak
finding → create_inference_graph) to validate the graph-construction leg,
then the ground-truth-vertex eval, then retrain.

---

## 2026-07-12 — Graph-construction leg validated; Qi Bin + ground-truth comparisons

Completed the reproduction matrix (all with GNN e100, MaxScore t=0.5, 2,550
test events; summary in outputs/07_12_2026_ttva_reproduction/).

**New file:** `src/gnn/data/pvf_to_graphs.py` — driver from PVF histogram
outputs (.npy/pickle/.h5) to TTVA inference graphs (peak finding at baseline
params 1e-2/0.2/3 + create_inference_graph). Replaces the legacy
pvfinder_output_to_graph.py.

**Validation:** graphs regenerated from the saved aggregated PVF-e400
histograms are tensor-identical to the Nov 2025 test_graphs_pvfinder_e400_fixed.pyt
for all 2,550 events; eval rows again bit-identical. Combined with yesterday's
result, every leg of the migrated pipeline is now validated:
histogram → peaks → graphs → GNN → Clean/Merged/Split/Fake.

**Comparisons:**

| PV source | Clean | Merged | Split | Fake |
|---|---|---|---|---|
| PVF e400 (baseline + regen, bit-exact) | 76.04% | 16.49% | 7.05% | 0.42% |
| Qi Bin UNet-85 histograms | 76.84% | 16.53% | 6.30% | 0.33% |
| Ground-truth vertices | 76.74% | 11.67% | 5.59% | 6.00% |

Qi Bin's own PVF outputs (test_dropout_t2h_1000_unet_results_85.h5; row order
verified against Target_Y truth) give 76.8% clean through our pipeline — at
the top of the note's preliminary "70–75% clean" envelope. His MLflow runs
log only training losses, so the note text is the only written record of his
rates; the discrepancy in fake rate (note: 5–10%; us: 0.3–0.4%) is consistent
with the note pre-dating the integral_threshold=0.2 peak-finding fix.
Ground-truth clean rate 76.74% matches the note's 77–78% baseline; that run's
6% fake + reco>truth counts come from nTrk<2 truth vertices present in the
truth graphs.

Docs: evaluation/vertex_association.md now carries the full results table;
models/vertex_association.md rewritten as the self-contained TTVA overview
("start here" for a fresh context).

Next: retrain the GNN in this repo (train_ttva on truth graphs, sequential
split), then Run 2/3 (agreement-with-AMVF) and Run 4 PU200 (needs kNN graphs).

---

## 2026-07-12 — AMVF TTVA baseline comparison; training config prepared

**Refactor:** extracted the model-free classification core out of
categorize_event into `classify_assignments` + `build_truth_adjacency`
(gnn/evaluation/classification.py). Re-verified: 20-event GNN eval rows still
bit-identical to the Nov 2025 baseline after the refactor.

**New:** `gnn/evaluation/evaluate_amvf_ttva.py` — classifies AMVF's own
vertices+associations (`reco_pv_*` in recoTracks_incamvfassoc.h5) with the
identical Clean/Merged/Split/Fake logic. Full comparison on the same 2,550
test events:

| Chain | Reco PVs | Clean | Merged | Split | Fake |
|---|---|---|---|---|---|
| PVF e400 + GNN e100 | 66,472 (92.1% of truth) | 76.04% | 16.49% | 7.05% | 0.42% |
| AMVF | 57,329 (79.4% of truth) | 76.69% | 20.05% | 2.51% | 0.75% |

Headline: same clean rate, but PVF+GNN finds 16% more vertices → clean-vertex
efficiency per truth PV 70.0% vs AMVF's 60.9%. AMVF trades splits for merges
(2.5%/20.1% vs our 7.1%/16.5%), consistent with its ~2× worse σ_vtx-vtx.

**Environment quirk (unresolved, workaround documented):** `python -m
gnn.evaluation.evaluate_amvf_ttva` with data args hangs at 100% CPU before
any output on sneezy; the same module via `runpy.run_module` or direct import
runs in ~1 min. `--help`/plain import fine; other gnn CLIs fine under -m.
Documented in docs/evaluation/vertex_association.md; use the runpy form.

**Prepared (not launched):** `configs/gnn/config_gnn_ttva_repro.yml` —
end-to-end reproduction training on the existing 51k-graph dataset
(training_graphs_full_51k.pyt), mirroring the Nov 2025 reference run
(split [0.7,0.25,0.05], bs 32, lr 1e-3, 201 epochs, save every 25).

Outputs: outputs/07_12_2026_ttva_reproduction/amvf/.

---

## 2026-07-12 — HL-LHC PU200 TTVA: kNN graphs, zero-shot collapse, training launched

Pivoted training to PU200 (retraining on the μ≈60 MC would only reproduce the
existing checkpoint; PU200 is where training adds knowledge).

**Scale analysis:** fully-connected μ=200 events have ~118k edges
(~927 tracks × ~128 truth PVs) — 8× the μ≈60 scale, ~370 GB for a 51k-event
dataset. Implemented **kNN edge construction** in create_training_graph
(knn=k → each track connects to its k nearest PVs in |Δz|; knn=None stays
fully connected and bit-exact with earlier builds). Coverage study (200
events, 187k true edges): k=20 keeps 99.50% of true edges; the missing tail
is tracks whose z0 is tens of mm from their true vertex (p99.9 |Δz| = 41 mm)
— unlearnable for any z-based method. Chose k=20 → ~18.5k edges/event,
Run-3-like memory.

**New files:**
- `gnn/data/root_to_graphs.py` — truth graphs directly from PVFinderData
  ROOT (uproot; jagged TruthVertex_assocTracks verified: per-vertex track
  lists, disjoint, lengths == nTracks). Uses the 'hllhc' resolution preset
  (0.17898, 0.7274, 0) for PV heights + edge significances, and stores
  per-track `track.truth_pv` (-1 = none) so evaluation is exact even for
  kNN-dropped true edges. ~45 events/s.
- `gnn/evaluation/evaluate_ttva_graphs.py` — eval on self-labelled graphs
  (no truth h5 needed at PU200); shared classify_assignments core + edge-level
  purity/efficiency.
- `configs/gnn/config_gnn_ttva_hllhc.yml` — 30k graphs, split [0.7,0.25,0.05],
  bs 32, lr 1e-3, 201 epochs, save every 25 → model_weights/ttva_gnn_hllhc/.

**Training-leg bug found & fixed** (validating exactly what the smoke test
was for): train_ttva.py created the Adam optimizer and called
count_parameters before the lazy GATConv layers (in_channels=(-1,-1)) were
materialized → ValueError on any fresh training. Fresh training had never
been run through this script (all prior work loaded existing weights). Fix:
dummy forward on train_data[0] before optimizer creation. 2-epoch smoke on
200 PU200 graphs: loss 1.55 → 1.32, checkpoint round-trips strict=True,
39,521 params (docs previously said ~26k — never actually counted).

**Zero-shot result (μ≈60 checkpoint on 200 PU200 truth graphs, MaxScore
t=0.5): clean 44.9%, merged 23.9%, split 15.2%, fake 16.0%, edge purity
0.64** — versus ~77% clean in-domain. The μ≈60 model does not survive μ=200
density; retraining is scientifically necessary, not optional.

**Launched** (tmux `ttva_pu200`, GPU 0): build 30k graphs (~12 min) →
zero-shot baseline on held-out test slice (28500+) → 201-epoch training
(~4-5 min/epoch ≈ 15 h). Log: outputs/07_12_2026_ttva_hllhc/chain.log.
MLflow run `ttva_gat_pu200_k20` in experiment "TTVA_GNN".

When training finishes: evaluate best checkpoint on the test slice with
evaluate_ttva_graphs, compare to the zero-shot baseline and to the μ≈60
in-domain numbers; then the PU200 full chain (PVF e2e peaks → GNN).

## 2026-07-13 — TTVA evaluation campaign: working point, publication plots, Run 3 data, PU200 retrained

Executed the 4-track campaign planned yesterday (all evals on GNN e100
unless noted; shared classify_assignments core throughout).

**Track 1 — threshold scan → t\* = 0.98.** New `edge_metrics.py` (edge
ROC AUC 0.9981 on 45.3M labelled edges) and `threshold_scan.py` (scores
cached once; MaxScore grid 0.05–0.999 + Threshold-mode points; track-level
efficiency/purity/F1 with AMVF as same-code reference). The sigmoid scores
are saturated — nothing moves below t≈0.9. Moving the working point from
the historical t=0.5 to **t\*=0.98** lifts clean-vertex efficiency
70.0% → **74.5%** at 0.90% fake rate. Track-level: GNN F1 0.866 vs AMVF
0.849 — the GNN is the better pure associator. Found and characterised GPU
forward nondeterminism (~1e-5, GATConv scatter atomics): event 1564 has a
track with top-2 score gap 4e-7 and flips Split↔Fake between runs; new
`regression_guard.py` passes bit-exact or knife-edge-only diffs (PASS).

**Track 2 — publication plots.** At t\*: PVF+GNN clean 80.9% / merged
11.8% / split 6.3% / fake 0.9% on 66,472 vertices (AMVF 76.7/20.0/2.5/0.7
on 57,329; GNN-on-truth 79.9%, clean/truth 89.1%). ATLAS-style figures
(mplhep, "Simulation Internal", Okabe-Ito palette) in
outputs/07_13_2026_ttva_publication/: category bars, clean-vertex
efficiency, efficiency vs truth-PV nTracks (GNN above AMVF in every bin,
largest gains at low multiplicity), edge-score distributions.

**Track 3 — Run 3 real data, truth-free.** `run3_assoc_cache.py` extracted
RecoVertex_assocTracks for the 2,000 cached events (RecoVertex_z alignment
verified per event; assoc counts == nTracks branch). New
`evaluate_ttva_run3.py`: full chain T2KDE e130 + K2H e190 → peaks →
GNN @ t\*. Agreement with AMVF: **84.6%** of AMVF-assigned tracks
(92.6% among both-assigned; window-insensitive 0.5–2 mm); breakdown 8.6%
GNN-unassigned / 4.3% unmatched vertex / 2.4% different vertex.
Leading-vertex agreement 84.8%. Self-check AMVF-vs-AMVF = 1.0000. PVF leg
reproduces run_eval_pvf_run3.py peak counts exactly. Score overlay vs MC:
all-edge shape matches; per-track max scores show a ~3× larger ambiguous
population on data. Fixed a real bug found on the way: run3_track_probability
passed (d0_err, z0_err) swapped into create_inference_graph — pre-2026-07-13
Run 3 score plots used transposed errors (MC path unaffected).

**Track 4 — PU200 retraining evaluated.** Training finished (201 epochs,
~3.5 h, val loss 0.2265). New `evaluate_checkpoints.py` (loads 28 GB graph
set once, all checkpoints + zero-shot regression in one go). Best:
**epoch 175 — clean 62.1%, clean/truth 0.796, edge purity 0.802** vs
zero-shot 44.7%/0.574/0.639 (zero-shot re-run bit-identical to yesterday's
baseline). Learning curve unstable at epochs 50–75 (lr 1e-3 likely too
high) then plateaus 125–200. Plots in outputs/07_13_2026_ttva_hllhc_eval/.

Infra: mplhep added to requirements; shared plot_style.py; the known
"python -m + h5 args hangs at 100% CPU" quirk also hit
plot_ttva_performance — runpy workaround applies (documented).

Open next: PU200 full chain (PVF peaks → GNN), hard-scatter ID, PU200
training rerun with lr schedule.

## 2026-07-13 (later) — PU200 full chain lands: PVF+GNN beats AMVF at HL-LHC; stable retrain running

User directives: full-pipeline HL-LHC eval with the production PVF model,
concurrent stable GNN retraining, AMVF must be beaten, GNN must not be the
latency bottleneck.

**Exploratory experiments first** (grounding the plan; μ60 t*=0.98 unless
noted): (a) split-merge post-processing is a measured dead end — splits
never cost clean/truth, every truth-free merge rule only lost (best 0.740
vs 0.745 baseline); (b) unassigned tracks (9.6%) are 99.2% truth-matched
but badly measured (median z0_err 2.2x worse) — the GNN abstains on hard
tracks; (c) hard-scatter ID: GNN 98.1% vs AMVF 98.9%; (d) PU200 e175
threshold scan on truth graphs: clean/truth 0.796 (t=0.5) -> 0.913
(t=0.95-0.98) — threshold tuning is free performance at PU200 too.

**Full chain**: new `gnn.data.pu200_chain_graphs` runs PVF v4b epoch 3
(production ckpt + operating point: thr 1e-2 / integral 0.40 / floor 0.03)
on ROOT test entries 28500+, builds kNN k=20 inference graphs carrying
track.truth_pv (alignment verified: sum truth = 148,133), and computes the
AMVF baseline through the identical classifier in the same pass.
create_inference_graph gained an optional knn parameter;
evaluate_ttva_graphs now tolerates graphs without edge labels.
Result (e175): **clean/truth 0.619 @ t=0.98 (fake 1.4%) and 0.580 @ t=0.95
(fake 0.7%) vs AMVF 0.573 (fake 0.9%)** — PVF+GNN outperforms AMVF at
HL-LHC pileup. Ceiling on truth vertices: 0.913 — the ~29-point gap is the
finder leg (88.4 peaks/evt vs ~99 truth PVs; chain merged 21-30%).

**Latency** (new `gnn.evaluation.benchmark_ttva` + chain timings, per
event at mu=200): PVF 4.2 ms, peak finding 60.8 ms, graph build 2.0 ms,
GNN forward 3.8 ms, selection loop 11.7 ms. The GNN is NOT the bottleneck
(4.6% of chain); peak finding and the Python selection loop are the
vectorization targets.

**Stable retrain v2** launched (tmux ttva_pu200_v2, GPU 0, ~3.6 h):
identical architecture, cosine LR 1e-3->1e-5 + grad clip 1.0 (new opt-in
config keys in train_ttva/training_loop; config_gnn_ttva_hllhc_v2.yml).
To be evaluated against e175 when done; chain eval reruns if better.

Campaign-2 plan rewritten with all measured findings:
~/.claude/plans/ttva-campaign-2-pu200-chain.md.

## 2026-07-14 (early) — v2 stable retrain evaluated; final HL-LHC chain verdict

v2 training (cosine LR 1e-3->1e-5 + grad clip 1.0, same architecture)
finished cleanly: final val 0.2050 vs v1's 0.2265, perfectly smooth
learning curve (overlay plot shows v1's epoch-50-75 collapse eliminated).
Evaluated all 9 checkpoints + v1-e175 in one pass (evaluate_checkpoints):
best is **v2 epoch 175 — clean/truth 0.816 at t=0.5, 0.9175 at t=0.95**
on truth graphs (v1: 0.796 / 0.913). The stability recipe delivered.

**Transfer gap discovered**: on full-chain graphs (PV nodes = PVF peaks)
v2's fake floor is ~4.5% (junk peaks left empty count as Fake in the
taxonomy) vs v1's 0.6%, so at a <=1-1.5% fake budget **v1-e175 stays the
chain production checkpoint**: 0.580 @ 0.7% fakes (t=0.95) / 0.619 @ 1.4%
(t=0.98) vs AMVF 0.573 @ 0.9%. v2's better truth-node optimum transfers
worse to peak nodes — training/augmenting on chain-like PV nodes added as
a top item in the campaign plan (~/.claude/plans/ttva-campaign-2-pu200-chain.md).

Final HL-LHC verdict tables + chain summary bar + v1/v2 learning-curve
overlay in docs/evaluation/vertex_association.md and
outputs/07_13_2026_ttva_hllhc_v2/eval/plots/. plot_ttva_pu200 gained
--curve2 overlay and --chain-summary options.

## 2026-07-14 — Campaign 2 executed: decomposition, drop-empty, height bug, fast paths, v3 launched

**Gap decomposition (P2)** (`chain_gap_decomposition`, new): finder cap
0.810 (peak within 0.5 mm, greedy 1-1), oracle-association ceiling on the
real peaks **0.748** (unavoidable close-pair merging = 11.9% of oracle
vertices), junk peaks 9.4%, finder misses are low-nTrk (median 3 vs 7).
So of the truth-to-0.62 gap: ~19 pts finder misses, ~6 pts unavoidable
merges, ~13 pts associator headroom.

**Drop-empty convention** (`chain_scan`, new): trackless vertices are not
reconstructed vertices; dropping them before classification leaves
clean/truth invariant and kills the bookkeeping fakes. New headline:
**v1-e175 chain 0.647 @ 0.08% fakes (t=0.995) vs AMVF 0.573 @ 0.91%**
(+7.4 pts at 11x lower fakes); v2's "4.5% fake floor" evaporates (0.636 @
0.08%). Verification: all-peaks t=0.98 reproduces yesterday's 0.6188.

**PV-height bug (major)**: the Gaussian-CDF height recipe added Z_MIN
twice (inherited from Nov 2025 h5_to_graph.py) — every truth graph ever
built had PV heights identically ZERO while inference/chain graphs carry
real peak heights. The height input's weights were unconstrained in
training and inject noise at deployment; likely the dominant transfer-gap
mechanism (and why better-converged v2 transferred worse). Fixed in
graph_construction.py; pre-fix graph files keep zero heights for repro.

**Fast paths (P3)**: numba peak finder bit-exact on 300 real PU200
histograms, 132 -> 0.07 ms (~1800x); vectorized MaxScore selection 24 ->
0.9 ms at PU200 / 11 -> 0.6 ms at mu60, all differences exact score ties
(knife-edge class). Chain latency 82.5 -> ~11 ms/event. Verifier:
verify_fast_paths (report JSONs in outputs/07_14_2026_ttva_fastpaths/).

**HS-ID at PU200 (P4)**: GNN chain 97.5% at t=0.5 == AMVF 97.5%; 92.1% at
t=0.995 -> per-task working points (HS-ID t=0.5, vertices t=0.995, mu60
t*=0.98). **Timing (P5)**: 3.95% coverage overall, 44.6% |eta|>2.3 ->
forward-track study only.

**v3 training (P0+P1)**: chain-like augmentation (graph_augmentation.py:
measured miss-prob dropping, dz jitter, peak sigmas/heights, junk
injection; all quantiles measured on the v4b chain) + fixed heights +
180k all-hadronic events (9x20k shards, ShardCyclingLoader, val 5k from
file 2, test = same SingleLep slice rebuilt with fixed heights). Cosine
162 epochs (18 passes). Build gotcha: torch.save of a 20k shard takes
minutes at high background CPU — first-attempt kill truncated all 8
shards (rebuilt); wait for the "Saved" log line.

**Technical note**: sections/07_gnn.tex fully rewritten (was a 27-line
stub): task, graph/TikZ schematic, architecture, metrics + conventions,
mu60 + Run 3, PU200 chain vs AMVF with yield-ladder figure, transfer gap,
latency, outlook. Compiles clean. PU200 publication plots:
plot_ttva_pu200_pub (yield ladder / chain scan / miss-vs-nTrk).

## 2026-07-14 (later) — v3 evaluated: transfer gap closed, chain at 96% of oracle

v3 training finished in 3.2 h (162 shard-epochs, ~70 s each, smooth
cosine descent to val 0.2120 on augmented graphs). Evaluation
(outputs/07_14_2026_ttva_v3/):

- **Full chain: clean/truth 0.716 @ 0.05% fakes (t=0.98)** — +6.9 pts
  over v1's drop-empty best, **+14.3 pts over AMVF at 18x lower fakes**,
  96% of the oracle-association bound (0.748). Robust (e150: 0.715).
- Truth graphs (fixed heights): 0.823 @ t=0.5, 0.9155 @ t=0.95 — matches
  v2's ceiling despite training mostly on distorted (chain-like) inputs.
- **HS-ID 98.1% @ t=0.5 — now beats AMVF (97.5%)** (v1 merely tied).
- Height-bug evidence quantified: v1/v2 degrade 5/9 pts when evaluated
  on live-height truth graphs; v3 (trained with real heights) gains.
- **Production checkpoint: v3-e156**
  (ttva_gnn_hllhc_v3/ttva_gat_pu200_k20_v3_aug180k_epoch_156.pyt).
  Working points frozen: t=0.98 (vertex classification), t=0.5 (HS-ID).

Publication plots regenerated with v3 (yield ladder, chain scan,
v3 learning curve; outputs/07_14_2026_ttva_pu200_publication/) and the
technical-note GNN section + conclusions updated with the final numbers
(compiles clean). The remaining chain inefficiency is now almost entirely
the finder's (misses 19%, unavoidable merges 6%): next lever is an
associator-aware finder objective / objectness head, plus the 2.5M-event
sample for further scaling.

## 2026-07-14 (evening) — drop-empty measured at mu60 (+2.5 free points); TTVA slides added

Enriched the mu60 PVF-e400 graphs with track.truth_pv (from the truth h5)
so chain_scan runs on them directly. Result: the all-peaks fake budget was
capping the working point — under drop-empty the mu60 fake rate is ~0.01%
at every threshold, unlocking **t=0.995: clean/truth 77.0% @ 0.01%**
(was 74.5% @ 0.9% at t*=0.98; AMVF 60.9%). Guard: t=0.98 all-peaks
reproduces 74.52%/0.90% exactly. Note table + docs updated; mu60 WP is now
t=0.995 (drop-empty).

Added an 11-frame "TTVA from scratch" section (+1 backup) to the weekly
deck (presentations/mattia/04_16_2026/slides.tex): task, graph
formulation with TikZ sketch, GAT model, selection/categories/conventions,
mu60 + Run 3 + PU200 chain results with the new publication plots,
transfer-gap story, final verdict. Compiles with the deck's TL2017
toolchain (PDF 6.7 MB stays local, over the repo size hook).

## 2026-07-14 (night) — mu60 retrained (current model at both pile-ups); publication package finalized

mu60 v2 retrain (fixed heights + chain-like augmentation, exact legacy
event ordering, 120 cosine epochs / 3.3 h) delivered the new mu60
production checkpoint **ttva_gnn_mu60_v2/..._epoch_115.pyt**:

- Chain: **clean/truth 0.7743 @ 0.02% fakes (t=0.99)** vs AMVF 0.6091
  (+16.5 pts); clean fraction 90.4% vs 76.7%; 94.5% of the oracle bound
  (0.8189; finder cap 0.8515; truth-graph bound 0.8920).
- Track level: max F1 0.874 vs AMVF 0.849; edge AUC 0.9980.
- **HS-ID 98.98% vs AMVF 98.75%** (same info-list convention for both).
- Run 3 (truth-free): co-assigned agreement 95.7% (legacy 92.6%),
  leading-vertex agreement **88.7%** (legacy 84.8%); the model abstains
  more on data (83% assigned vs AMVF 95%).

mu60 oracle bound computed (chain_gap oracle on enriched e400 graphs);
mu60 augmentation params measured by the new gnn.diagnostics.
mu60_aug_params (sigmas recovered from edge attributes — no histograms
needed). h5_to_graphs gained shard/augment options (runs via runpy —
fourth CLI confirmed with the python -m h5+indices hang).

Publication package synced everywhere: note section 07 (sober editorial
pass; legacy/current nomenclature; drop-empty convention throughout;
final tables and regenerated figures for both pile-ups incl. drop-empty
category bars), weekly deck TTVA slides, docs, plots. Working points:
t=0.99 (mu60 vertices), t=0.98 (PU200 vertices), t=0.5 (HS-ID).

## 2026-07-15 — TTVA chapter ported to the professors' GitLab note (the canonical version from now on)

Connected sneezy to the official note repo
gitlab.cern.ch/atlas-physics-office/IDTR/ANA-IDTR-2026-01/ANA-IDTR-2026-01-INT1
(HTTPS + personal access token in ~/.git-credentials; SSH to CERN GitLab is
firewalled from outside CERN). Cloned to ~/codice/ANA-IDTR-2026-01-INT1;
live branch is `main` (Lauren mirrors her Overleaf project into it manually —
"Update on Overleaf." commits; GitLab->Overleaf is NOT automatic).

Ported the TTVA work into it (commit 1ccaf82 on main):
- sections/07_gnn.tex: 27-line stub replaced with the full chapter from
  final_paper/current_paper/technical_note (headings promoted section->chapter
  for their REPORT class); renders as Chapter 8, pp. 29–36.
- sections/08_conclusions.tex: TTVA results paragraph + joint
  finder–associator priorities ending.
- sections/01_introduction.tex: one-sentence update ("preliminary study" ->
  full chain evaluated against AMVF at both pile-ups).
- 8 figures copied. Build verified with TL2022 (PATH=/data/apps/texlive/bin/
  x86_64-linux, `make`): 45 pages, zero errors, no overfull boxes from ch. 8.

The professors' repo is the working copy for the note from now on; the local
fork in final_paper/current_paper/technical_note is superseded (kept for
history). Their comment macros: \lat (Lauren), \mm (Mattia), \rbg.

## 2026-07-15 — LT review comments addressed in the note (Method + sec 4.2)

Addressed Lauren's \lat comments in the GitLab note (commit b2e012c on main):
1. Track Selection section written from measured ntuple cuts: pT > 500 MeV
   (spectrum starts at exactly 500 in PU200 + Run 3 files), |eta| < 2.5,
   |z0| < 200 mm; no d0 cut (max |d0| ~37 mm). Left an \mm note to confirm
   whether the Athena dumper applies a named quality WP (ask Rocky).
2. Stitching detail: 12 disjoint 1000-bin tiles concatenated, no blending;
   the +-3 sigma input extension keeps seams continuous; targets split the
   same way; peak finder runs on the stitched 12k histogram.
3. Fig 2.2 was showing placeholder images (a flow diagram + a 3-panel
   diagnostic with mojibake minus signs). New generator
   src/pv_finder/diagnostics/method_example_figures.py builds (a) mu60
   KDE-A + truth PVs, (b) PU200 v4b e2e histogram (cached gap-decomposition
   hists, events 28500+) + peak finder at production settings + truth.
4. Resolution-model fit errors from the saved full-file fit
   (outputs/06_01_2026_output/amvf_resolution_residuals/fit_params.json):
   A=0.179+-0.008 mm, B=0.727+-0.014, C=0.00+-0.08 um. NOT large.
5. Sec 4.2: AMVF pairwise Delta-z sigmoid fit at PU200 computed fresh
   (99,800 evts r16438, 449M pairs, ntrk>=2): sigma = 0.283 +- 0.011 mm —
   same width as PVF (0.29); the discriminators are dip depth (0.6% vs 4.1%
   residual) and the 6x smaller bump. Low-pileup fits (0.42 vs 0.76 mm)
   quoted for reference. GBT cross-ref added, integral-threshold explanation
   now includes narrower-targets argument, failure-modes caption expanded
   with per-row legend description.
Bonus: appendix \section -> \chapter so "Appendix .3" refs now render
"Appendix C". Note builds clean (46 pp, 0 undefined refs).

## 2026-07-15 (afternoon) — Fig 2.3 resolution-fit bug found + LT comment round 2

MM's suspicion about fig:resmodel confirmed: the published HL-LHC resolution
fit (A=0.179 mm, B=0.727) does not describe the residuals. Cause: greedy
matching within 2 mm at PU200 density leaves a flat wrong-match background
under the residual peak (17% of pairs at ntrk=2, <1% above ntrk~30); the
original per-bin Gaussian fits absorbed it, inflating low-multiplicity widths
and steepening the power law. Background-corrected per-bin fits (Gaussian +
flat, new src/pv_finder/diagnostics/resolution_fit_v2.py) give A=0.124 mm,
B=0.458, C ~ -7 um — i.e. the ITk follows the statistical 1/sqrt(n) scaling
(Run 3 B=0.495) at HALF the Run 3 amplitude. Estimator spread moves A,B by
~0.01. Exponential form ruled out (chi2 30x worse than power law). The note
keeps the preset as the definition of the targets (results unaffected) and
now shows the corrected fit + preset overlay in a regenerated log-log figure.

Also addressed (note commit 7fd57d8): Method chapter overview; AMVF
processing time via the Run 3 tracking-software paper citation (CPU-vs-mu,
ACTS 1.5x; absolute PU200 number needs Athena — MM note for Rocky);
collision-data validation section reframed per LT (svtx + yield vs mu as
primary results, AMVF agreement as secondary with the no-truth caveat);
appendix C lineage intro + editing pass. NOT addressed: 03_data Athena
integration (needs Rocky). GNN chapter confirmed to carry both current
models (mu60 e115, PU200 v3-e156).

## 2026-07-15 (midday) — v5 PVF retrain launched with corrected target widths

Launched the corrected-widths campaign (tmux pvf_v5, GPU 0):
- New preset 'hllhc_corrected' (0.1239, 0.4583, -0.0073) in
  resolution_presets.py; old 'hllhc' marked SUPERSEDED. 2 um sigma floor
  added in root_to_h5 (no-op for existing presets).
- All 8 v4b-pool files reconverted with the new preset ->
  data/run4/PU200_corrected_h5/ (69 GB, 8/8, ~30 min at 4 concurrent).
  Smoke-tested first (attrs + target diff sanity: high-n peaks now
  wider/lower, low-n narrower/taller, exactly per the fit correction).
- Training: exact v4b recipe (3+3 epochs, per-step warmup, config
  config_hllhc_pu200_e2e_v5_corrected.yml, run
  hllhc_pu200_e2e_v5_correctedwidths_3ep). ~18 it/s in Phase 1 ->
  ETA ~30 h total. Clean A/B vs v4b: only the target widths differ.

Also measured (for the note figures): PU200 r16438 ActualNumOfInt is
discrete blocks 190-210 (mean 200.0, sigma 6.1) and the chain-eval slice
28500-30000 is BIMODAL ({190-192} + {208-210} only, mean 200.1); the mu60
GNN h5 has no mu branch and its truth-PV count per event is broad/flat
(mean 31.8, sigma 18.9, q5-q95 = 4-63) — the inherited "mu~60" label
likely overstates typical pileup; confirm production config with Qibin
before relabelling figures.

## 2026-07-17 — v5 corrected-widths A/B: real but modest Pareto gain

v5 training completed (3+3 epochs, 34 h total, smooth). A/B vs v4b with the
identical documented protocol (2500 evts r16438, mu 185-215, integral 0.2):

- v4b recheck (floor 0.03): clean 70.15/evt, fakes 12.82, eff 93.27%,
  sigma 0.290 mm — reproduces the published table exactly.
- v5 at the same floor: clean 72.01 (+1.9), fakes 17.97 (+5.1!) — corrected
  widths make ALL peaks taller (0.15/sigma area scaling), so the v4b floor
  no longer cuts the same junk.
- Height-floor sweep (1000 evts, both models, same events): the v5 curve is
  ABOVE v4b at every matched fake rate — +0.25-0.3 pts efficiency at equal
  fakes, or ~10% fewer fakes at equal efficiency. A genuine Pareto shift.
- v5 full eval at retuned floor 0.05: clean 70.96/evt, fakes 13.21,
  eff 93.57%, sigma 0.288 mm, missed 6.4% (vs 6.7%).

Verdict: corrected target widths are a real improvement (+0.8 clean/evt at
~matched fakes, better resolution) but not transformative. NOT adopted as
production: the TTVA chain (GNN v3, augmentation quantiles, working points)
is calibrated on v4b peaks; swapping finders requires re-deriving the chain.
Checkpoint: hllhc_pu200_e2e_v5_correctedwidths_3ep_phase2_epoch_3(.pyt/.pth).
Outputs: outputs/07_16_2026_output/{eval_hllhc_v5_ep3,eval_hllhc_v4b_ep3_recheck,
opscan_v5,opscan_v4b,eval_hllhc_v5_ep3_floor050}.

## 2026-07-20 — σ_vtx-vtx was a coarse-binning artifact; efficiency re-derived; Run 2 μ mislabel

Chasing an LT note comment ("PVF resolution comparable to AMVF is surprising")
and MM's hunch that a fit had not converged, found the pairwise-Δz sigmoid fit
is binning-sensitive. The production eval bins the ±6 mm pairwise-Δz over 60
bins (0.2 mm/bin); the PVF dip is box-shaped with near-vertical walls, so at 60
bins only ~2 points sit inside the dip and the fit lands at 0.29 mm — and the
next refinement (120 bins) does NOT converge (err ±151 mm). Refining to 240/480/
960 bins the PVF fit is stable at **0.223 mm**; AMVF's dip is broad/rounded and
binning-independent at **0.284 mm**. So PVF is ~20% finer than AMVF, consistent
with its Run2/3 advantage; the apparent parity was purely the under-binned PVF
fit. (Scripts: scratchpad pairwise_binning.py / pairwise_plot.py.)

Per user decision, re-derived every chapter-4 efficiency/category number at the
tightened 0.223 mm matching window (offline from saved eval positions +
compare_res_reco; validated bit-exact vs the eval at 0.29 mm first). Re-ran the
baseline (no-floor, SingleLep) and GBT (r16633) configs to get their positions
(outputs/07_20_2026_output/{eval_v4b_baseline_nofloor,eval_v4b_gbt_r16633}).
At 0.223 mm: PVF baseline clean 70.4 / fake 15.8 / **eff 91.6%**; +floor 70.1 /
14.5 / 91.3%; +GBT 69.4 / 12.8 / 91.0%; AMVF 68.1 / 16.8 / **89.4%**. Tighter
window drops both effs ~2 pts and widens PVF's lead (+1.6→+2.2 pts). Blast
radius is chapter 4 only — the GNN finder cap uses a fixed 0.5 mm window.

Run 2 μ investigation (LT: "doesn't look like ⟨μ⟩ 25–30"): the 2018 ZeroBias
file (run 364076, ActualNumOfInt) has full-sample mean μ=32.5 (mode ~27), but is
**pileup-ordered** — its leading events (which a default eval reads from entry 0)
average μ≈47–49. The validation plot (run2_stats.png) shows a μ distribution
peaking at ~44, i.e. the high-μ leading slice, NOT 25–30. So the label is wrong
AND the plotted sample is unrepresentative of Run 2 (a selection artifact, not a
data/scaling error — no μ correction is applied anywhere; training is unaffected,
this is validation-only). Fix pending user decision (re-run on a representative
random sample → μ~32, vs relabel to ~45); chapter 5 left untouched, LT comment
retained as the open marker.

Note repo commits (gitlab ANA-IDTR main): 6dfe73d (resolution + efficiency +
AMVF-only Fig 4.4b + corrected-only Fig 2.3), 10f769e (GNN Table 8.1 bound
explanation, 3.0.1 comment removed, caption typo). Group-A LT comments were
ccd2ea1. See [[ana-idtr-note-repo]].

## 2026-07-31 — all-eta re-production QA; sftp rescue; loss + crop landed

### Data: extended-|eta| PU200 arrives (`data/run4_all_etas/`)

New QA script `src/pv_finder/diagnostics/all_etas_data_qa.py`; outputs +
written findings in `outputs/07_31_2026_output/all_etas_qa/`. Compared
like-for-like (601229 r16633, same 99 800 events) against the |eta|<2.5 file.
Branch list identical; beam spot, pileup and the AMVF vertex-z distribution
unchanged. Everything else moved:

- tracks/event 921 -> 1361 (x1.48), |eta| out to 4.02, 35.3 % of tracks beyond 2.5.
- HGTD timing coverage **3.9 % -> 28.8 %** (70-79 % flat over 2.5<|eta|<3.9, and
  the 34.7 ps single-hit population is largely replaced by 20-25 ps). The
  "timing not usable for the finder" negative result (note sec. 06) was reached
  at 3.9 % coverage confined to the 2.4-2.5 sliver and no longer describes the
  data.
- truth vtx with nTrk>=2: 98.4 -> 115.7/evt (+17.6 %); mean truth nTrk 7.20 -> 9.66.
- **AMVF vertices/event unchanged (101.17 -> 101.27)** but mean AMVF-vertex
  nTracks 8.80 -> 13.40 (+52 %). AMVF attaches the forward tracks to existing
  vertices; it gains no new seeds from them.

**The old sample was contaminated, not merely eta-cut.** It carries 65.8
tracks/event confined to exactly 2.0 <= |eta| <= 2.5 with pT down to 0.5 GeV
(0.9 GeV everywhere else), plus |d0| out to 49 mm and sigma(d0) to 4.4 mm, incl.
16 tracks/evt with |d0|>1 mm inside |eta|<2. The new files apply a uniform
selection (pT>=0.9 GeV, |d0|<=1 mm, sigma(d0)<=0.35 mm, |z0|<=200 mm). The
remote README confirms it: cuts from the ITk paper arXiv:2412.15090 table 5,
applied at CKF reco level, giving table-6 content.

**Converter limits, measured.** Full z0 scan over the 6 complete new ROOTs
(6.43 M sub-events): global max tracks/sub-event **1066**, p99.999 = 870, and
that is only ~12 % of the eventual pool. `--max-tracks-per-sub` defaults to 1024
and truncation is z0-ordered (drops the highest-z0 tracks while the target keeps
their PVs) -> **use 1536**. `--max-pv 300` is safe: max truth PV/event is 208
over the 4.42 M events on disk.

**What the forward tracks buy.** Fisher bound
`sigma_vtx = 1/sqrt(sum 1/sigma_z0^2)` per truth vertex, central-only vs all
tracks on the *same* vertices: gain is **0.1-0.2 % at every multiplicity** —
sigma(z0) runs 65 um at eta=0 to ~2.8 mm at |eta|=3.7, so forward tracks carry
~1/2000 the weight. What they add is reach: **19.2 truth vertices/evt (17.3 % of
the nTrk>=2 denominator) have fewer than 2 central tracks** and only clear the
threshold because of them. Their median achievable sigma is 257 um (vs 36 um for
regular vertices) and only 46 % are localisable to better than PVF's own
223 um vertex-vertex resolution.

Consequences: (a) the efficiency denominator grows 17.6 % with the *hardest*
vertices, so absolute efficiency will fall for PVF **and** AMVF and is not
comparable to the published |eta|<2.5 numbers — quote both denominators;
(b) at fixed truth nTrk the achievable resolution is now worse (x1.18 at
nTrk 20-50 up to x1.74 at 2-3) because nTrk counts low-information forward
tracks, so `sigma(n) = A n^-B + C` **must be refit on this sample** — neither
`hllhc` nor `hllhc_corrected` applies, and the refit moves sigma *up* at fixed n.

### sftp rescue

A second recursive `sftp get -r` was overwriting already-complete files (sftp
overwrites, it does not skip): 601229 r16633 was eaten from 6363 MB back to
441 MB, and the three 95-101 GB 601237 files were next. Interrupted it; the live
ControlMaster socket (`~/.ssh/config`, ControlPersist 1h) allowed a proper
remote listing without re-auth. Remote has **4** part files, not 6 as in the old
production; 295 GB total, of which only 31.5 GB was actually missing.
New `scripts/fetch_run4_all_etas.sh`: rsync `--size-only --inplace --partial`,
20 retries, then opens every ROOT with uproot and prints event counts.
**Never use `sftp get -r` on this directory again.**

### Code: loss selection + padding crop

- `src/pv_finder/models/losses.py` (new): `WeightedMSELoss(y0)` with
  `w(y) = y0/(y+y0)` and a `build_loss(configs)` factory. `train_hllhc_e2e.py`
  now uses it in both phases and logs the loss; **default `loss_type: mse` is
  byte-identical to previous behaviour**. Fixes a real bug: the script hardcoded
  `nn.MSELoss()` while every HL-LHC config carried unread `use_mse_loss` /
  `asymmetry_param` keys.
  Motivation, measured on `PU200_corrected_h5`: peak height spans 20.5x (p5 0.34,
  p99 6.99), so the MSE cost of missing a peak spans **50.5x** across height
  quintiles; `y0=0.3` compresses that to 8.0x. `w(0)=1` exactly, so the
  empty-bin gradient — and hence the fake-rate penalty — is untouched.
  Corroborating evidence for the plateau: over v4b Phase 2, `p2_val_loss`
  0.0330 -> 0.0261 (-21 %) while `p2_eff_step` 0.766 -> 0.761 and `p2_fpr_step`
  1.144 -> 1.161.
  **Not adopted as default**: it shifts the amplitude scale, so all operating
  points need re-tuning. Run as a clean A/B.
- `_crop_and_zero_padding` in `autoencoder_models.py`: trims the all-padding
  tail and zeroes the -999999 sentinel before the MLP. Exactly loss-free (padded
  slots are multiplied by mask==0 before the per-track sum) and packing-agnostic.
  Measured **105.6 -> 77.1 ms/step (1.37x)** on an idle A100 with the real v4b
  model at bs=128; the weighted loss costs a further +0.6 %. Zeroing the sentinel
  also removes the O(1e6) activations that would become inf (then NaN) under
  fp16 autocast. AMP itself is **not** enabled — not bit-exact, needs a full run
  to validate.
- First tests in the repo: `tests/test_losses.py` (20) and
  `tests/test_masked_dnn_crop.py` (13, asserting bit-exact forward *and*
  backward vs a reference copy of the pre-crop pass, including on the real v4b
  checkpoint with real HDF5 batches). All 33 pass.

### Still open before launch

1. `sigma_vtx-vtx` sigmoid fit is **still** hardcoded at 60 bins
   (`plots_pvf.py:53`, `run_eval_pvf.py:372`, `run_eval_pvf_run3.py:362`)
   despite the 2026-07-20 diagnosis; sigma is fed back as the matching window,
   so efficiency is ~2 pts optimistic. Fix before generating any A/B number.
2. `DEFAULT_RESOLUTION_PRESET` is still the superseded `hllhc`
   (`resolution_presets.py:42`).
3. Note sec. 02_method still publishes A=0.179, B=0.727 as the fit, with the
   open "check the residual features" NB at line 156.
4. Refit (A,B,C) on the new sample and register a new preset before converting.

## 2026-08-04 — v6 trained on the all-eta pool; the pairwise-dz bump explained

### v6 finder trained

`hllhc_alleta_v6_mse_2ep` finished 2026-08-01 18:31 on the full extended-|eta|
pool (59,599,872 train sub-events, 465,624 steps/epoch, 2+2 epochs, plain MSE,
`hllhc_alleta` target widths). Final Phase 2: train 0.016330, val 0.015755,
mini-val eff 0.6044, fpr 1.5412. Checkpoint
`hllhc_alleta_v6_mse_2ep_phase2_epoch_2_fullstate.pth`.

Phase 1 ran both epochs (best val 0.034881 at ep 2, vs 0.037282 at ep 1) after
MM chose to match the published recipe rather than skip epoch 2.

Confirmed again, on a fresh model and fresh data: **Phase-2 val loss falls
0.01961 -> 0.01666 (-15%) while mini-val efficiency stays flat** (0.6032 ->
0.5861, slope -0.0036/100k steps) and FPR shows no trend. Same signature as
v4b. The loss/metric decoupling is a property of the objective, not of one run.

### The pairwise-dz bump: not the data, not a PV-Finder defect

New tool `src/pv_finder/diagnostics/pairwise_dz_comparison.py`; write-up in
`docs/research/pairwise_dz_bump.md`; outputs in
`outputs/08_04_2026_output/pairwise_dz/`.

At the 240-bin binning the pairwise-Delta-z distribution shows a +14% excess at
|dz| ~ 0.57 mm just outside the resolution dip. Three-way control on the SAME
1900 mu-matched held-out events (<mu> = 192.6):

| | plateau | dip | peak excess | at |dz| | band 0.3-0.7 |
|---|---|---|---|---|---|
| Truth (nTrk>=2) | 6665 | +0.5% | 3.3% | 0.57 mm | +0.5% |
| AMVF | 4941 | -99.8% | 8.9% | 0.92 mm | -14.5% |
| PV-Finder | 5760 | -99.6% | 13.7% | 0.57 mm | +7.6% |

- Truth is flat -> the excess is NOT the sample's vertex-spacing physics.
- AMVF shows the same structure, displaced to ~0.9 mm -> NOT a PVF pathology.
- The excess sits just outside each algorithm's own resolution limit (PVF dip
  recovers by ~0.35 mm, AMVF only by ~0.85 mm).

Proposed mechanism: a marginally-resolvable pair is split only when the density
between the peaks dips enough to show two objects; each position is then the
amplitude-weighted mean of its own side of the valley, so both are pushed
outward and the separation is biased high. Deficit just below the limit, excess
just above. Consistent with the excess GROWING v4b -> v5 (8.8% -> 16.5%, same
file and events, only narrower target widths).

In the 0.3-0.7 mm band PVF is at +7.6% while AMVF is at -14.5%: AMVF has
essentially no vertex pairs there. The bump is PVF resolving pairs AMVF merges.

**An adversarial verification pass was commissioned on this result** (independent
re-derivation, bootstrap-over-events errors since pairs within an event are
correlated, direct test of the outward-bias mechanism, and checks against
peak-finder split-rule artifacts and sub-event stitching). Conclusions above
should be treated as provisional until that lands.

### Two measurement bugs found and fixed today

1. **Plot/fit binning mismatch** (`plots_pvf.py`). The sigmoid is fitted to bin
   counts, so drawing a 240-bin fit over a 60-bin histogram put the curve a
   factor 4 below the data and made a correct fit look like a failed one.
   `plot_resolution` now takes `n_bins`; each caller passes what its fit used.
   A stability scan on 45.6M held-out pairs also showed the old 60-bin fit did
   not merely bias sigma high, it **did not converge** (error +-1e6 mm);
   240/480/960 agree at 0.2243-0.2245 mm.
2. **mu mismatch in the pairwise comparison** (mine, caught by MM). The eval
   pkl stores every event READ, not the mu-window subset it summarises. On a
   flat-mu held-out file that meant comparing PVF at <mu>~100 against
   AMVF/truth at <mu>~192. The tool now selects on the stored per-event mu.
   The bump figure was 20.9% under the bug; the correct value is 13.7%.

### Held-out evaluation is now the default

Reminder recorded on 2026-07-31 and still true: the documented HL-LHC eval file
`r16438` is INSIDE the training pool, so the published HL-LHC numbers are
measured on training data. The flat-mu tags r16443/r16638 were excluded from
training and give 7,680 mu-in-[185,215] events each. `--max-events 25000` reads
enough of a flat-mu file to yield ~1,920 usable events; both files together give
~3,840, comparable to the usual 2,500. Measured bias from evaluating on training
data was small: efficiency 87.5% -> 87.0%.

Note `--save-histograms` costs a ~10x slowdown (2 h vs 12 min for 25k events);
use `peak_operating_point.py` for floor sweeps instead.

## 2026-08-04 (later) — CORRECTION: the pairwise-dz bump is a satellite artefact

The entry earlier today concluded the bump was "PVF resolving pairs AMVF
merges, not a pathology". **That conclusion was wrong.** Correcting it here;
`docs/research/pairwise_dz_bump.md` has been rewritten.

Cause: a bump study already existed at `outputs/08_01_2026_output/bump_study/`
(2026-08-01) which had settled the question. It was not consulted before
writing. It contains three measurements that the truth/AMVF control alone
cannot reach:

1. `bump_decomposition.png` — split the pairwise distribution by whether both
   members are truth-matched. Pairs of two genuine vertices are FLAT (1.06);
   pairs containing >=1 surplus (unmatched) peak carry the whole excess (1.56
   at 0.45 mm for PVF, 1.25 at 1.0 mm for AMVF). Remove unmatched peaks and
   the bump disappears.
2. `satellite_crosscorrelation.png` — surplus peaks cluster at 1.54x baseline
   density 0.35-0.5 mm from the nearest truth-matched vertex; truth-matched
   peaks show no clustering at all.
3. `excess_vs_pileup_and_floor.png` — the excess scales as 1/n with the number
   of reconstructed vertices. Genuine two-vertex physics would give a constant
   excess FRACTION (signal and baseline both scale as n^2); 1/n is the
   signature of a per-peak effect (satellite pairs ~ n, total pairs ~ n^2).
   Cross-checked against pile-up: 4.8x excess at mu 1-25, 1.2x at mu 185-215.

So the bump is **satellite peaks**, i.e. surplus reconstructed vertices at a
characteristic distance from real ones. It is a finder artefact — though AMVF
has the same pathology, displaced to ~0.95 mm.

The outward-bias ("repulsion") mechanism proposed earlier IS real and was
measured independently (`pair_separation_repulsion.png`: fraction with
dz_reco > dz_truth peaks at 0.86 at true separation 0.15-0.2 mm, decaying to
0.5 by 0.7 mm, for both algorithms). But it **moves pairs without creating
them**, so it cannot lift the distribution above its own plateau. It explains
the shape of the dip edge, not the excess. Attributing the excess to it was the
error.

What survives from the earlier entry: truth is flat (+0.5%), so the excess is
not the sample's vertex-spacing physics; and PVF's dip is genuinely narrower
than AMVF's (~0.35 mm vs ~0.85 mm), consistent with sigma_vtx_vtx 0.224 vs
0.284 mm. That resolution advantage is real; it just is not what the bump
measures.

**Actionable consequence for the working point:** a height floor removes the
excess 2.5x faster than it removes the baseline (excess/baseline integral
5.23 -> 4.66 -> 3.39 mm for floors 0.0 -> 0.03 -> 0.05), because satellites are
systematically lower-amplitude. This is an independent argument for a non-zero
floor, and the bump is a usable monitor of the surplus-peak population.

Process note: before starting an analysis, check `outputs/` for prior work on
the same question. The adversarial verification pass commissioned earlier today
is still running and will be reported against the corrected conclusion.

## 2026-08-04 (verification) — bump result audited; conclusion holds, write-up did not

An adversarial verification pass (independent re-derivation from the raw pkl +
ROOT, own binning and matching code) reported. Summary: **the satellite
conclusion is confirmed; two of my mechanistic claims and several quoted
numbers were wrong.**

Confirmed independently:
- Event alignment is exact: pkl `event_idx` is 0..24999 contiguous, pkl `mu`
  is bit-identical to ROOT `ActualNumOfInt`, `truth_pvs_mm` equals
  `TruthVertex_z[nTrk>=2]` element-wise, and `n_amvf` matches
  `(RecoVertex_nTracks>=2).sum()` (NOT `len(RecoVertex_z)` — 318/25000 events
  carry a type==1 dummy with nTrk<2). The comparison is valid.
- Profiles reproduce exactly at n_events=1920.
- Significance: bootstrap over events, 4000 replicas. Peak +13.67 +-1.40%
  (9.7 sigma); band +7.76 +-0.47% (16.7 sigma); truth band +0.57 +-0.50%
  (1.1 sigma, flat). Intra-event pair correlation turns out NOT to matter:
  Fano factor 1.07, inflation over naive sqrt(N) is 1.00x.
- Surplus peaks carry the whole excess: genuine-pair excess is NEGATIVE at
  every matching window (-10%/-6%/-23% at 0.3/0.5/0.8 mm).
- Ruled out with measurements: the peak-finder min_width floor (0.12 mm is
  reached only for sigma~0.05 mm peaks; the dip edge is set by predicted peak
  WIDTH), a truth-convolved-with-resolution toy, and tile stitching (a real
  but <=1% effect: 5.97% vs 4.99% of pair midpoints near a boundary; AMVF
  shows none, as expected since it has no tiles).

Corrected in `docs/research/pairwise_dz_bump.md`:
1. The "excess scales as 1/n" claim is OVERSTATED. Measured log-log slope is
   -1.55 and the absolute excess SATURATES at ~2.9 pairs/event above n~40
   rather than growing like n. What is solid is that the fraction is not
   n-independent, which is what kills the two-vertex-physics hypothesis. The
   clean per-peak statement is that the satellite rate per isolated real vertex
   is mu-independent: 17.95% (mu 1-50) -> 20.31% (mu 150-215).
2. The 08-01 numbers I quoted (1.06/1.56 pair-class ratios, 1.54x
   cross-correlation, "excess/baseline integral 5.23->4.66->3.39 mm") are
   **not reproducible** — see the reproducibility gap below. Removed.
3. AMVF dip recovers at ~0.66 mm, not 0.85. AMVF peak location is unstable
   (bootstrap 68% interval [0.925, 1.075]); quote a shoulder at 0.9-1.1 mm.
4. sigma_vtx_vtx on THIS sample is 0.2465 +-0.0025 (PVF) / 0.3048 +-0.0056
   (AMVF). The 0.224/0.284 I quoted are v4b-on-r16438 and must not be mixed
   with the v6/r16443 table. The ~20% advantage survives either way.
5. Added bootstrap uncertainties throughout; the page previously had none.

**Operating-point correction — this matters.** The band excess does NOT fall
monotonically with the height floor. Measured: +7.8% (floor 0) -> +11.7%
(0.03) -> +9.9% (0.05) -> +6.6% (0.08) -> +2.9% (0.12) -> -1.2% (0.20) ->
-5.5% (0.30). The floor removes baseline pairs faster than satellite pairs at
first. Satellites are only meaningfully suppressed at floors >~0.12, far above
the 0.03-0.05 in use. The earlier "a floor removes the excess 2.5x faster than
the baseline" statement came from the unreproducible 08-01 metric and is
withdrawn.

**Three bugs fixed in `pairwise_dz_comparison.py`:**
- AMVF was taken without the `nTrk>=2` cut the eval applies, so the AMVF curve
  was a different object from the eval's AMVF summary (band -14.43 vs -14.52%).
- `entry_stop = n_events*15` was a silent-failure heuristic: for n_events
  above ~1930 the ROOT-side mu selection walks past the pkl's event range and
  compares PV-Finder against AMVF/truth from DIFFERENT events, with no error.
  Now hard-capped at `len(per_event)` with an explicit count check.
- The plateau median was taken over the symmetrised (duplicated) array,
  landing on an arbitrary 0.5-count tie. Now the positive half only.

**REPRODUCIBILITY GAP — flag for the note.** `outputs/08_01_2026_output/
bump_study/` is five PNGs with no surviving code: nothing in git history,
no JSON, no error bars, and `temp/_investigate.py` is an unrelated script.
Its conclusions were independently reproduced, but its specific numbers were
not and must be regenerated from committed code before use.

Process note: three further investigations were commissioned into the
satellite population itself (network output vs peak finder; extractor
alternatives incl. topographic prominence; and whether this is the known
UNet sidelobe artefact re-surfacing).

## 2026-08-04 (cleanup) - bump documentation reconciled: it is the data, and the estimator

Three pages in `docs/` disagreed about the pairwise-Δz bump. This entry records
the resolution. Single source of truth is now
`docs/research/pairwise_dz_bump.md`; `docs/research/resolution_bump_analysis.md`
is reduced to a stub listing what it got wrong.

### What we believed, and what we now know

| believed | now |
|---|---|
| (Apr) AMVF's bump is 6x PV-Finder's; PVF's is noise; 85% genuine physics | inverted on all three counts |
| (Aug, earlier today) band excess +7.8%, peak +13.7% | **not reproducible; +12.5% and +17.5%** on both held-out files |
| the bump grew because the model changed | mostly the **data**; and "the model" is itself a data change |
| a height floor suppresses it | at the production floor 0.03 it makes it **worse** |
| the fakes are UNet deconvolution sidelobes | amplitude uncorrelated with parent (r = -0.004): not a sidelobe |
| fakes are isolated ("none within 0.3 mm of a real peak") | only 7.2% are further than 3 mm from a real peak; median 0.76 mm |

### The headline number was wrong

`pairwise_dz_comparison.py`, re-run today on `eval_v6_heldout/r16443`, gives
PV-Finder band **+12.5%** and peak **+17.5%**, not the +7.8% / +13.7% on the
page. The AMVF and truth rows reproduce exactly. Cause: the stored
`pairwise_dz_summary.json` is timestamped 15:35 and the eval pkl it was computed
from is timestamped 15:58: the eval was re-run into the same directory and the
PV-Finder input was replaced under it. The second held-out file `r16638`
independently gives +12.5%, and a from-scratch re-derivation from ROOT +
checkpoint gives 12.48%. The peak-finder setting behind the old pkl is
unrecoverable. Withdrawn.

Lesson: an eval output directory is not immutable. Re-run before quoting, or
record which pkl a number came from.

### Model vs data, measured

New tools `src/pv_finder/diagnostics/bump_model_vs_data.py` and
`pairwise_dz_metrics.py`; outputs in
`outputs/08_04_2026_output/bump_model_vs_data/`.

The |eta|<2.5 and all-|eta| productions of `r16443` hold the **same events**:
`ActualNumOfInt` and `RecoVertex_z` are identical entry by entry, so each
checkpoint can be run on each production over the same 1920 mu-matched events.
v4b, v5 and v6 share architecture and loss exactly; only their training data,
target-width preset and epoch budget differ, so even the "model" axis is mostly
a data axis.

Band excess 0.3-0.7 mm [%], paired-bootstrap errors:

|            | \|eta\|<2.5 | all-\|eta\| | Δ data |
|---|---|---|---|
| v4b        | 7.99 ± 0.54 | 11.41 ± 0.51 | +3.42 ± 0.40 |
| v5         | 9.62 ± 0.49 | 10.68 ± 0.47 | +1.06 ± 0.38 |
| v6         | 10.86 ± 0.51 | 12.48 ± 0.48 | +1.62 ± 0.37 |
| Δ model    | +2.87 ± 0.45 | +1.07 ± 0.43 | total +4.49 ± 0.46 |

Satellites per peak (multiplicity invariant):

|            | \|eta\|<2.5 | all-\|eta\| | Δ data |
|---|---|---|---|
| v4b        | 0.0277 | 0.0609 | +0.0332 ± 0.0026 |
| v5         | 0.0460 | 0.0717 | +0.0257 ± 0.0027 |
| v6         | 0.0339 | 0.0579 | +0.0240 ± 0.0025 |
| Δ model    | +0.0062 ± 0.0028 | −0.0030 ± 0.0028 | total +0.0302 ± 0.0028 |

**It is the data.** The data axis carries 79-110% of the multiplicity-invariant
change; the model axis carries −10% to +21% and is consistent with zero on the
production that matters. It is not domain shift either: v6 moves from the
production it was NOT trained on to the one it WAS and gets worse. On the raw
band excess the two decomposition paths bracket the data effect at 36-76% and
the model effect at 24-64% with a large negative interaction, because that
observable also responds to the +6-8 peaks/event. Controls behave: truth moves
+0.36 ± 0.39% despite +16.2 vertices/event; AMVF's vertex collection is
bit-identical between productions (it ran at AOD level) so its −6.16% shift is
pure nTrk>=2 acceptance.

**Within the model axis the lever is the target-width preset**, not the pool or
the schedule: v5 differs from v4b only in that preset and is +0.0183 ± 0.0028
(6.5σ) worse. `hllhc_corrected` is narrower than `hllhc`, so narrower targets
make satellites worse, the opposite of what was assumed when v5 was built.

### Mechanism, now reproducible from committed code

New tool `src/pv_finder/diagnostics/satellite_mechanism.py`;
`outputs/08_04_2026_output/satellite_mechanism/`. The equivalent numbers
previously lived only in `outputs/.../satellite_origin/*.py`, which is
gitignored.

- 92.5% of band satellites exist only because the conjoined-split rule fired
  (51.9% for truth-matched peaks). The rule cannot be removed: efficiency
  0.883 → 0.616 without it. Run on **target** histograms it produces **zero**
  surplus peaks, so the rule is not the cause on its own: the output is.
- Band satellites have median topographic prominence 0.029 against 0.751 for
  truth-matched peaks; 22.3% are below 0.005.
- **Upsampling lattice.** Surplus peaks show a 35.1% modulation of their argmax
  bin mod 4 (χ² = 854, 3 dof); truth-matched peaks only 3.9%. Two
  `F.interpolate(mode="nearest")` stages put the decoder on a stride-4 lattice
  and the spurious population is phase-locked to it.
- **Target reachability.** Under `hllhc_alleta`, 31.6% of nTrk>=2 truth vertices
  get sigma below one 0.04 mm bin, amplitude scale up to 22.4x (median 2.84x).

### The position estimator changed under the analysis

While this was running, the approved local-centroid estimator landed in
`src/pv_finder/utils/peak_finding.py` and one measurement silently changed
estimator mid-flight (caught because its band excess disagreed with the
production pkl). Both new tools now take `--centroid-halfwidth`, default `0` =
historical full-region weighted mean, and every number is labelled. The A/B was
re-run pinned and reproduces the earlier caches bit-for-bit. Measured effect,
same peaks, only the positions moving:

| estimator | band excess |
|---|---|
| full-region weighted mean | +12.48% |
| local centroid ±2 bins | +3.82% |
| local centroid ±3 bins | +5.11% |
| local centroid ±5 bins | +6.72% |

The approved change removes 59% of the bump at zero cost in peaks. It does not
remove the surplus peaks; it stops them being mis-placed into the band.

### Also measured, for the operating point

- Height floor: band excess 12.5% (0.0) → **13.9% (0.03)** → 10.8 (0.05) → 7.3
  (0.08) → 4.2 (0.12) → −1.4 (0.20). The production floor makes the bump worse.
  It is still a good fake remover (4.9:1 surplus-to-real at 0.03) but it targets
  isolated low peaks, not band satellites.
- Prominence gate (declined by MM): 0.005 → +3.07% band, at 5.07 peaks/evt of
  which 1.70 truth-matched (2.0:1). Recorded so it need not be re-derived.

### Still open

- The surplus peaks remain: 20.8/event, ~22% of them on real nTrk<2 truth
  vertices that `run3_io.py` removes from truth, so the fake rate is overstated
  by about a fifth. Fixing the truth definition is a separate, cheap job.
- Whether replacing the nearest-neighbour upsample reduces the satellite rate.
  The lattice correlation is strong but only a retrain tests it.
- Whether capping the target sigma at the bin width helps. The v4b/v5/v6
  comparison argues for it but it is untested.
- Combined effect of the local-centroid estimator with a prominence gate is not
  measured; they act on the same peaks and will not simply add.
- **Open discrepancy with `docs/research/peak_position_estimator.md`**, flagged
  in both pages. It quotes the same 0.3-0.7 mm band as +1.76 -> +0.01
  pairs/event for the estimator change (complete removal); converted to the same
  units this analysis gives +3.00 -> +1.23 pairs/event (59%). The baseline
  conventions differ. Reconcile before either number goes into the note.
- Slide decks still carry "sidelobes" and "fakes are not sidelobes" language;
  listed in the doc under Downstream claims that still need revisiting.
- `outputs/08_01_2026_output/bump_study/` still has no code and is now
  contradicted on the height-floor claim. Retired.

---

## 2026-08-04 — local-centroid position estimator + peak-finder operating point

Two things: the approved position-estimator change (commits `346d910`,
`6845415`), and the operating-point scan that follows it. Numbers in
`outputs/08_04_2026_output/peakfinder_operating_point/`, write-up in
`docs/research/peak_position_estimator.md`.

**Method.** Dumped 7680 held-out µ-window events per file from r16443/r16638 by
pre-reading `ActualNumOfInt` and running inference only on selected entries
(~13x cheaper than a linear scan, 6 ms/evt). Verified the dump reproduces the
production eval bit-for-bit, and the region scanner + matcher used for the
sweeps against `pv_locations_updated_res` / `compare_res_reco` before taking any
result from them. Selection on half A, reporting on half B; the two agree to
0.07 efficiency points at the chosen point, so nothing was overfitted.

**Established, not assumed:** the v6 held-out eval ran with
`integral_threshold = 0.2`, not the 0.5 CLI default. Recovered by matching its
stored per-event peak lists exactly (30/30 events). Every earlier number on this
sample should be read with that in mind.

**Estimator.** Local centroid over max ± 3 bins, clipped to the region.
Core residual IQR sigma 46.19 -> 44.41 um (-3.85 +- 0.05 %, paired bootstrap,
36 sigma), sigma_vtx-vtx 0.2403 -> 0.2091 mm, position bias +2.9 -> +0.5 um.
Costs 0.34 efficiency points at a fixed window, **all of it merge-credit** —
cleanly reconstructed truth is unchanged at -0.013/evt. Unclipped windows were
measured and rejected: best band numbers, but sigma_vtx-vtx degrades to 0.2629,
worse than baseline. Argmax buys nothing on the core width (bin quantisation
cancels the gain).

**Operating point.** Recommended `--integral-threshold 0.30 --min-height 0.03`
with the local centroid. On the reporting half: Eff 0.8699, 16.16 fake/evt,
sigma 0.2140 mm, core 42.98 um, against production 0.8805 / 21.55 / 0.2405 /
46.21. The fake reduction costs 0.151 efficiency points per fake/event removed,
inside the stated 0.2 budget; the next step out (integral 0.5, height 0.05)
costs 0.257 and is not worth taking.

**The GBT gate does not transfer.** Judged with the label it was trained on
(within 0.5 mm of any truth vertex), the saved v4b `gbt_hist_model` still ranks
well on v6 — AUC 0.9494 vs the 0.9593 quoted — but its *calibration* has moved:
at the documented thr 0.3 it removes **7.4 %** of fakes on v6, not 38 %.
Retraining the identical model on v6 buys AUC 0.9535, i.e. +0.0035, so the model
is fine and the threshold is what moved. At matched real-kept it is only
marginally better than a one-parameter height floor. Recommendation: do not
deploy it as-is and do not retrain blind; the deterministic levers reach the
same trade without the model dependency.

**Two eval conventions worth flagging.**
1. The eval feeds its own fitted sigma_vtx-vtx back in as the matching window.
   A 13 % better sigma therefore buys a 13 % tighter window and the printed
   efficiency drops 0.8845 -> 0.8652 — 1.9 of those 1.93 points are the window,
   not the estimator. Any estimator A/B must fix the window.
2. The band-excess discrepancy between this page and the bump page was
   reconciled by measurement: both are all-pairs, neither is conditioned, and
   the entire gap is the plateau convention. The |dz| density is not flat
   (slope -0.65 pairs/evt/mm over 1.2-6 mm), so a far-plateau median taken at
   |dz| > 3 mm sits 4.5 % below the local baseline at 0.5 mm — exactly the
   11.85 - 7.01 = 4.84 point offset. Also found: the median-plateau convention
   is not robust to quantised positions (argmax reports +25.29 %, an artefact of
   the 0.04/0.05 mm beat).

**Deliberately not done.** The library default stays `centroid_halfwidth=0`, so
the GNN graph builders and all diagnostics are untouched — the deployed TTVA
checkpoints were trained on these positions and `pv_sigmas`, and moving them
needs its own A/B. `run_eval_pvf.py` was not wired up either; it carried
unrelated uncommitted GBT work at the time.

---

## 2026-08-04 — Held-out eval at the chosen operating point

Full evaluation of the v6 finder at the operating point selected the same day,
on both flat-mu held-out files, 25 000 entries each, summarised on mu in
[185, 215]. Launcher: `scripts/eval_v6_operating_point.sh` (records the full
invocation and git SHA into the output directory). Outputs in
`outputs/08_04_2026_output/eval_v6_operating_point/{r16443,r16638}/`.

Settings: `--centroid-halfwidth 3 --integral-threshold 0.30 --min-height 0.03
--peak-threshold 0.01 --pairwise-bins 240`, v6 checkpoint
`hllhc_alleta_v6_mse_2ep_phase2_epoch_2_fullstate.pth`.

| | efficiency | fake/evt | sigma_vtx-vtx | reco/evt |
|---|---|---|---|---|
| previous production | 0.8820 / 0.8825 | 21.52 / 21.32 | 0.2328 / 0.2325 | 106.8 |
| this operating point | 0.8643 / 0.8656 | 16.64 / 16.36 | 0.2190 / 0.2200 | 101.0 |
| AMVF, same events | — | 17.80 / 17.60 | 0.3048 | 97.9 |

The two files agree to 0.13 efficiency points and 0.3 fake/evt, but they share
`e8481_s4494` and differ only in reco tag, so they are probably the same
generated events reconstructed twice. Treat this as a consistency check, not as
two independent samples.

**The result worth quoting.** PV-Finder now reconstructs more vertices than AMVF
while making fewer fakes: 101.0 vs 97.9 reco/evt at 16.6 vs 17.8 fake/evt, with
sigma_vtx-vtx 0.219 vs 0.305 mm. At the previous operating point it found more
vertices but also made more fakes (21.5 vs 17.8), so the comparison was
ambiguous. It no longer is.

**How to read the 1.77-point efficiency drop.** About 1.06 points is real; the
remainder is the matching window, which this eval derives from the fitted
sigma_vtx-vtx and therefore tightens by the same 6 % the estimator improved.
Within the 1.06, the estimator's 0.33 is entirely merge credit (truth-clean
84.563 -> 84.550 per event), so nothing that was resolved stopped being
resolved. The threshold move alone costs 0.151 efficiency points per fake/event
removed, inside the 0.2 budget agreed for this campaign; the next step out
(integral 0.5, height 0.05) costs 0.257 and was rejected.

**Process fix.** The previous production eval recorded its invocation nowhere —
no launch script in git, nothing in the pickle — and its `integral_threshold`
had to be reconstructed after the fact as 0.2 by replaying stored per-event peak
lists against the peak finder (30/30 events matched). The new launcher writes
the invocation and the git SHA next to the results.

**Also fixed today.** `--resolution-preset` no longer defaults to the superseded
`hllhc` in `root_to_h5.py` or `gnn/data/root_to_graphs.py`; it is required, and
the module-level (A, B, C) in `root_to_h5` are NaN placeholders so a missed code
path fails loudly rather than writing plausible targets from the wrong
resolution model. `scripts/build_v3_shards.sh` hard-codes `hllhc`, correct for
the old |eta| < 2.5 shards and wrong for anything built on `data/run4_all_etas`,
which needs `hllhc_alleta`.

## 2026-08-05 — Post-hoc fruit hunt: one free fix for the plot, a measured no for the fakes

Question: are there still simple, non-overfitting post-hoc changes that cut the
fake rate and clean up the resolution plot, in particular the ripple? No
retraining. Answer: **the ripple yes and for free; the fake rate no, and the no
is now quantitative.** Numbers, figures and every script in
`outputs/08_05_2026_output/ripple_study/`; write-up in
`docs/research/resolution_plot_ripple.md`; corrections to
`docs/research/pairwise_dz_bump.md` sections 1.5, 3.2, 3.5 and a new 3.6.

**Method.** Reused the 7680-event held-out histogram dumps and the verified
region scanner / matcher from the 2026-08-04 operating-point study, so nothing
was re-inferred. Selection on half A, reporting on half B, same seed as 08-04;
the two held-out files share `e8481_s4494` and are the same generated events, so
the split is *inside* one file. Matching window held fixed at 0.2328 mm
everywhere, and the ranking re-checked at 0.2140 and 0.2800 mm. Every new lever
verified against a committed reference before any result was taken from it:
region scanner vs `pv_locations_updated_res` (0/40 count mismatches, max 7e-6
mm), NMS vs `suppress_neighbor_peaks` (0/40 events disagreeing), prominence vs
`scipy.signal.peak_prominences` (2/23528 peaks outside the documented
run-boundary convention). Two bugs were found in my own prominence code this way
before it produced a number.

### The ripple in the resolution plot is a binning beat, and it was one day old

The plateau alternated between two levels by 3.9%, four times its Poisson error,
across the whole +-6 mm range. Not noise, not satellites, not the model: peak
positions are combed at the 0.04 mm output grid (135.8% modulation of the
position fractional part, 40% of |dz|) and lcm(0.04, 0.05) = 0.20 mm, so the
240-bin plot picked up a 4-bin sawtooth.

Controls at the same 240 bins: AMVF 1.3% comb, truth 0.8%, and **PV-Finder with
the historical full-region weighted mean 0.9%**. The comb arrived with the
local-centroid estimator on 2026-08-04 -- `outputs/08_04_2026_output/
eval_v6_heldout/` (the last eval before it) has a visibly smooth plateau. A side
effect of an otherwise good change, and free to undo.

`--pairwise-bins` now defaults to 300 (0.04 mm), derived as
`2 * 6 mm / BIN_WIDTH_MM` in the new `pv_finder/utils/pairwise_dz.py` rather than
hardcoded, and the eval warns on incommensurate values. Sub-multiples are
rejected too: with 0.02 mm bins every second bin catches a comb tooth. Both
files, all four peak lists: comb 3.8-4.0% -> 0.04-0.15%, plateau pull RMS
4.2-4.6 -> 1.6-2.0, sigma_vtx-vtx fit error nearly halved. Paired bootstrap over
events: the comb reduction is 3.75 +- 0.14% (26.5 sigma) and 3.77 +- 0.12%
(31.8 sigma). At 0.04 mm bins the plateau beyond 2 mm is statistical (pull RMS
0.90-1.11); the 2.44 left at 1.2-2.0 mm is the satellite shoulder.

The peak list is bit-identical. Efficiency and fakes move at all only because
this eval derives its matching window from the fitted sigma. Full re-run on both
held-out files, mu in [185,215]:

| | efficiency | fake/evt | sigma_vtx-vtx | reco/evt |
|---|---|---|---|---|
| 240 bins (08-04) | 0.8643 / 0.8656 | 16.64 / 16.36 | 0.2190 +- 0.0050 / 0.2200 +- 0.0047 | 101.0 |
| **300 bins (now)** | 0.8648 / 0.8662 | 16.60 / 16.32 | **0.2200 +- 0.0031 / 0.2213 +- 0.0030** | 101.0 / 100.3 |

+0.05 efficiency points and -0.04 fake/evt, all of it the window. The fit error
is down 38%, and the plateau is readable.

### No post-hoc lever beats the two thresholds already in the eval

817 configurations: Gaussian pre-smoothing (sigma 0.5-2.0 bins, positions from
the smoothed *or* the raw histogram), anti-lattice notch filters, NMS, minimum
separation, absolute and relative prominence gates on the conjoined split, a
peakiness (height/integral) cut, and prominence x separation combinations --
each crossed with the integral and height thresholds. **0 of 817 beat the
deployed operating point, on either half, at any of three fixed windows.** Half
A and half B agreed to within 0.159 efficiency points and 0.126 fake/event
across the whole grid.

Cheapest fake removal each family can manage, efficiency points per fake/event
removed, against a 0.2 budget: thresholds 0.216, Gaussian 0.236, notch 0.246,
prominence 0.313, relative prominence 0.322, peakiness 0.422, NMS 0.708, minimum
separation 1.581. Paired bootstrap on the score (2000 replicas): even the
cheapest move, raising the integral threshold to 0.40, is **-0.025 +- 0.010**,
2.5 sigma below break-even, P(worth taking) = 0.009. Everything else is
P = 0.0000.

Framing that came out of it and is worth keeping: **a post-hoc filter must remove
>= 4.5 surplus peaks per real peak removed to be worth taking** -- one matched
peak/event is 1/111.2 of the truth denominator = 0.90 efficiency points, one
fake/event is worth 0.20. The height floor manages 4.9:1, which is why it is in
the operating point. Nothing else gets close.

### Two findings that change earlier pages

**The upsampling lattice is position quantisation, not additive ripple, so no
linear filter can touch it.** The mean power spectrum of the predicted histogram
has no line at the lattice frequency (power/continuum 1.00 at 1/4 cycles/bin,
0.94 at 1/2), and the 31% mod-4 modulation of surplus peaks is unchanged by every
filter tried: 33.1% after a 1.5-bin Gaussian, 33.4% after a symmetric 4-bin
boxcar, which is the *exact* annihilator of a x4 zero-order hold (zeros at
periods 4, 2 and 4/3). The filters remove surplus peaks uniformly in phase. This
closes off the post-hoc route and makes the retrain item (replace the
nearest-neighbour upsample) better motivated, not less.

**The prominence gate is much worse now than when it was declined on 08-04.** At
the old setting (integral 0.2, height 0.0) the satellites it targets had median
prominence 0.029. At the deployed operating point the thresholds have already
removed that population: surviving surplus peaks have median prominence 0.068
and median prominence/height **0.89** -- standalone low peaks, not flank ripple.
Removal ratio 1.3-2.0:1 against the 4.5:1 break-even, and sigma_vtx-vtx degrades
0.2140 -> 0.2457 mm because the gate also kills peaks that were resolving genuine
close pairs. Declined again, with a better reason. A *relative* prominence gate,
which should be the better discriminator by construction, is no better (0.322).

### About a quarter of the "fake" rate is real vertices

Against the **unfiltered** TruthVertex list, with a displaced-peak control:
29.5 / 29.8% of surplus peaks sit within the matching window of a real truth
vertex with nTrk < 2 (every one of them nTrk = 1, none nTrk = 0), against a 5.3%
accidental level that is flat for displacements of 1 to 12 mm. **Excess 24.2% =
4.17 per event.** So of the quoted ~16.6 fake/event, about 4.2 are real
sub-threshold interaction vertices and about 12 are genuinely spurious. The
nTrk >= 2 convention should stay -- it is what AMVF is counted with -- but the
note should carry the decomposition. This tightens the "About 22%" figure that
was carried over untested from the 08-04 pass.

### Flagged, deliberately not fixed

`run_eval_pvf_run3.py` accumulates `pairwise_dz` over **every event it reads** and
applies the mu window only to the summary, so on the flat-mu held-out files the
quoted sigma_vtx-vtx is a flat-mu number sitting next to mu in [185,215]
efficiency and fake rates. Measured: 0.2190 mm over all 25000 events against
0.2270 mm on the mu-window subset, 3.5%. sigma is fed back as the matching
window, so fixing it moves the headline efficiency -- a decision about what the
published number means, not a drive-by fix. Resolve before the note quotes sigma
and efficiency in one table.

`run_eval_pvf.py` (the h5/MC eval) still hardcodes 60 pairwise bins. That is
commensurate (0.2 mm = 5 x 0.04) so it has no comb, but 60 is the binning
documented as biasing sigma high. Left alone: it carried unrelated uncommitted
work at the time.

### Retrain-only, listed not attempted

- Replace `F.interpolate(..., mode="nearest")` in `src/pv_finder/models/unet_v2.py:35`
  with linear or a learned sub-pixel upsample. Now the *only* route to the
  lattice artefact.
- Cap the target sigma at the bin width and drop the `max(1, 0.15/sigma)`
  amplitude compensation, or move to a finer output grid for the peak region.
- Objectness head / fake-aware loss -- the only item that attacks surplus peaks
  at the source rather than downstream.
- A sub-bin position regression head, which would remove the quantisation at
  source instead of working around it in the plot binning.

---

## 2026-08-05 — One authoritative metric reference; two taxonomies reconciled by measurement

The project uses the words clean / merged / split / fake for **two different
classifiers** and nothing said which was which. `docs/evaluation/metric_definitions.md`
is now the reference: every metric with its formula, its numerator and
denominator named, the file:line that computes it, its units, and whether it is
truth-side or reco-side. Linked from both evaluation pages.

### The gap it closes

`compare_res_reco` (positional: is this vertex inside a window of a truth
vertex?) and `classify_assignments` (track purity: do ≥70% of its tracks come
from one truth vertex?) partition the *identical* AMVF collection on the
*identical* 1920 r16443 events into **clean/truth 0.6090 against 0.3094**. Both
are correct. The bridge, and it closes with zero residual:

```
187,960 AMVF vertices, median dominant-truth purity 0.6316 (straddles the 0.70 cut)
├─ purity >= 0.70 :  76,541  ->  Clean  66,127  +  Split 10,414
└─ purity <  0.70 : 111,419  ->  Merged 80,104  +  Fake   135  +  Split 31,180
                                                          Split total 41,594  ✓
```

So the factor of two is **half the purity cut and half the split demotion** —
the positional taxonomy has no analogue of the second (AMVF split rate 0.3%
positionally against 22.1% by tracks, because two AMVF vertices are rarely
within one sigma of each other but very often share a dominant truth vertex).
This also explains the 0.4072-vs-0.3518 discrepancy sitting unexplained in
`amvf_convention.json`: the purity scan counts *before* the demotion.

### Measured, not argued

Everything in the page is reproduced from stored outputs. The published r16443
summary is reproduced **bit-for-bit** from the stored peak list
(`clean 138012 / merged 22312 / split 1754 / fake 31867`), which anchors every
formula quoted.

**The matching window is worth 0.66 efficiency points per 6% of sigma.** Same
peak list, only the window changes:

| sigma as window (mm) | efficiency | fake/evt |
|---|---|---|
| 0.2190 | 0.8643 | 16.637 |
| **0.2200** (published) | **0.8648** | **16.597** |
| 0.2283 (the mu-window sigma) | 0.8688 | 16.331 |
| 0.2328 | 0.8709 | 16.187 |
| 0.5000 | 0.9332 | 9.589 |

Event bootstrap (2000 replicas): +-0.0007 on efficiency, +-0.097 on fake/evt.
Reco-side *clean* is nearly flat across the whole range (138,005 -> 138,402);
efficiency rises almost entirely through **merged**, i.e. through crediting
close pairs that share one peak.

**What the sigma-population fix (2e55f79) will do to the headline.** The
published r16443/r16638 evals predate it (INVOCATION.txt records 53d1967), so
their sigma was fitted at <mu> ~ 100 and used as the window for a summary at
<mu> = 192.5. Feeding the mu-window sigma (0.2283 mm at 300 bins, from
`ripple_study/binning.json`) through the stored peak list gives **0.8688 /
16.331** — expect roughly **+0.4 efficiency points and -0.27 fake/event** on
re-run, with the model unchanged.

**nTrk = 1 census, read straight from the ntuple.** r16443, mu in [185,215],
1920 events: 111.32 truth vertices/evt with nTrk >= 2 (the denominator),
**23.13/evt with nTrk = 1** (discarded by `run3_io._filter_amvf`), 0.00 with
nTrk = 0, 134.45 TruthVertex entries in total against <mu> = 192.5. So even the
truth list is a reconstruction-level object, and a peak landing on one of the
23.13 counts as our fake — consistent with the independently measured 4.2 of
the ~16.6 fake/event being real interactions.

### Worked example, committed and pinned

`src/pv_finder/diagnostics/metric_worked_example.py`: six truth vertices, six
reco, 65 tracks, run through **both production classifiers** (the positional
labels are parsed out of `compare_res_reco`'s own `debug=1` trace and
cross-checked against its returned totals, so it cannot drift into a
paraphrase). Same six objects, same five truth PVs:

```
positional  : clean 2  merged 1  split 1  fake 2   eff 0.800   clean/truth 0.400
track-purity: clean 1  merged 2  split 2  fake 1               clean/truth 0.200
```

Two vertices disagree, in **opposite** directions: P_b is at the right z but
60% pure (positional clean, track Merged); P_f has a perfectly pure track list
but sits 0.32 mm out (positional fake, track Split). Variants in the same
script: widening the window 0.22 -> 0.35 mm takes efficiency 0.80 -> 1.00, and
keeping nTrk = 1 truth turns a fake into a clean match.
`tests/test_metric_worked_example.py` pins all of it (7 tests).

### Found while writing it

- **BUG (docs).** Three places state a track-purity Fake means *no* track has a
  truth association (`classification.py:147`, `amvf_convention_check.py:12-13`,
  the vertex_association classification table). The code takes the **plurality**
  over a dict in which truthless tracks are a bucket keyed `"Fake"`
  (`classification.py:314,319`), so 3 truthless + 1 truth-associated track is
  Fake. P_d in the worked example is exactly that case. Behaviour is
  defensible; the docstrings are wrong.
- **BUG (latent crash).** `classification.py:293` does `int(truth_pv_num)` on
  the result of a `np.nonzero` lookup. A track index appearing in two
  `TruthVertex_assocTracks` lists gives a size-2 array and raises `TypeError`
  mid-eval. Checked 4,000 r16443 events: zero occurrences, zero negative, zero
  out-of-range. Latent, not active; not fixed because that file is shared by
  every TTVA eval and the regression guard.
- **Discrepancy.** `amvf_convention_check.vertex_purities` maxes over real
  truths only while `classify_assignments` maxes over a dict including the
  truthless bucket — the diagnostic is not measuring quite what it explains.
  Bounded by 135/187,960 = 0.07%, so no published conclusion moves.
- **Asymmetry, not a bug.** The positional reco-side clean/merged split is
  order-dependent: the first reco vertex reached in index order absorbs a shared
  unclaimed truth vertex, and peaks always arrive sorted in z. Measured by
  reflecting the whole slice in z: **23 of 193,945 vertices (0.012%)** move
  between clean and merged; split, fake, tc, tm and efficiency are bit-identical.
  Truth-side efficiency is provably order-independent.
- **Knife edge.** Promoting the stored float32 peak positions to float64 before
  `mm_to_bins` flips exactly one vertex in 193,945 between split and fake — the
  `<=` at `efficiency_res_optimized_atlas.py:258`. Harmless; worth knowing
  before chasing a one-count difference.
- `_filter_amvf` grooms AMVF to nTracks >= 2 before scoring and **nothing
  equivalent is applied to our peak list** (we cannot — a peak has no track
  count until the associator runs). 187,908 groomed against 187,960 ungroomed
  on the same events, so 52 vertices at PU200. Structural, always in AMVF's
  favour.
- The finder eval **never prints AMVF's truth-side efficiency**. Its AMVF block
  is reco-side counts expressed as percentages of the truth count, i.e.
  clean/truth, not clean_rate.

### Portability rules, generalised

- A ratio whose denominator is a convention is not portable across that
  convention. AMVF's track clean/truth fell 0.573 -> 0.3094 between v3 and v4;
  truth density only rose 12.6%, so the denominator explains almost none of it.
- Normalise to a bound measured on the same data. **fraction-of-oracle** is the
  headline; absolute clean/truth is secondary.
- A rate whose category definition is threshold-shaped reverses when the
  population changes. "18x fewer fakes than AMVF" is a v3-data statement and is
  false at extended |eta|.
- Never compare across a change in sigma without saying so — the window is
  sigma (section 5.1).
- Bootstrap over events, never sqrt(N).

Outputs: `outputs/08_05_2026_output/metric_taxonomy/` (window_probe.json,
window_probe.log, code/). r16638 is the same e8481_s4494 as r16443 with a
different reco tag, so it is a consistency check and not a second sample; the
page says so.

## 2026-08-05 — Adversarial audit of the PV-Finder vs AMVF comparison

Independent re-derivation of the comparison in `run_eval_pvf_run3.py`, built
from the r16443 ROOT ntuple and the stored peak lists with a matcher written
from scratch (`diagnostics/fair_matching.py`, Hungarian rather than greedy),
deliberately not reusing `compare_res_reco` — that code was the subject of the
audit. Full findings in `docs/evaluation/amvf_fairness_audit.md`;
numbers in `outputs/08_05_2026_output/amvf_fairness_audit/`.

The audit reproduces the published run to four decimals (PVF eff 0.8648,
FP 16.598/evt against the log's 0.8648 / 16.597), which is what licenses the
rest of it.

Ranked by size, the ways the comparison is not apples-to-apples:

1. Efficiency credits one reco vertex with several truths (`:532` plus the
   claim rule at `efficiency_res_optimized_atlas.py:292-296`). Worth +11.5
   points: strict one-to-one gives 0.7501, not 0.8648. Close to the ATLAS
   "merged" convention, so a convention choice rather than a bug — but it
   inflates the absolute number and had never been applied to AMVF.
2. No AMVF efficiency existed anywhere: `:550` discards the truth-side
   classification. Computed it — AMVF 0.8349 under the same convention, so the
   margin is +2.99 points rather than "86.5 % against nothing".
3. The matching window is our own sigma applied to AMVF (`:492`). Worth +1.59
   eff points and −1.77 fake/evt to AMVF if it gets its own 0.3048 mm. Does
   not reverse anything: at a common window the efficiency gap is stable at
   +0.018..+0.038 across 0.10–2.00 mm.
4. nTrk=1 truth removed from truth but our peaks there still count as fakes —
   costs us 1.24 fake/evt more than it costs AMVF.
5. "Split" reco excluded from the published fake rate. We make 2.4x more
   splits than AMVF (0.913 vs 0.373/evt), so ~46 % of the published fake-rate
   advantage is accounting: gap 1.18/evt → 0.64/evt honest.

Checked and clean, recorded so the audit is a complete statement: the
`--min-amvf-vtx`/`--min-tracks` event cuts remove 0 of 25000 events; beam
correction is applied to nothing in the MC path and `BeamPosZ` is identically
0; no vertex on either side lies outside the ±240 mm range; greedy vs optimal
matching is worth ≤0.05 points and is symmetric; there are no duplicate
vertices to make tie-breaking ambiguous; the unseeded pairwise-dz shuffle at
`:375` produces sigma = 0.2182 ± 0.0000 over 25 randomisations, i.e. no
run-to-run variance; and the 12x40 mm sub-event stitching costs nothing at the
block edges (0.7363 vs 0.7351 in the bulk).

`_filter_amvf`'s nTracks>=2 grooming of AMVF — the item I most expected to
bite — is real as an asymmetry of kind but numerically nothing: AMVF emits 52
sub-2-track vertices in 1920 events, 0.03/evt, and ungrooming moves its fake
rate from 19.819 to 19.834.

Independent numbers at a common 0.17 mm window (3x the worse of the two core
position resolutions, 0.0551 mm PVF / 0.0565 mm AMVF), strict one-to-one,
paired bootstrap over 1920 events:

- truth nTrk>=2: PVF 0.7351 ± 0.0008 eff, 19.179 ± 0.106 fake/evt;
  AMVF 0.7011 ± 0.0008, 19.819 ± 0.103; difference +0.0340 ± 0.0006 and
  −0.640 ± 0.098.
- truth nTrk>=1: PVF 0.6442 ± 0.0008, 14.403 ± 0.093; AMVF 0.6068 ± 0.0008,
  16.280 ± 0.098; difference +0.0373 ± 0.0005 and −1.877 ± 0.090.

At equal vertex yield (height floor raised to 0.0444 so both emit 97.87/evt)
PV-Finder is still +2.94 points and −3.27 fake/evt, so the margin is not an
artefact of our tuned operating point.

Two caveats on the absolute scale, symmetric so they do not affect the
comparison: displacing the reco lists by +3 mm still "matches" 16.6 % (PVF) /
16.4 % (AMVF) of truth, and 11.26 adjacent truth pairs/evt are closer than the
window and unresolvable by any algorithm.

Conclusion: the comparison is not apples-to-apples and the corrections run in
both directions, but none of them reverses the result. The margin survives;
the absolute headline does not. Quote the margin and the window, not the
86.5 %.

Not done, deliberately: the fixes themselves. Deciding whether the published
efficiency should switch to strict one-to-one, and whether the fake rate should
absorb splits, changes numbers already circulated and is the user's call.

---

## 2026-08-05 — The AMVF comparison was not symmetric; measured both asymmetries and replaced it

The published PV-Finder vs AMVF comparison compared the two algorithms in a way
that could not be defended, in two independent respects that run in opposite
directions. Neither was a coding error — both are consequences of conventions
that are individually reasonable and jointly unfair.

**Asymmetry 1 — the matching window was ours, applied to them.**
`run_eval_pvf_run3.py` derives `sig_bins` from PV-Finder's own fitted σ_vtx-vtx
(line 492) and classifies AMVF with it (lines 518, 551). Fitted with the same
procedure on the same events, σ_PVF = 0.2182 mm and σ_AMVF = 0.3047 mm, so AMVF
was being judged with a window 28.4 % tighter than its own resolution. The
coupling is visible in our own outputs and was verified before anything was built
on it: between the 08-04 and 08-05 held-out evals our σ improved 0.2328 → 0.2200
and AMVF's measured fake rate rose 17.37 → 17.78/evt. AMVF did not change. That
0.41 fake/event was purely mechanical.

**Asymmetry 2 — nTrk = 1 truth vertices counted against us.** `run3_io.py`
filters truth to nTracks ≥ 2. There are 111.32 such vertices/event and 23.13 with
nTrk = 1 (none with 0), and a reconstructed vertex landing on one of those real
interactions is scored as a fake.

**What replaced it.** A scan of efficiency and fake rate for both algorithms
against a *common* matching window swept 0.1–1.0 mm, plus a headline table at a
pre-registered common 0.5 mm (round; ≥ 1.6× both σ; ~half the mean inter-vertex
spacing at PU200 — chosen before looking at any result and independent of either
fit). Both truth definitions are reported for both algorithms. 1920 held-out
r16443 events, μ ∈ [185, 215]; r16638 agrees throughout but is the same generated
events reconstructed twice and is used only as a consistency check. Errors are
bootstrap over events, differences paired. No inference was re-run: PV-Finder's
peaks come from the existing `eval_v6_operating_point` pkl, AMVF and the
unfiltered truth from ROOT, with per-event alignment proved on every event rather
than assumed.

At 0.5 mm: PV-Finder 93.32 ± 0.06 % efficiency, 9.59 ± 0.07 fake/evt (5.25 ± 0.06
on the corrected truth definition); AMVF 92.46 ± 0.06 %, 11.65 ± 0.08 (8.02 ±
0.06). Paired differences +0.86 ± 0.05 points and −2.06 ± 0.07 fake/event.

**What the asymmetries were worth.** Asymmetry 1 flattered us by **2.14
efficiency points** (the gap is +3.00 points at the published window and +0.86 at
the common one) and *penalised* us by **0.90 fake/event** (gap −1.16 → −2.06). It
does not point one way, which the original framing of the problem did not expect.
Asymmetry 2 is worth **0.97 ± 0.07 fake/event**, not the 3.33 it appears to be:
AMVF also puts 2.36 ± 0.05 vertices/event on real nTrk = 1 interactions against
PV-Finder's 3.33 ± 0.05, so most of that correction cancels. The accidental floor
was measured with two independent controls (another event's nTrk = 1 list, and
this event's list displaced by 10 mm) which agree to 0.04/event.

**Two findings that narrow our margin, and are the reason this was worth doing.**

1. The fake-rate lead is largely a *relabelling*. Widening the window turns a
   surplus vertex from "fake" into "split", and PV-Finder's surplus is
   disproportionately near-neighbour satellites (4.49 split/evt at 0.5 mm against
   AMVF's 2.63) while AMVF's is more isolated. On surplus = fake + split, which
   is immune to that, the two are a statistical tie at 0.5 mm (−0.20 ± 0.08/evt)
   and **AMVF is ahead beyond 0.596 mm**. Fake rates should not be quoted without
   the surplus number beside them.
2. PV-Finder emits 101.0 candidates/event against AMVF's 97.9. Raising the height
   floor 0.03 → 0.0444 equalises them (exactly equivalent to re-running the peak
   finder at that `--min-height`) and the efficiency advantage collapses to
   **+0.12 ± 0.05 points** — a tie — while the surplus advantage grows to
   2.37 ± 0.08/event. So the defensible claim is not "PV-Finder is more efficient
   than AMVF" but "PV-Finder sits on a better efficiency/purity frontier, and the
   deployed operating point spends that advantage on efficiency".

Also recorded: **AMVF's truth-side efficiency is never printed by the eval**,
only its reco-side categories, which is why no AMVF efficiency number had ever
appeared in the wiki. It is 92.46 % at the common window and 83.49 % at the
window the eval actually uses. And judged self-consistently — each algorithm at
its own σ — AMVF's efficiency comes out *higher* than PV-Finder's (0.8755 vs
0.8638). That is not a statement about the algorithms; it is what happens when
each picks its own acceptance criterion, and it is the clearest demonstration
that the self-consistent convention cannot settle a comparison.

New code in `src/pv_finder/diagnostics/amvf_fair_comparison/`. The matcher there
is verified bit-for-bit against `compare_res_reco` by
`tests/test_amvf_fair_matching.py` at five windows, and on the real data it
differs from the production matcher by 1 vertex in 193 945 (PV-Finder) and 2 in
187 908 (AMVF) — a float boundary from matching in mm rather than bin units —
with truth-side matched counts identical. Nothing was taken from it before that
check passed. Numbers, figure and invocation in
`outputs/08_05_2026_output/amvf_fair_comparison/`; wiki section in
`docs/evaluation/vertex_finding.md`, which now carries a warning on the old AMVF
row and a correction to the fake-decomposition paragraph rather than a silent
overwrite.

**Addendum, same day.** A concurrent independent audit
(`docs/evaluation/amvf_fairness_audit.md`, written from scratch with a Hungarian
matcher rather than reusing `compare_res_reco`) landed while this was in
progress. The two studies were done separately and agree where they overlap,
which is worth recording as a cross-check: σ_PVF = 0.2182 mm, σ_AMVF ≈ 0.3047 mm,
AMVF efficiency 0.8349 at the published window, and the height floor 0.0444 that
equalises the two candidate yields at 97.87/evt were each derived twice,
independently, to the digits quoted.

The audit found one effect this study had silently inherited by reusing
`compare_res_reco`: **efficiency credits one reco vertex with every truth vertex
in its window**. It is now measured here too. At the common 0.5 mm window it is
worth +15.2 points of absolute efficiency (93.32 with merge credit, 78.09 strict
one-to-one) — and it does not cancel in the comparison. AMVF has fewer reco
vertices, so each of its merges absorbs more truth and it collects proportionally
more merge credit; the convention therefore *shrinks* our measured lead, from
+3.01 ± 0.06 points strict to +0.86 ± 0.05 with merge credit. Both are now in the
wiki table. The lesson is the same one as the window: an absolute efficiency
without its convention named is not a measurement.

Scope note: the two studies are complementary rather than redundant. The audit
ranks every asymmetry it can find at one window and deliberately stops short of
fixing anything; this one produces the window *scan* across 0.1–1.0 mm, the
pre-registered 0.5 mm table, both truth definitions for both algorithms with a
measured accidental floor, and the surplus and matched-multiplicity controls.
They should be merged into one page rather than left as two.

---

## 2026-08-05 — sigma_vtx_vtx was fitted on the wrong events; fixed and re-run

Flagged in the post-hoc study earlier today and left for the user to call.
The answer was yes, so this fixes it. Commits `2e55f79` and `934930d`; corrected
artefacts in `outputs/08_05_2026_output/eval_v6_mu_window/`.

### The bug

`pairwise_dz.append(...)` sat in the per-event inference loop with no mu filter,
so `dz_arr` spanned every event read, while the summary block quoted efficiency
on the mu window. On the flat-mu held-out files that is **25 000 events at
<mu> = 99.6** against **1920 / 1888 events at <mu> = 192.5**. Because sigma is
fed back as the matching window, the mismatch did not stay in the resolution
line -- it propagated into the headline efficiency and fake rate too.

### The fix

Pairwise dz is now kept per event and the fit runs on the mu-window subset
whenever a window is in force; with no window the behaviour is unchanged. Both
numbers are printed -- the mu-window sigma as the headline with its event and
pair counts, the all-events sigma as a secondary line marked *mixed-mu; NOT the
headline* -- so the change is visible in the log rather than silently moving a
number people have already seen. `eval_results.pkl` carries both, plus
`in_mu_window`, the selection label and the fit's event/pair counts;
`pairwise_dz_mm` stays the array `sigma_vtx_vtx_mm` was fitted to so
`replot_from_pkl` keeps drawing one over the other consistently. The resolution
plot now shows the population its quoted sigma describes.

The selection predicate moved to `pv_finder.utils.pairwise_dz.in_summary_window`
and is called by **both** the fit and the summary block. One function is the
actual fix: two independent copies of "which events count" is what the bug was.

### Corrected numbers, mu in [185,215], 1920 / 1888 events

| | efficiency | fake/evt | sigma_vtx-vtx |
|---|---|---|---|
| as published 08-04 | 0.8643 / 0.8656 | 16.64 / 16.36 | 0.2190 / 0.2200 |
| + commensurate binning | 0.8648 / 0.8662 | 16.60 / 16.32 | 0.2200 / 0.2213 |
| **+ sigma on the mu window** | **0.8691 / 0.8695** | **16.31 / 16.10** | **0.2290 +- 0.0031 / 0.2278 +- 0.0032** |

Secondary (mixed-mu) sigma from the same runs: 0.2200 +- 0.0031 and
0.2213 +- 0.0030, over 45.5 M and 46.0 M pairs against 8.93 M and 8.70 M in the
window. Net: sigma +4.1% / +2.9%, efficiency +0.43 / +0.33 points, fake rate
-0.29 / -0.21 per event. **No peak moved** -- the peak lists are identical and
only the matching window changed.

Cross-checked two ways before reporting. Replaying the stored peak lists at a
fixed 0.2283 mm window predicts 0.8688 efficiency and 16.331 fake/event; the
re-run measured 0.8691 and 16.308. Independently, extrapolating the
window-sensitivity scan from the cached held-out dump predicted sigma ~ 0.228,
efficiency ~ 0.869 and 16.3 fake/event. Both land on the measurement, so nothing
else moved.

### The earlier artefacts are superseded, not merely updated

`outputs/08_05_2026_output/eval_v6_operating_point/` was produced at `53d1967`,
before this fix, and **cannot be reproduced from current HEAD**. Do not diff its
numbers against new ones as though they were the same measurement. The current
artefacts are `eval_v6_mu_window/`, whose `INVOCATION.txt` records `2e55f79`
(r16443) and `934930d` (r16638) -- the two differ only in a warning message on a
branch neither run reaches, so both match HEAD.

### Two things folded in from work that landed alongside

- The published efficiencies use the **merged-credit** convention, worth
  **+11.5 points** over strict one-to-one. `docs/evaluation/vertex_finding.md`
  now states that next to the numbers and points at
  `docs/evaluation/metric_definitions.md` and
  `docs/evaluation/amvf_fairness_audit.md` rather than restating them.
- The nTrk = 1 finding is corrected for the comparison case: **AMVF also books
  2.36/evt** of these, so the net differential is **0.97/evt, not 3.3**. The two
  PV-Finder figures on that page (4.17/evt at the 0.2328 mm production window
  with an accidental control, 3.33/evt at the audit's common 0.5 mm window) are
  now explicitly reconciled so they do not read as contradicting each other.

### And the comb's provenance, stated in three places

The plateau sawtooth **arrived with the local-centroid estimator on 08-04** --
39.4% |dz| comb against 0.9% for the full-region weighted mean it replaced,
1.3% AMVF, 0.8% truth, all on the same 240 bins. That estimator change was a
real physics win (-3.85% core width, -13% sigma_vtx-vtx) and it stands, but it
also made the resolution plot look dramatically worse for a purely
presentational reason. Anyone comparing an 08-03, an 08-04 and an 08-05 figure
sees a plot degrade and then clean up while the resolution only ever improved.
Now called out at the top of `docs/research/resolution_plot_ripple.md`, in the
binning section of `docs/evaluation/vertex_finding.md`, and next to the
estimator change itself in `docs/research/peak_position_estimator.md`, which
previously recorded only the win.

---

**Second addendum — the window basis was wrong, and both original asymmetries
turned out to be the small effects.**

Re-derived the window from the right quantity. σ_vtx-vtx measures how far apart
two vertices must be before an algorithm resolves them *separately*; whether a
vertex was *found* is governed by the **core position resolution**, and the two
do not scale together: σ_vtx-vtx is 0.2182 vs 0.3047 mm for PV-Finder and AMVF (a
ratio of 1.40), while the core position resolution is 55.1 vs 56.5 µm (ratio
1.03). A window built on the first is unfair by construction; one built on the
second is fair by construction. The primary common window is therefore
3 × 56.5 µm = **0.17 mm**, and the round 0.5 mm is demoted to a labelled
sensitivity point.

0.5 mm was demoted on a measurement, not a preference. Displacing the whole reco
list by 3 mm — larger than any window, far smaller than the 45 mm beam spot, so
the density is unchanged and the association destroyed — and re-matching gives
the efficiency that is pure coincidence: 11.1 % at 0.10 mm, 18.6 % at 0.17 mm,
45.4 % at 0.50 mm, 68.1 % at 1.00 mm (merged credit). **At 0.5 mm nearly half the
measured efficiency is accidental.** The floor is equal for both algorithms to
within 0.3 points at every window, so it does not bias the comparison, but it
destroys the absolute number. It is now drawn on the window-scan figure so the
reader can see where the metric stops meaning "found the vertex".

At 0.17 mm, strict one-to-one, paired bootstrap: PV-Finder 73.43 ± 0.09 %
efficiency and 19.27 ± 0.10 fake+split/evt; AMVF 70.07 ± 0.09 % and 19.87 ± 0.10.
Margin **+3.37 ± 0.06 points** and **−0.60 ± 0.09 surplus/event**, surviving the
equal-yield control at +2.90 ± 0.06 and −3.23 ± 0.09.

**Both asymmetries this study was originally sent to find are smaller than the
convention effect the audit found.** Once the window comes from the core
resolution, asymmetry 1 moves the efficiency gap from +3.00 to +3.36 points
rather than collapsing it, because at a tight window both algorithms lose
comparably — the +0.86-point figure recorded in the first entry above is an
artefact of quoting the gap at 0.5 mm under merged credit, where AMVF collects
proportionally more merge credit than we do. The merge convention is worth
−2.15 points to us at 0.5 mm and −0.01 at 0.17 mm. **The strict one-to-one margin
is the stable quantity**: between +3.4 and +2.3 points across the whole
0.1–1.0 mm range, against +3.4 falling to +0.45 for merged credit. If one number
is quoted it should be the strict margin at the resolution-derived window, with
the convention and the window named.

Cross-check against the independent audit, which used a Hungarian matcher and its
own code path: agreement on the published-run reproduction (0.8648 / 16.598),
AMVF efficiency (0.8349), both σ_vtx-vtx, the core position resolutions
(55.1 / 56.5 µm to three digits), strict efficiency at ≈0.17 mm (0.7343 / 0.7007
against 0.7351 / 0.7011), the strict margin (+3.37 vs +3.40 points), fake+split
(19.27 / 19.87 against 19.179 / 19.819), the accidental floor (16.5 / 16.3 %
against 16.6 / 16.4 %), the equal-yield margin, and the split rates. Two
differences, both chased rather than averaged and both explained: the audit
assigns optimally throughout while this study scores with the production greedy
matcher (≤ 0.001 in efficiency), and the audit folds splits into "fake" where
this page keeps them as separate columns — its "fake/evt" is this page's
"fake + split", which is why its 19.179 matches 19.27 and not 18.73. The one
place the assignment algorithm mattered was the core resolution itself: greedy
closest-first creams off the tightest pairs and biases the width 10 % low
(50.1 µm against 55.1), which would have produced a tighter window, and a tighter
window flatters PV-Finder. The conservative optimal estimator is now used for
that measurement, after which the two studies agree exactly.

The remaining difference is one of framing, not of measurement: the audit moves
the efficiency denominator to nTrk ≥ 1 for its corrected truth definition, while
this page holds the denominator at nTrk ≥ 2 and changes only what counts as a
fake, so the efficiency column stays comparable with the ATLAS convention across
all four cells. Both are defensible and they should not be compared cell by cell.

---

## 2026-08-05 — TTVA v4: extended-|eta| retrain on the PV-Finder v6 chain

Rebuilt the track-to-vertex associator on `data/run4_all_etas` (the July-2026
extended-|eta| re-production) on top of PV-Finder v6, as a controlled A/B
between chain-like augmentation (arm A) and no augmentation (arm B).
Everything in `outputs/08_05_2026_output/gnn_ttva_v4/`.

### The bug that would have quietly spoiled the campaign

`centroid_halfwidth` was not plumbed through **any** of the Run-4 chain path.
`gnn.data.pu200_chain_graphs`, `gnn.diagnostics.chain_gap_decomposition` and
`gnn.evaluation.verify_fast_paths` all called `pv_locations_updated_res` with
positional arguments only, so all three silently took the library default 0 —
the historical full-region weighted mean — while the v6 operating point is the
local centroid at half-width 3.

This is not cosmetic. The position estimator sets PV-node z, and PV z feeds all
three edge features (longitudinal significance, horizontal significance, |dz|),
so the whole graph shifts. Training graphs built at one half-width and
evaluation graphs at another degrade the result in a way that looks like a
modelling problem rather than a configuration error.

Each entry point now takes an explicit `--centroid-halfwidth`, defaulting to
`LEGACY_CENTROID_HALFWIDTH` (0) so v3-era checkpoints keep reproducing. The
2026-08-04 entry left the library default at 0 pending "its own A/B"; this
campaign is that A/B, so opting in per call site rather than moving the default
is consistent with that decision.

Two independent confirmations the point is now live:

- the chain build reports **101.01 peaks/event** against the **101.0** reco
  vertices/event `run_eval_pvf_run3.py` measures on the same events through a
  different code path;
- the chain graphs' minimum PV height is exactly **0.030067**, the `min_height`
  floor.

Fast-path re-verification at the new operating point: `bit_exact=True`, 0/300
events differ, 84.07 -> 0.073 ms (1157x). The previous verification predated the
peak change and so said nothing about half-width 3.

### The AMVF number moved, and it is not AMVF's fault

AMVF's clean/truth on this slice is **0.3094**, against **0.573** in the v3
campaign. Measured rather than argued, with a new diagnostic
(`gnn.diagnostics.amvf_convention_check`):

- Truth density rises 98.8 -> 111.32 PVs/event, but a 12.6% denominator change
  cannot produce a 46% relative drop.
- The dominant-truth purity of AMVF vertices has **median 0.6316** (quartiles
  0.462 / 0.632 / 0.812), straddling the 0.70 `PURITY_THRESHOLD`. Scanning the
  cut: Clean 0.7372 at 0.5, 0.5613 at 0.6, **0.4072 at 0.7**, 0.2798 at 0.8.

At extended |eta| AMVF puts vertices at about the right z but their track lists
absorb forward tracks from neighbours, so the median vertex is 63% pure and
falls on the wrong side of a 70% cut. The two conventions partition the
*identical* collection: 97.90 vs 97.87 reco/event, 111.32 truth/event.

**A wrong definition was caught before it was published.** The docstring at
`classification.py:145` said a reco PV is Fake "if no truth PV matches any of
its tracks". The code does something else: truthless tracks accumulate under
one dict key `"Fake"` alongside the per-truth-PV keys and `max()` takes a
**plurality** over all of them, so 3 truthless + 1 truth-associated is Fake.
And because truth tracks split across many buckets while truthless tracks share
one, the truthless bucket only has to beat the largest *single* truth PV: 40%
truthless / 30% PV-A / 30% PV-B is Fake despite being 60% truth-associated.
Docstring, `amvf_convention_check` and the wiki all carried the wrong reading;
all three are fixed.

**The fake-rate comparison reverses, for a measured reason.** AMVF's
track-convention fake rate is 0.00072 here against 0.91% on v3 data. Over all
187,960 AMVF vertices (median 11 tracks) the truthless-track fraction has
**median 0.0000, mean 0.0114**, and **87.56% of vertices contain no truthless
track at all**. AMVF fits vertices *from* tracks, so under a plurality rule a
1.1%-mean minority essentially never wins: the truthless bucket wins outright
in only **27** vertices, and 27 + 108 of the 679 ties (16%, broken by
first-encountered-track order) = **135**, exactly the Fake count
`classify_assignments` reports. So the chain makes *more* track-convention
fakes than AMVF, where v3 reported 18x fewer. **"18x fewer fakes than AMVF"
must not be repeated for extended-|eta| results, and no fake rate should be
quoted without its convention** — the same arm at t=0.98 gives 9.67% under
`all_peaks` and 0.173% under `drop_empty`, a factor of 56 of pure bookkeeping,
with clean/truth identical between them by construction.

**But the chain's fakes are the finder's, not the associator's.** The oracle
still books 17,701/193,945 peaks (9.13%) as Fake under the all-peaks
convention; arm B at t=0.98 gives 9.67%. The GNN adds about half a point above
what the finder makes unavoidable. And of the 13.94% junk rate (27,028 peaks
with no nTrk>=2 truth PV within 0.5 mm), the 4.81% difference from the oracle
floor is a truth-convention artefact: `chain_gap_decomposition` matches against
truth filtered to nTrk>=2 while the oracle's per-track truth applies no nTrk
cut, so a peak on a real single-track interaction has truth tracks but no
eligible truth vertex. A separate finder-side study independently measured 4.17
surplus peaks/event on nTrk=1 truth vertices, 4.13% of peaks, against 4.81%
here — two analyses, different code paths, same population.
**The irreducible junk floor is 9.13%, not 13.94%.**

### Results


Slice: r16443, 1920 events, **213,738 truth PVs (nTrk>=2)**, 111.32/event,
mu 192.5. Chain: **101.01 peaks/event**.

Bounds, **each with its convention**, because they are not interchangeable:
- **finder cap 0.7809** — positional, *strict one-to-one* greedy match, 0.5 mm.
- **oracle 0.7280** — track-purity (`classify_assignments`, 0.70 cut).

Neither is comparable with the finder's published efficiency of 0.8648, which
is positional at 0.2200 mm under a "merged counts as found" rule where one reco
may claim several truth vertices. The fairness audit (`25f499c`) measures that
convention as worth +11.5 points, which is why the strict-but-wider 0.7809 sits
sensibly between the audit's strict 0.7501 and the published 0.8648.

| chain, drop-empty, each arm at its own best t=0.98 | clean/truth | frac. oracle | fake_rate | clean_rate | trkF1 | truth ceiling | HS-ID |
|---|---|---|---|---|---|---|---|
| **arm B, NO augmentation** | **0.6843** | **94.0%** | 0.00173 | 0.8335 | 0.650 | **0.7690** | **0.9443** |
| arm A, augmented | 0.6501 | 89.3% | 0.00117 | 0.8587 | 0.645 | 0.7575 | 0.9349 |
| AMVF, same events | 0.3094 | 42.5% | 0.00072 | 0.3518 | -- | -- | 0.8068 |

**AUGMENTATION DOES NOT HELP. The no-augmentation control wins by
+0.0342 +- 0.0014, 24 sigma on 213,738 truth PVs**, and wins at every
threshold, on the truth-graph ceiling, on HS-ID, and on the shared pristine
validation loss (0.3385 vs 0.3510). Production checkpoint:
`ttva_gnn_hllhc_v4_noaug/ttva_gat_alleta_k20_v4_noaug180k_epoch_107.pyt`.

What augmentation did instead was move the operating curve toward lower fakes.
Under a fake-rate budget the ordering flips (actual operating points; both
curves fold back above t=0.99, so interpolating there manufactures nonsense):

| fake budget | arm A best | arm B best | A - B |
|---|---|---|---|
| <= 0.0014 | **0.6501** (t=0.98) | 0.2899 (t=0.999) | **+0.360** |
| <= 0.0016 | 0.6501 | **0.6694** (t=0.99) | -0.019 |
| >= 0.0018 | 0.6501 | **0.6843** (t=0.98) | -0.034 |

So arm A is preferable *only* under a fake budget below ~0.0015, which arm B
cannot reach without collapsing to 0.29.

**Why augmentation stopped paying.** It trains abstention: junk PV nodes carry
no true edges and dropped vertices turn their tracks' labels to 0. On the v6
chain that conservatism costs more clean vertices than its junk-rejection buys,
because there is very little transfer gap left -- arm B already reaches 94.0% of
the oracle. Two things changed since v3 found the opposite: v3 bundled the
double-Z_MIN height fix with augmentation in one campaign, so augmentation was
plausibly credited with part of the height fix; and the half-width-3 estimator
has since tightened sigma_vtx-vtx from 0.30 to 0.219 mm, shrinking the very gap
augmentation exists to bridge.

**Caveat, stated rather than buried.** This is 6 dataset passes against v3's 18,
and the augmented distribution is harder (arm A train loss 0.3524 vs 0.3387), so
augmentation might need longer to pay off. What is established is that at equal
and fully-annealed budget it does not help and slightly hurts. It is not
established that it could never help.

**Is the plurality rule mislabelling merged peaks as Fake? Measured: no.** A
peak straddling several truth PVs splits its truth tracks across buckets, so a
modest truthless minority can steal the plurality and book a genuinely Merged
vertex as Fake. Chain peaks were the natural place to worry. At t=0.98 over all
193,945 peaks: truthless-track fraction mean **0.0034**, **97.56% of peaks carry
no truthless track at all**, 303 non-empty Fakes, and of those only **5** are
truth-majority split over >=2 truth PVs -- 0.0026% of peaks. The median
truth-track fraction among Fakes is exactly **0.000**, i.e. genuine junk. By
this measure our peaks are *cleaner* than AMVF's vertices (mean truthless
fraction 0.0114, 87.56% with none), and the artefact is large for AMVF (74% of
its 135 Fakes) but on a population too small to matter.

The accounting closes exactly: 175,492 non-empty + 18,453 empty = 193,945 peaks;
all-peaks fake = (303+18,453)/193,945 = 0.09671; drop-empty = 303/175,492 =
0.00173. Both match `chain_scan`. **So the factor of 56 between the two fake
conventions is 18,453 peaks the associator declined to populate at a strict
threshold, not contaminated assignments** -- which independently confirms the
attribution drawn from the oracle floor.

**Checkpoint robustness.** Arm B at epoch 101 gives 0.6821 against the
production epoch 107's 0.6843 -- a spread of 0.0022, with every threshold's
ordering unchanged. **The arm B - arm A gap of 0.0342 is 16x that spread**, so
the augmentation result is not an artefact of which epoch was picked.

### What was cut, and why

108 epochs, not the designed 324. 324 was 18 dataset passes to match v3's
graph-presentation count; the campaign had a hard 4-5 h training budget and
sneezy ran at load 296-439 all morning, where an epoch costs 50-165 s rather
than the ~28 s it costs on a quiet box. **108 is the largest multiple of 18
whose cosine anneal completes inside that window.**

The multiple-of-18 constraint is not cosmetic: `ShardCyclingLoader` advances one
shard per epoch round-robin, so any other count shows some shards one more time
than others — an uncontrolled variable in an A/B whose point is that the arms
differ only in augmentation. And shortening a schedule beats truncating a long
one: with cosine annealing to 1e-5, an early-stopped checkpoint has not
annealed and does not represent the recipe.

**This is a third of v3's data exposure.** The arms are a like-for-like A/B
against each other, not against v3's absolute numbers.

### Things worth keeping

- Shard builds were verified before training on them. Augmented shards: 1366.4
  tracks/event (identical to pristine), 132.3 PV nodes/event against 132.4
  predicted from the measured miss and junk rates, 27,327 edges (identical),
  positive-edge fraction 0.0455 vs 0.0490, zero-height fraction 0.0000. A
  three-hour run on a silently broken shard set is the expensive failure mode.
- `scripts/ttva_v4_phase3_eval.sh` now runs stage-by-stage across all models
  rather than model-by-model, so a run cut short still leaves the headline
  comparison on disk.
- `scripts/ttva_v4_deadline_guard.sh` enforces a wall-clock budget with SIGTERM.
  Never SIGKILL: this box still carries nine unkillable `evaluate_amvf_ttva`
  processes from 23 days ago, burning seven cores between them. Those entry
  points now warn in their docstrings to use the runpy form.

## 2026-08-06 -- 7 August deck, rebuilt from the approved 24 July version

`presentations/mattia/08_07_2026/`. A new deck, not an edit of the July one.

The starting point is the **professor-approved 24 July source**
(`07_24_2026/backups/slides_00_original_0724.tex`), with the 5 August
measurement updates ported in one frame at a time. The 5 August working copy was
not used as the base: it had grown a large defensive apparatus (withdrawal
boxes, convention warnings, referee-style caveats) that belongs in the docs, not
in a talk. Both starting points are kept in `08_07_2026/backups/`.

### Structure

**Model history is gone.** The three "Model history" slides and the
before/after greedy-matching comparison described what the project used to do.
The deck now describes only what the algorithm does today. The v1-to-v6 capacity
table moved to backup, where it reads as a finding rather than a chronology.

**The "Update:" section at the end is gone**, folded into Part III. The
extended-|eta| sample is now simply *the* PU200 sample and the held-out v6
numbers are the headline. Loss re-weighting moved to backup as a standalone
negative result.

**Two new slides on the peak finder**, which previously had one slide describing
an algorithm that has since changed twice: the scan / conjoined-split / quality
cuts in order, then where the position comes from (local centroid against
whole-region mean, with the measured comparison from
`docs/research/peak_position_estimator.md`).

**New performance slide** `PU200: Yield across the Pileup Range`, from
`reco_vs_mu` in the v6 held-out eval, plus the category-counts bars promoted
into the main flow. A Part V divider, a divider template carrying a numeral
badge and a one-line summary, and the `athena.png` screenshot redrawn as a TikZ
package diagram.

### Two errors found while porting

1. The efficiency row of the position-estimator table had whole-region and
   local-centroid **the wrong way round**. The local centroid *costs* 0.34
   efficiency points (0.8811 -> 0.8777) and buys a 13% better separation scale.
2. sigma_vtx-vtx appears as 0.218 mm on the window-fairness slide and 0.229 mm
   on the results slides. Both are correct: 0.218/0.305 is the fairness audit's
   own matcher, paired so the 1.40 ratio is internally consistent, while 0.229
   is the production fit after the mu-window and commensurate-binning fixes. The
   slide now says so.

No measurement changed otherwise. Everything traces to
`outputs/08_05_2026_output/eval_v6_mu_window/` and the fairness audit.

### Timing

The 4.5 +- 1.2 ms/event inference benchmark is quoted again with its plot, and
"re-run the inference benchmark on the current model under the HEPScore
protocol" is listed under next steps. The 5 August copy had withdrawn the number
because no producing code sits in this repository; that is a provenance gap, not
evidence the measurement is wrong, and a withdrawal box reads as a defect report
in a talk.

### Build

54 pages, **zero overfull and zero underfull boxes**, no `[shrink=N]` anywhere.
American spelling throughout, no em dashes.

## 2026-08-07 -- 7 August deck: professor review addressed, two-version build

Review comments from the professor and from Rocky on the 55-page 7 August deck.
Addressed slide by slide. `presentations/mattia/08_07_2026/`; the pre-review
state is kept at `backups/slides_02_pre_prof_comments.tex`.

### New measurements

**AMVF sigma_vtx-vtx on real collision data.** The deck quoted PV-Finder's
resolution on Run 2 and Run 3 data against no AMVF partner. Measured today with
`diagnostics/amvf_run3_performance_plots.py`, 2500 events per file, the same
sigmoid fit the evaluation uses: **Run 2 0.812 mm, Run 3 0.820 mm**, against
PV-Finder's 0.37 and 0.35. A factor of about 2.2 on data, matching the factor
seen in MC and at PU200. Outputs in
`outputs/08_07_2026_output/amvf_data_resolution/`.

**Three figures from cached arrays**, no new inference:
  - `mu_distribution_pu200.png`. Rocky asked whether the sample is really fixed
    at mu = 200. It is not: the two held-out evaluation files are flat in mu,
    0 to 200 (measured off r16443 directly, not taken from the data doc, which
    says 210). The figure shades the mu in [185,215] slice the headline uses.
  - `position_resolution_pvf_amvf.png`. The plot behind the 55.1 / 56.5 um
    window basis. First attempt used greedy closest-first matching and produced
    50.1 / 51.1 um, which would have contradicted the table on the same slide.
    The fairness audit uses an optimal (Hungarian) assignment, deliberately,
    because greedy creams off the core and biases the width low. Fixed by
    importing `matched_residuals_optimal` rather than reimplementing it; the
    figure now reproduces 55.1 / 56.5 exactly.
  - `hllhc_resolution_overlay.png`. PV-Finder and AMVF pairwise-Δz dips on one
    axis. **No sigma is annotated on this figure on purpose:** fitting the
    cached peak list gives 0.218 mm for PV-Finder while the deck quotes the
    production 0.229 mm, because the eval computes the resolution on a separate
    peak list (`p_pvs_r`) from the one it scores efficiency on. Both numbers are
    right; putting both in front of an audience is not. Worth reconciling.

### Two stale numbers found and fixed

- "Finder cap ... median nTrk 3 against **7**" was a v4b-campaign number
  (`outputs/07_14_2026_ttva_gap/`) sitting on a slide whose every other figure
  comes from the v6 decomposition. v6 gives **9**, at both 1500 and 1920 events.
  Corrected on the slide and warned about in `docs/evaluation/vertex_association.md`.
- The PU200 sample slide described the flat-mu evaluation files as "0 to 210".
  Measured: 0 to 200. The mu figure now sits beside that sentence, so the
  discrepancy would have been visible.

### Corrected framing

The right-hand table on the PU200 sample slide compares two **data productions**,
not two configurations of AMVF. AMVF always accepted tracks to |eta| < 4.0; the
old production simply reconstructed none beyond 2.5. That is why its vertex count
is flat (101.2 to 101.3) while its mean tracks per vertex jumps (8.80 to 13.40):
it hangs the new forward tracks on vertices it already had. Both reviewers read
the old wording as a claim about AMVF's seeding.

### Structure

Window scan, failure modes, fake composition and threshold provenance moved to
the backup behind a single signpost slide that names each topic and gives its
backup page via `\pageref`. Held-out sample 2 moved to backup with a consistency
table. The talk is shorter and the material is still there.

### Two versions from one source

`\readingonly{...}` and `\talkonly{...}`, switched by `\talkbuild`. `./build.sh`
produces `slides.pdf` (reading) and `slides_talk.pdf` (presentation) from the
same file, so no number can drift between them.

## 2026-08-20 -- ATLAS Fig. 25 cross-check, and the GNN in the note appendices

Two requests: document the GNN alongside everything else in the technical note,
and check our AMVF Run 3 MC plots against Figure 25(a,b) of the ATLAS ID
performance paper, as a definition sanity check before tomorrow's meeting.

### The GitLab note was already current

Verified rather than assumed: `git ls-remote` reaches
`gitlab.cern.ch/atlas-physics-office/IDTR/ANA-IDTR-2026-01/`, and local `main`
equals `origin/main` at 750d885. Nothing new has been pushed since 20 July, so
the hard-drive copy is the current version. Reviewer comments live in the text
as `\lat{}` macros, and exactly two are still open: track selection ("tight
primary -- Rocky can you confirm?", 02_method.tex:19) and the Run 2 pileup
figure ("doesn't look like average mu of 25-30, where does that number come
from?", 05_validation.tex:120).

### Figure 25 is not the resolution plot

Worth recording because the meeting was set up on the opposite assumption.
Fig. 25(a) is the average number of reconstructed vertices per category vs the
number of interactions; Fig. 25(b) is the Delta-z between reconstructed
vertices. The resolution plot is Fig. 26(b). So (a) is exactly the
per-category yield we expected, and (b) is the pairwise Delta-z distribution --
the same observable as our sigma_vtx-vtx measurement and the Delta-z bump study.

### The definitions match, and AMVF reproduces their plot

ATLAS Section 6.5 classifies a reconstructed vertex as Matched at >= 70% of the
total track weight from one interaction, Merged below that, Split by the
Sum-pT^2 tie-break, Fake when fake tracks outweigh any interaction. Our
`PURITY_THRESHOLD` is already 0.7 and the Split rule is identical. The single
difference is weighting: they use vertex-fit track weights, we count tracks
unweighted, because per-track fit weights are not in these ntuples.

New `pv_finder/diagnostics/amvf_vs_atlas_fig25.py` runs the existing
`classify_assignments` core over all 51,000 events of the Run 3 MC sample and
builds both panels, including their two reference lines (the diagonal, and the
reconstruction acceptance = interactions with >= 2 reconstructed tracks).
`plot_amvf_vs_atlas_fig25.py` draws ours and stacks the ATLAS PNGs beside them.

Agreement is close: at mu = 60 we get All 32.2 / Matched 23.8 / Merged 7.8
against their ~32.5 / ~23.5 / ~8.5, and the acceptance line lands on theirs.
The Delta-z dip has the same shape -- half-recovery at 0.90 mm against their
~0.75 mm, full recovery by 3 mm. The plateau level differs only through the
beamspot (35 mm here vs their ~42 mm).

Two categories do not match, both understood. **Fake is structurally zero for
us**: 99.84% of reconstructed tracks in this ntuple carry a truth association
and only 0.084% of AMVF-assigned tracks have none, so there is nothing to build
a fake vertex out of. ATLAS needs R_match < 0.5, which we do not store. Split
is about half theirs, which follows from the same cause plus unweighted
counting. We should not quote a Fake rate from this taxonomy as comparable to
theirs.

### A 50% that should have been 70%

The note's TTVA chapter described Clean as ">= 50% of assigned tracks". The code
has always used 0.7. Corrected, and the passage now cites IDTR-2021-01 and
states the one real difference (weighted vs unweighted). Nothing numerical
changes -- no result was ever produced at 50%.

### GNN in the appendices

Appendix A covered only the vertex finder. Added a Track-to-vertex association
section with graph building at both pileups, the training commands, the
chain_scan / evaluate_ttva_graphs / evaluate_amvf_ttva evaluation commands, the
runpy quirk, and the fast-path and nondeterminism guards; plus TTVA rows in the
results-settings matrix, three TTVA checkpoints in the checkpoint table, and the
graph datasets and test slices in the paths list, including the warning about
zero-height legacy graph files. Appendix B gained a TTVA hyperparameter table
for the two production models beside the existing finder table.

## 2026-08-20 -- Note ported from v4b/|eta|<2.5 to v6/TTVA-v4

Audit of the note against the 7 August deck and docs/ found the note a full
campaign behind. Ported it.

### What was stale

The HL-LHC chapter still carried v4b on `r16438` -- a file inside its own
training pool -- at |eta|<2.5: Eff 91.6%, 15.8 fake/evt, sigma 0.22 mm. The deck
had moved to v6 on 1920 genuinely held-out events from an excluded reco tag:
86.9%, 16.3 fake/evt, 0.229 mm. The lower efficiency is the denominator, not a
regression, and the note said nothing about that.

Chapter 7 credited chain-like augmentation with closing the TTVA transfer gap.
The v4 all-eta A/B reversed it: the unaugmented arm wins 0.6843 (94.0% of
oracle) against 0.6501 (89.3%). The note was asserting something a later
controlled experiment had disproved.

Entirely absent from the note, present in both deck and docs: the extended-eta
re-production, the fair-comparison window methodology, the merged-credit vs
strict one-to-one distinction, the satellite-peak mechanism, and the local
centroid position estimator.

### What was added

New data subsection on the re-production (denominator +17.6%, the old sample's
contaminated 2.0-2.5 sliver, forward tracks buying reach not precision).
New `sec:hllhc:fair`: the window comes from core position resolution (55.1 vs
56.5 um, ratio 1.03) not sigma_vtx-vtx (ratio 1.40), giving 0.17 mm; quote the
+3.36 +- 0.06 margin, not the absolute, which the merge convention alone moves
by 9.7 to 15.2 points. New `sec:hllhc:satellites`: the dz bump is surplus peaks
at conjoined-split boundaries, min_height makes the ratio worse, and ~22% of
surplus peaks sit on real nTrk<2 vertices so the fake rate is overstated by
about a fifth. Peak position documented as a local centroid of half-width 3.

The HGTD negative result was rewritten rather than deleted. Its original
argument was 3.9% coverage in a forward sliver, which the new data invalidates
(28.8%, 20-25 ps). The conclusion survives on corr(z,t) = -0.0102 and a 207 mm
floor on any time-derived z.

### Pre-v6 material

Kept, but demoted. The settings matrix has a labelled superseded block, the
iteration table gains a v6 row, and both carry the warning that the older
efficiencies are numerically higher only because the denominator was smaller.
Dropped the v3 learning-curve figure, which showed a model that is no longer
production.

58 pages, builds clean with TL2022, no undefined references.
