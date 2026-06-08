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
