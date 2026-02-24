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
