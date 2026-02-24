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
