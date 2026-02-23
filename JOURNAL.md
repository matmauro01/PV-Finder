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
