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
