# PV-Finder

Primary vertex reconstruction for ATLAS Run 3 using neural networks.

## Setup

```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Commands

```bash
# Tests (when they exist)
pytest tests/

# Lint + format
ruff check . && ruff format --check .

# Type check
mypy src/
```

## Commit Rules

Enforced by pre-commit hooks. One concern per commit. Keep commits small.

```
feat:     new functionality
fix:      bug fix
refactor: restructure without behavior change
test:     add or update tests
docs:     documentation only
chore:    tooling, deps, config
```

All enforced automatically on `git commit`:
- Conventional commit message format (rejects non-compliant messages)
- Ruff lint + format (auto-fixes where possible)
- File size limit (500 lines for Python files)
- Large file blocker (no files > 500KB)

## Code Rules

- Type hints on all new functions
- Max 500 lines per file (enforced by pre-commit); split if larger
- No data or model weights in git (see .gitignore)
- Validate data at boundaries, trust internal code
- Prefer simple and readable over clever

## Documentation

Two systems, both must stay current:

- **docs/** — Wiki. Current truth about the project. Gets rewritten as things change.
- **JOURNAL.md** — Append-only log. What was done, when, why. Never edit old entries.

When you make a meaningful change: update the relevant doc in docs/ AND append to JOURNAL.md.

## Project Map

```
CLAUDE.md                        ← you are here
README.md                        ← project overview
JOURNAL.md                       ← append-only log
requirements.txt                 ← dependencies
docs/                            ← wiki (current truth)
  models/                          vertex_finding, vertex_association
  data/                            monte_carlo, run_3
  training/                        vertex_finding, vertex_association
  evaluation/                      vertex_finding, vertex_association
  diagnostics/                     vertex_finding, vertex_association
src/pv_finder/                   ← source code
  models/                          autoencoder_models, alt_loss_A
  data/                            collectdata_poca_KDE, h5_dataset
  training/                        train scripts, trainNet loop, weight init
  evaluation/                      metrics, efficiency, comparisons
  diagnostics/                     plotting, visualization
    feature_distribution/            MC vs Run 3 feature comparison
    kde_study/                       KDE model vs analytical comparison
  scratch/                           data_exploration.ipynb
  utils/                           utilities, efficiency, jagged
configs/vertex_finding/          ← YAML training configs
tests/                           ← tests (coming)
```
