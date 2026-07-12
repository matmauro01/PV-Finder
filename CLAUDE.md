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
- Important: all code added to this repo must be at minimum thoroughly proofread, and if possible tested. You can do it yourself for small changes, and launch subagents (such as expert coder) for larger ones.

## Documentation

Two systems, both must stay current:

- **docs/** — Wiki. Current truth about the project. Gets rewritten as things change.
- **JOURNAL.md** — Append-only log. What was done, when, why. Never edit old entries.

When you make a meaningful change: update the relevant doc in docs/ AND append to JOURNAL.md.
Always pyush you're code when you're done, and make sure to document it properly before doing so.

## Output Organization

Use dated subfolders for outputs: `outputs/MM_DD_YYYY_output/`. When launching evals or
other output-producing scripts, always use a dated folder. If unsure of the date or
convention, ask the user before writing outputs.

## Running Heavy Processes

**sneezy** is a shared machine. Runaway processes become unkillable zombies (kernel 4.15 bug) and crash the Cursor SSH session.

Rules:
- **Always run training, evaluation, and inference inside `tmux`** — never directly in Cursor's terminal
- Use `python -u` (unbuffered stdout) so you see output immediately
- Set resource limits for experimental/untested runs: `ulimit -v 33554432` (32 GB)
- If Cursor disconnects: from a local terminal run `cursor-fix` (alias in `~/.bashrc`), then Reload Window in Cursor

```bash
# Correct way to run heavy work
tmux new -s eval
source venv/bin/activate
python -u src/pv_finder/evaluation/vertex_finding/run_eval_pvf.py ...
# Ctrl+B, D to detach — safe to close Cursor
```

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
src/pv_finder/                   ← source code (vertex finding)
  models/                          autoencoder_models, alt_loss_A
  data/                            collectdata_poca_KDE, h5_dataset
  training/                        train scripts, trainNet loop, weight init
  evaluation/                      vertex_finding eval pipeline
  diagnostics/                     plotting, visualization
    domain_shift_investigation/      MC vs Run 3 analysis
      feature_distribution/          feature comparison
      kde_study/                     KDE model vs analytical
    per_vertex_visualization/        per-vertex histogram plots
  scratch/                           data_exploration.ipynb
  utils/                           utilities, efficiency, jagged, constants
src/gnn/                         ← source code (track-to-vertex association)
  models/                          ttva_gat (heterogeneous GAT)
  data/                            graph_construction, pvf_to_graphs (PVF hists → graphs)
  training/                        train_ttva, training_loop
  evaluation/                      evaluate_ttva (Clean/Merged/Split/Fake)
  diagnostics/                     track probability distributions (MC, Run3)
configs/vertex_finding/          ← YAML training configs (PVF)
configs/gnn/                     ← YAML training configs (GNN TTVA)
tests/                           ← tests (coming)
```
