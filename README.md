# PV-Finder

Primary vertex reconstruction for ATLAS Run 3 proton-proton collisions using neural networks.

A UNet (PV-Finder) maps track features to a 1-D z-axis histogram, optionally post-filtered by a Graph Neural Network (GNN).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Documentation

- `docs/` — current-truth wiki
- `JOURNAL.md` — append-only log of what was done and why
- `CLAUDE.md` — AI assistant instructions
