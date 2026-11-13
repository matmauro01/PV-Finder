#!/usr/bin/env bash
# GNN v3 campaign: train one run (configs/gnn/config_gnn_ttva_v3_<RUN>.yml).
# Usage (repo root, inside tmux gnnv3_<RUN>):  RUN=b bash scripts/run_gnn_v3_train.sh [--resume]
# The config's device_id is the physical GPU. Log: outputs/11_13_2026_output/gnn_v3_campaign/train_<RUN>.log
set -euo pipefail
cd /data/home/matmauro/codice/PV-Finder
source venv/bin/activate
export PYTHONPATH=src
ulimit -v "${VMEM_KB:-100663296}"  # 96 GiB virtual (v2 needed 64 GiB, 48 was too tight)
: "${RUN:?}"
LOG=outputs/11_13_2026_output/gnn_v3_campaign/train_${RUN}.log
python -u -m gnn.training.train_ttva_v3 -c "configs/gnn/config_gnn_ttva_v3_${RUN}.yml" "$@" 2>&1 | tee -a "$LOG"
