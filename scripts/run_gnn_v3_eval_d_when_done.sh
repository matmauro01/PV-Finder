#!/usr/bin/env bash
# GNN v3 campaign: wait for phase-1 run d to finish training, then evaluate it exactly as a/b/c
# (scripts/run_gnn_v3_eval.sh, lowest monitor-loss epoch) and write one summary of v2 e527, a, b, c, d
# on the identical 7,500 validation events.
# Usage (repo root, inside tmux gnnv3_eval_d):  bash scripts/run_gnn_v3_eval_d_when_done.sh
set -euo pipefail
cd /data/home/matmauro/codice/PV-Finder
E=outputs/11_13_2026_output/gnn_v3_campaign/eval
until [ -f model_weights/ttva_gnn_v3_20261113/d/DONE ]; do sleep 300; done
echo "d finished: $(date)"
TAG=phase1_d GPU=0 bash scripts/run_gnn_v3_eval.sh d
source venv/bin/activate
PYTHONPATH=src python -u scripts/gnn_v3_eval_summary.py --metrics \
    "v2e527=$E/phase1_abc/metrics/v2e527.npz" "a=$E/phase1_abc/metrics/a.npz" \
    "b=$E/phase1_abc/metrics/b.npz" "c=$E/phase1_abc/metrics/c.npz" "d=$E/phase1_d/metrics/d.npz" \
    --out "$E/phase1_all/summary" 2>&1 | tail -3
echo "EVAL phase1_all DONE $(date)"
