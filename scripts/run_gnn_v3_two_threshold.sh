#!/usr/bin/env bash
# GNN v3 campaign: two-threshold working points (chain_t_metrics --two-threshold) for one model of a
# top-2 inference directory (chain_inference_v3 --top2). Global t in [0, 0.5]; t_HS grid per model
# (BCE runs need t_HS up to 0.99, their scores sit near 1).
# Usage (repo root, inside tmux):  INFER=<dir> OUT=<dir> THS="0.5 ... 0.95" bash scripts/run_gnn_v3_two_threshold.sh c
set -euo pipefail
cd /data/home/matmauro/codice/PV-Finder
source venv/bin/activate
export PYTHONPATH=src
ulimit -v "${VMEM_KB:-67108864}"
: "${INFER:?}" "${OUT:?}" "${THS:?}"
m=$1
mkdir -p "$OUT"
python -u -m gnn.evaluation.chain_t_metrics --infer "$INFER" --gnn "$m" --hs-min-tracks 2 \
    --two-threshold --t-grid 0 0.1 0.2 0.3 0.4 0.5 --t-hs-grid $THS \
    --workers "${WORKERS:-15}" --out "$OUT/$m.npz" 2>&1 | tail -2
echo "TWOTHR $m DONE"
