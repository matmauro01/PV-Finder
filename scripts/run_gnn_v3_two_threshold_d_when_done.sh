#!/usr/bin/env bash
# GNN v3 campaign: after run d's standard evaluation (eval_phase1_d.log says "EVAL phase1_all DONE"),
# top-2 inference for d at its lowest monitor-loss epoch, its two-threshold metrics (c's grid; d is a
# softmax run) and one two-threshold summary of b, c, d.
# Usage (repo root, inside tmux gnnv3_twothr_d):  bash scripts/run_gnn_v3_two_threshold_d_when_done.sh
set -euo pipefail
cd /data/home/matmauro/codice/PV-Finder
C=outputs/11_13_2026_output/gnn_v3_campaign
E=$C/eval
W=model_weights/ttva_gnn_v3_20261113
until grep -q "EVAL phase1_all DONE" "$C/eval_phase1_d.log" 2>/dev/null; do sleep 300; done
source venv/bin/activate
export PYTHONPATH=src
ep=$(python -c "
import json; rows=[json.loads(l) for l in open('$W/d/learning_curve.jsonl')]
print(min(rows, key=lambda x: x['val_loss'])['epoch'])")
echo "d: epoch $ep $(date)"
(ulimit -v 67108864; python -u -m gnn.evaluation.chain_inference_v3 \
    --root data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root \
    --entries data/run4_all_etas/ttva_graphs/fullmu_v1_20260924/validation_entries.npy --candidates s8 --top2 \
    --gnn "d=$W/d/ttva_v3_d_epoch_${ep}.pt" --out "$E/twothr_d/infer" -d 0)
INFER=$E/twothr_d/infer OUT=$E/twothr_d/metrics THS="0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95" \
    bash scripts/run_gnn_v3_two_threshold.sh d
python -u scripts/gnn_v3_two_threshold_summary.py --metrics "b=$E/twothr_bc/metrics/b.npz" \
    "c=$E/twothr_bc/metrics/c.npz" "d=$E/twothr_d/metrics/d.npz" --out "$E/twothr_bcd/summary" | tail -3
echo "TWOTHR bcd DONE $(date)"
