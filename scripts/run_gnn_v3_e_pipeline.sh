#!/usr/bin/env bash
# GNN v3 campaign, control run e (truth-candidate training, c's recipe): wait for run d to finish
# (GPU 0), train e, then evaluate it exactly as c on s8 candidates:
#   1. single t: scripts/run_gnn_v3_eval.sh e (lowest monitor-loss epoch), summary with c (matched points);
#   2. two-threshold: chain_inference_v3 --top2, run_gnn_v3_two_threshold.sh (c's grid), summary with c.
# Usage (repo root, inside tmux gnnv3_e):  bash scripts/run_gnn_v3_e_pipeline.sh
set -euo pipefail
cd /data/home/matmauro/codice/PV-Finder
C=outputs/11_13_2026_output/gnn_v3_campaign
E=$C/eval
W=model_weights/ttva_gnn_v3_20261113
until [ -f $W/d/DONE ]; do sleep 300; done
echo "d finished, training e: $(date)"
[ -f $W/e/DONE ] || RUN=e bash scripts/run_gnn_v3_train.sh $( [ -f $W/e/last_fullstate.pth ] && echo --resume )
echo "e trained: $(date)"
TAG=phase1_e GPU=0 bash scripts/run_gnn_v3_eval.sh e
source venv/bin/activate
export PYTHONPATH=src
python -u scripts/gnn_v3_eval_summary.py --metrics "v2e527=$E/phase1_abc/metrics/v2e527.npz" \
    "c=$E/phase1_abc/metrics/c.npz" "e=$E/phase1_e/metrics/e.npz" --out "$E/phase1_ce/summary" | tail -2
ep=$(python -c "
import json; rows=[json.loads(l) for l in open('$W/e/learning_curve.jsonl')]
print(min(rows, key=lambda x: x['val_loss'])['epoch'])")
(ulimit -v 67108864; python -u -m gnn.evaluation.chain_inference_v3 \
    --root data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root \
    --entries data/run4_all_etas/ttva_graphs/fullmu_v1_20260924/validation_entries.npy --candidates s8 --top2 \
    --gnn "e=$W/e/ttva_v3_e_epoch_${ep}.pt" --out "$E/twothr_e/infer" -d 0)
INFER=$E/twothr_e/infer OUT=$E/twothr_e/metrics THS="0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95" \
    bash scripts/run_gnn_v3_two_threshold.sh e
python -u scripts/gnn_v3_two_threshold_summary.py --metrics "c=$E/twothr_bc/metrics/c.npz" \
    "e=$E/twothr_e/metrics/e.npz" --out "$E/twothr_ce/summary" | tail -2
echo "PIPELINE e DONE $(date)"
