#!/usr/bin/env bash
# GNN v3 campaign: full evaluation of one phase-2 run once its DONE marker exists, on one GPU:
#   1. single t (scripts/run_gnn_v3_eval.sh, lowest monitor-loss epoch) + summary with c;
#   2. top-2 inference + two-threshold grid (c's) + two-threshold summary with c.
# Usage (repo root, inside tmux):  CUDA_VISIBLE_DEVICES=0 bash scripts/run_gnn_v3_phase2_eval_chain.sh p2c
# (the GPU is selected by CUDA_VISIBLE_DEVICES; inside the scripts it is device 0)
set -euo pipefail
cd /data/home/matmauro/codice/PV-Finder
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU}"
r=$1
E=outputs/11_13_2026_output/gnn_v3_campaign/eval
W=model_weights/ttva_gnn_v3_20261113
until [ -f "$W/$r/DONE" ]; do sleep 120; done
echo "$r finished training: $(date)"
TAG=phase2_$r GPU=0 bash scripts/run_gnn_v3_eval.sh "$r"
source venv/bin/activate
export PYTHONPATH=src
python -u scripts/gnn_v3_eval_summary.py --metrics "v2e527=$E/phase1_abc/metrics/v2e527.npz" \
    "c=$E/phase1_abc/metrics/c.npz" "$r=$E/phase2_$r/metrics/$r.npz" --out "$E/phase2_$r/summary_vs_c" | tail -1
ep=$(head -1 "$E/phase2_$r/checkpoints.txt" | awk '{print $3}')
(ulimit -v 67108864; python -u -m gnn.evaluation.chain_inference_v3 \
    --root data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root \
    --entries data/run4_all_etas/ttva_graphs/fullmu_v1_20260924/validation_entries.npy --candidates s8 --top2 \
    --gnn "$r=$W/$r/ttva_v3_${r}_epoch_${ep}.pt" --out "$E/twothr_$r/infer" -d 0)
INFER=$E/twothr_$r/infer OUT=$E/twothr_$r/metrics THS="0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95" \
    bash scripts/run_gnn_v3_two_threshold.sh "$r"
python -u scripts/gnn_v3_two_threshold_summary.py --metrics "c=$E/twothr_bc/metrics/c.npz" \
    "$r=$E/twothr_$r/metrics/$r.npz" --out "$E/twothr_$r/summary_vs_c" | tail -1
echo "CHAIN $r DONE $(date)"
