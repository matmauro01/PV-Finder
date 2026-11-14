#!/usr/bin/env bash
# GNN v3 campaign, 14 Nov: two-threshold grid for the BCE references v2 e527 and a (top-2 inference + metrics
# with b's t_HS grid + summary), for the preliminary comparison. Usage (tmux):  CUDA_VISIBLE_DEVICES=0 bash scripts/run_gnn_v3_two_threshold_v2a.sh
set -euo pipefail
cd /data/home/matmauro/codice/PV-Finder
: "${CUDA_VISIBLE_DEVICES:?}"
E=outputs/11_13_2026_output/gnn_v3_campaign/eval
source venv/bin/activate
export PYTHONPATH=src
(ulimit -v 67108864; python -u -m gnn.evaluation.chain_inference_v3 \
    --root data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root \
    --entries data/run4_all_etas/ttva_graphs/fullmu_v1_20260924/validation_entries.npy --candidates s8 --top2 \
    --gnn v2e527=model_weights/ttva_gnn_hllhc_fullmu_v2_20261111/ttva_gat_alleta_k20_fullmu_v2_150k_epoch_527.pyt \
          a=model_weights/ttva_gnn_v3_20261113/a/ttva_v3_a_epoch_516.pt --out "$E/twothr_v2a/infer" -d 0)
THS="0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 0.96 0.97 0.98 0.99"
INFER=$E/twothr_v2a/infer OUT=$E/twothr_v2a/metrics THS="$THS" bash scripts/run_gnn_v3_two_threshold.sh v2e527 &
INFER=$E/twothr_v2a/infer OUT=$E/twothr_v2a/metrics THS="$THS" bash scripts/run_gnn_v3_two_threshold.sh a &
wait
python -u scripts/gnn_v3_two_threshold_summary.py --metrics "v2e527=$E/twothr_v2a/metrics/v2e527.npz" \
    "a=$E/twothr_v2a/metrics/a.npz" --out "$E/twothr_v2a/summary" | tail -1
echo "TWOTHR v2a DONE $(date)"
