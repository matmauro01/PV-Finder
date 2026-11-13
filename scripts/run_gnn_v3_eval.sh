#!/usr/bin/env bash
# GNN v3 campaign: full-chain evaluation of finished runs on the 7,500 r16443 validation entries.
# One inference pass (s8 peaks with the working_points.py cuts, read at run time) for v2 e527 and the
# given runs, at each run's lowest monitor-loss epoch (EPOCH_<run>=k overrides); then chain_t_metrics
# (>= 2-track rule, fine t grid) per model and gnn_v3_eval_summary.
# Usage (repo root, inside tmux):  TAG=phase1_a GPU=1 bash scripts/run_gnn_v3_eval.sh a [b ...]
set -euo pipefail
cd /data/home/matmauro/codice/PV-Finder
source venv/bin/activate
export PYTHONPATH=src
ulimit -v "${VMEM_KB:-67108864}"
: "${TAG:?}" "${GPU:?}"
CAND=${CAND:-s8}
E=outputs/11_13_2026_output/gnn_v3_campaign/eval/$TAG
W=model_weights/ttva_gnn_v3_20261113
ROOT=data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root
ENTRIES=${ENTRIES:-data/run4_all_etas/ttva_graphs/fullmu_v1_20260924/validation_entries.npy}
TGRID="0 0.1 0.2 0.3 0.4 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.92 0.94 0.95 0.96 0.97 0.98 0.985 0.99 0.995"
mkdir -p "$E/metrics"
gnns=("v2e527=model_weights/ttva_gnn_hllhc_fullmu_v2_20261111/ttva_gat_alleta_k20_fullmu_v2_150k_epoch_527.pyt")
for r in "$@"; do
  var="EPOCH_$r"
  ep=${!var:-$(python -c "
import json; rows=[json.loads(l) for l in open('$W/$r/learning_curve.jsonl')]
print(min(rows, key=lambda x: x['val_loss'])['epoch'])")}
  gnns+=("$r=$W/$r/ttva_v3_${r}_epoch_${ep}.pt")
  echo "$r: epoch $ep" | tee -a "$E/checkpoints.txt"
done
python -u -m gnn.evaluation.chain_inference_v3 --root "$ROOT" --entries "$ENTRIES" --candidates "$CAND" \
    --gnn "${gnns[@]}" --out "$E/infer" -d "$GPU" 2>&1 | tee -a "$E/inference.log"
specs=()
for g in "${gnns[@]}"; do
  n=${g%%=*}
  [ -f "$E/metrics/$n.npz" ] || python -u -m gnn.evaluation.chain_t_metrics --infer "$E/infer" --gnn "$n" \
      --hs-min-tracks 2 --t-grid $TGRID --workers 15 --out "$E/metrics/$n.npz" 2>&1 | tail -2
  specs+=("$n=$E/metrics/$n.npz")
done
python -u scripts/gnn_v3_eval_summary.py --metrics "${specs[@]}" --out "$E/summary" 2>&1 | tail -3
echo "EVAL $TAG DONE"
