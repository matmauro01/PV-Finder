#!/usr/bin/env bash
# Step 2b (GNN v2 freeze): full-chain inference on r16443, both density candidates.
#
# Sets (scripts/gnn_v2_freeze_hygiene.py proves both are outside GNN v2 training):
#   test : r16443 entries 0-24,999 (the note's sample; never GNN train/val/test)
#   val  : the 7,500 GNN validation entries (t scan)
# The 7,500 GNN reserved-test entries are not touched.
#
# Usage (repo root, inside tmux):  SET=test DENSITY=v6 DEVICE=0 bash scripts/run_gnn_v2_freeze_inference.sh
set -euo pipefail
cd /data/home/matmauro/codice/PV-Finder
source venv/bin/activate
export PYTHONPATH=src
ulimit -v "${VMEM_KB:-33554432}"
: "${SET:?test or val}" "${DENSITY:?v6 or s8}"
DEVICE=${DEVICE:-0}
OUT=outputs/11_12_2026_output/gnn_v2_freeze/infer/${SET}_${DENSITY}
ROOT=data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root
V2=model_weights/ttva_gnn_hllhc_fullmu_v2_20261111/ttva_gat_alleta_k20_fullmu_v2_150k
V1=model_weights/ttva_gnn_hllhc_fullmu_v1_20260924/ttva_gat_alleta_k20_fullmu_v1_60k
case "$SET" in
  test) ENTRIES=0:25000 ;;
  val) ENTRIES=data/run4_all_etas/ttva_graphs/fullmu_v1_20260924/validation_entries.npy ;;
  *) echo "unknown SET $SET"; exit 1 ;;
esac
mkdir -p "$OUT"
python -u -m gnn.evaluation.chain_inference --root "$ROOT" --entries "$ENTRIES" \
    --density "$DENSITY" --save-histograms -d "$DEVICE" --out "$OUT" \
    --gnn "v2e527=${V2}_epoch_527.pyt" "v2e539=${V2}_epoch_539.pyt" "v1e213=${V1}_epoch_213.pyt" \
    2>&1 | tee -a "$OUT/inference.log"
