#!/usr/bin/env bash
# GNN v3 phase 2 training graphs (s8 candidates, truth labels):
#   r16438 (PU200, 601229 SingleLep) thinned to flat mu (pile-up count U[0, mu]), 20 shards;
#   r16633 (PU200, 601229 SingleLep) unthinned, 20 shards;
#   r16443 validation entries (7,500, reference for the peak-quality check, never trained on).
# s8 was trained on PU200 events from these PU200 files (accepted for training graphs; see the
# peak-quality check in docs/evaluation/gnn_v3_campaign_2026-11-13.md).
# Usage (repo root, inside tmux):  GPU=1 PART=0 NPART=2 bash scripts/run_gnn_v3_build_phase2.sh
set -euo pipefail
cd /data/home/matmauro/codice/PV-Finder
source venv/bin/activate
export PYTHONPATH=src
ulimit -v "${VMEM_KB:-50331648}"
: "${GPU:?}" "${PART:?}" "${NPART:?}"
OUT=${OUT:-data/run4_all_etas/ttva_graphs/chain_s8_v3_20261113/phase2}
F=data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/ATLAS_PVFinderData_601229_e8481_s4494
mkdir -p "$OUT/logs"
jobs=("r16443:$OUT/entries/r16443_validation_all.npy:r16443_validation_all:")
for i in $(seq -w 0 19); do
  jobs+=("r16438:$OUT/entries/r16438_$i.npy:r16438_thin_shard_$i:$((20261113 + 10#$i))")
  jobs+=("r16633:$OUT/entries/r16633_$i.npy:r16633_pu200_shard_$i:")
done
for k in "${!jobs[@]}"; do
  if (( k % NPART != PART )); then continue; fi
  IFS=: read -r tag entries name seed <<< "${jobs[$k]}"
  if [ -f "$OUT/$name.json" ]; then echo "skip $name"; continue; fi
  extra=()
  if [ -n "$seed" ]; then extra=(--thin-seed "$seed"); fi
  python -u -m gnn.data.chain_training_graphs --root "${F}_${tag}_PU200.root" \
      --entries "$entries" --out "$OUT/$name.pt" -d "$GPU" ${extra[@]+"${extra[@]}"} \
      > "$OUT/logs/$name.log" 2>&1
  echo "done $name $(date -u +%H:%M:%S)"
done
echo "PART $PART finished"
