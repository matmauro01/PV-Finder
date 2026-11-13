#!/usr/bin/env bash
# GNN v3: s8-candidate training graphs with truth labels, on the exact events of
# the v2 truth shards (v1 r16443 train shards 00-11, r16638 train shards 00-17)
# and of the v1 validation monitor (1,500 r16443 validation entries).
#
# Usage (repo root, inside tmux):  GPU=1 PART=0 NPART=3 bash scripts/run_gnn_v3_build_chain.sh
# builds every NPART-th shard starting at PART. Peak cuts: S8 in working_points.py,
# read at run time (override with CUTS=a,w,i,h,c and a different OUT).
set -euo pipefail
cd /data/home/matmauro/codice/PV-Finder
source venv/bin/activate
export PYTHONPATH=src
ulimit -v "${VMEM_KB:-50331648}"
: "${GPU:?}" "${PART:?}" "${NPART:?}"
OUT=${OUT:-data/run4_all_etas/ttva_graphs/chain_s8_v3_20261113}
V1=data/run4_all_etas/ttva_graphs/fullmu_v1_20260924
V2=data/run4_all_etas/ttva_graphs/fullmu_v2_20261111
F=data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/ATLAS_PVFinderData_601229_e8481_s4494
mkdir -p "$OUT/logs"
jobs=("r16443:$V1/validation_monitor.entry_indices.npy:validation_monitor")
for i in $(seq -w 0 11); do jobs+=("r16443:$V1/train_shard_$i.entry_indices.npy:r16443_train_shard_$i"); done
for i in $(seq -w 0 17); do jobs+=("r16638:$V2/r16638_train_shard_$i.entry_indices.npy:r16638_train_shard_$i"); done
extra=()
if [ -n "${CUTS:-}" ]; then extra=(--cuts "$CUTS"); fi
for k in "${!jobs[@]}"; do
  if (( k % NPART != PART )); then continue; fi
  IFS=: read -r tag entries name <<< "${jobs[$k]}"
  if [ -f "$OUT/$name.json" ]; then echo "skip $name"; continue; fi
  python -u -m gnn.data.chain_training_graphs --root "${F}_${tag}_PU200.root" \
      --entries "$entries" --out "$OUT/$name.pt" -d "$GPU" ${extra[@]+"${extra[@]}"} \
      > "$OUT/logs/$name.log" 2>&1
  echo "done $name $(date -u +%H:%M:%S)"
done
echo "PART $PART finished"
