#!/bin/bash
# Rebuild train shards 0-7 (killed mid-torch.save on the first attempt).
# Same seeds -> identical content. Caps at 5 concurrent so the still-running
# shard_8/val/test jobs keep the total at <= 8 builders.
set -u
cd /data/home/matmauro/codice/PV-Finder
source venv/bin/activate

ALLHAD=data/run4/PU200_withTiming/ATLAS_PVFinderData_601237e8481_s4494_r16633_PU200/ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200_1.root
OUT=data/run4/ttva_graphs/v3_shards
AUG=outputs/07_14_2026_ttva_gap
LOG=outputs/07_14_2026_ttva_v3/build_logs

pids=()
for i in 0 1 2 3 4 5 6 7; do
    while [ "$(jobs -rp | wc -l)" -ge 5 ]; do sleep 15; done
    python -u -m gnn.data.root_to_graphs \
        --input "$ALLHAD" --output "$OUT/train_shard_$i.pt" \
        --start-event $((i * 20000)) --max-events 20000 \
        --resolution-preset hllhc --seed $((100 + i)) \
        --augment-params "$AUG" --aug-prob 0.7 \
        > "$LOG/shard_${i}_rebuild.log" 2>&1 &
    pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] && echo "REBUILD DONE" || echo "REBUILD FAILED"
