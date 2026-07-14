#!/bin/bash
# Build the v3 TTVA training dataset (chain-like augmented graphs).
# 9 train shards x 20k events (all-hadronic 601237 r16633 file 1),
# 1 augmented val shard (file 2), 1 unaugmented fixed-height truth test
# rebuild (SingleLep entries 28500+, matching the v1/v2 test slice).
# Run inside tmux: bash scripts/build_v3_shards.sh
set -u
cd /data/home/matmauro/codice/PV-Finder
source venv/bin/activate

ALLHAD_DIR=data/run4/PU200_withTiming/ATLAS_PVFinderData_601237e8481_s4494_r16633_PU200
SINGLELEP=data/run4/Run4_MC21_ITk/ATLAS_PVFinderData_HLLHC_mc21_14TeV_ttbar_SingleLep_PU200.root
OUT=data/run4/ttva_graphs/v3_shards
AUG=outputs/07_14_2026_ttva_gap
LOG=outputs/07_14_2026_ttva_v3/build_logs
mkdir -p "$OUT" "$LOG"

build() {  # build <input> <output> <start> <n> <seed> <extra...>
    local input=$1 output=$2 start=$3 n=$4 seed=$5
    shift 5
    python -u -m gnn.data.root_to_graphs \
        --input "$input" --output "$output" \
        --start-event "$start" --max-events "$n" \
        --resolution-preset hllhc --seed "$seed" "$@"
}

pids=()
throttle() {  # keep at most 8 concurrent jobs
    while [ "$(jobs -rp | wc -l)" -ge 8 ]; do sleep 10; done
}

for i in 0 1 2 3 4 5 6 7 8; do
    throttle
    build "$ALLHAD_DIR/ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200_1.root" \
        "$OUT/train_shard_$i.pt" $((i * 20000)) 20000 $((100 + i)) \
        --augment-params "$AUG" --aug-prob 0.7 \
        > "$LOG/shard_$i.log" 2>&1 &
    pids+=($!)
done

throttle
build "$ALLHAD_DIR/ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200_2.root" \
    "$OUT/val_shard.pt" 0 5000 999 \
    --augment-params "$AUG" --aug-prob 0.7 \
    > "$LOG/val_shard.log" 2>&1 &
pids+=($!)

throttle
build "$SINGLELEP" \
    data/run4/ttva_graphs/pu200_truth_k20_test_fixedheights.pt 28500 1500 1 \
    > "$LOG/test_fixedheights.log" 2>&1 &
pids+=($!)

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] && echo "ALL BUILDS DONE" || echo "SOME BUILDS FAILED"
ls -la "$OUT"
