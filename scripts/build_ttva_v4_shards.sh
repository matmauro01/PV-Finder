#!/bin/bash
# Build the v4 TTVA training dataset on the extended-|eta| July-2026
# re-production (data/run4_all_etas), for the GNN that sits on top of
# PV-Finder v6.
#
# Two arms, so the value of chain-like augmentation can be measured on THIS
# data rather than assumed from the v3 campaign:
#   pristine   - truth graphs as-is                (no peak-finder dependency)
#   augmented  - chain-like augmentation, p=0.7    (needs augmentation_params
#                                                   measured on the v6 chain)
#
# Both arms share the same events, the same shard boundaries and the same
# per-shard seeds, so they differ only in the augmentation.
#
# IMPORTANT: --resolution-preset is hllhc_alleta, NOT the 'hllhc' hard-coded
# in scripts/build_v3_shards.sh. At fixed truth nTracks the AMVF-truth
# residual is 13-26% wider at extended |eta| because nTracks now counts
# forward tracks with sigma(z0) up to 2.8 mm. The preset sets PV-node heights
# and edge significances, i.e. features the GAT trains on: getting it wrong
# does not crash, it just trains on mis-scaled inputs.
#
# Usage (inside tmux):
#   bash scripts/build_ttva_v4_shards.sh pristine
#   bash scripts/build_ttva_v4_shards.sh augmented outputs/<date>/ttva_gap_v6
set -u
ARM=${1:?usage: build_ttva_v4_shards.sh <pristine|augmented> [aug_params_dir]}
AUG_DIR=${2:-}
cd /data/home/matmauro/codice/PV-Finder
source venv/bin/activate

ALLHAD=data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200
FILE1=$ALLHAD/ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200_1.root
FILE2=$ALLHAD/ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200_2.root
OUT=data/run4_all_etas/ttva_graphs/v4_shards_$ARM
LOG=${LOG:-outputs/08_04_2026_output/gnn_ttva_v4/build_logs_$ARM}
mkdir -p "$OUT" "$LOG"

N_SHARDS=18
SHARD_SIZE=10000   # ~13 GB resident per shard; fits the 32 GB ulimit
VAL_SIZE=5000
MAX_JOBS=${MAX_JOBS:-6}   # 6 x ~22 GB RSS on a 500 GB shared box

# The validation shard is pristine in BOTH arms and both configs point at the
# one in v4_shards_pristine/, so a second identical copy under
# v4_shards_augmented/ is 6.9 GB of pure duplication. SKIP_VAL=1 omits it.
SKIP_VAL=${SKIP_VAL:-0}

AUG_ARGS=()
if [ "$ARM" = "augmented" ]; then
    [ -n "$AUG_DIR" ] || { echo "augmented arm needs an aug params dir"; exit 2; }
    [ -f "$AUG_DIR/augmentation_params.json" ] || {
        echo "missing $AUG_DIR/augmentation_params.json"; exit 2; }
    AUG_ARGS=(--augment-params "$AUG_DIR" --aug-prob 0.7)
elif [ "$ARM" != "pristine" ]; then
    echo "arm must be 'pristine' or 'augmented'"; exit 2
fi

build() {  # build <input> <output> <start> <n> <seed> <log>
    ( ulimit -v 33554432
      python -u -m gnn.data.root_to_graphs \
        --input "$1" --output "$2" \
        --start-event "$3" --max-events "$4" \
        --knn 20 --resolution-preset hllhc_alleta \
        --seed "$5" "${AUG_ARGS[@]+"${AUG_ARGS[@]}"}" ) > "$6" 2>&1
}

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 10; done; }

pids=()
for i in $(seq 0 $((N_SHARDS - 1))); do
    throttle
    build "$FILE1" "$OUT/train_shard_$i.pt" $((i * SHARD_SIZE)) "$SHARD_SIZE" \
        $((100 + i)) "$LOG/shard_$i.log" &
    pids+=($!)
done

if [ "$SKIP_VAL" != "1" ]; then
    throttle
    # Validation comes from file 2 and is ALWAYS pristine, in both arms, so the
    # two learning curves are read against one common yardstick.
    ( AUG_ARGS=(); build "$FILE2" "$OUT/val_shard.pt" 0 "$VAL_SIZE" 999 \
        "$LOG/val_shard.log" ) &
    pids+=($!)
else
    echo "SKIP_VAL=1: reusing v4_shards_pristine/val_shard.pt"
fi

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ "$fail" -eq 0 ]; then echo "ALL BUILDS DONE ($ARM)"; else echo "SOME BUILDS FAILED ($ARM)"; fi
du -sh "$OUT"; ls -la "$OUT"
