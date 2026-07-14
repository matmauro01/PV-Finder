#!/bin/bash
# Build the mu60 v2 (augmented, fixed-height) TTVA training dataset.
# Order matches the Nov 2025 training file: training_indices_no_test.p
# positions 0-35,700 = train (2 shards), 35,700-48,450 = val. The shuffled
# test events (qibin_test_main_indices_v2.p) are excluded by construction.
# NOTE: gnn.data.h5_to_graphs hangs under `python -m` with h5+indices args
# (the documented sneezy quirk) -> invoked via a runpy wrapper.
set -u
cd /data/home/matmauro/codice/PV-Finder
source venv/bin/activate

H5=/share/lazy/qibinlei/recoTracks_incamvfassoc.h5
IDX=/data/home/matmauro/codice/atlas_pvfinder/tracks_to_vertex/training_indices_no_test.p
AUG=outputs/07_14_2026_ttva_gap/mu60_aug_params
OUT=data/mu60/ttva_graphs
LOG=outputs/07_14_2026_ttva_mu60v2/build_logs
mkdir -p "$OUT" "$LOG"

WRAP=$(mktemp /tmp/run_h5g_XXXX.py)
cat > "$WRAP" <<'PY'
import runpy, sys
sys.argv = ["h5_to_graphs"] + sys.argv[1:]
runpy.run_module("gnn.data.h5_to_graphs", run_name="__main__")
PY

build() {  # build <output> <start> <n> <seed> [extra...]
    local output=$1 start=$2 n=$3 seed=$4
    shift 4
    python -u "$WRAP" -f "$H5" -i "$IDX" \
        --start-event "$start" -n "$n" -o "$output" --seed "$seed" "$@"
}

pids=()
build "$OUT/mu60v2_train_shard_0.pt" 0 17850 300 \
    --augment-params "$AUG" --aug-prob 0.7 > "$LOG/train_0.log" 2>&1 &
pids+=($!)
build "$OUT/mu60v2_train_shard_1.pt" 17850 17850 301 \
    --augment-params "$AUG" --aug-prob 0.7 > "$LOG/train_1.log" 2>&1 &
pids+=($!)
build "$OUT/mu60v2_val_shard.pt" 35700 12750 302 \
    --augment-params "$AUG" --aug-prob 0.7 > "$LOG/val.log" 2>&1 &
pids+=($!)
# Fixed-height (unaugmented) truth TEST graphs for the associator bound
python -u "$WRAP" -f "$H5" -i configs/qibin_test_main_indices_v2.p \
    -o "$OUT/mu60_truth_test_fixedheights.pt" \
    > "$LOG/test_fixedheights.log" 2>&1 &
pids+=($!)

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
rm -f "$WRAP"
[ "$fail" -eq 0 ] && echo "MU60 BUILDS DONE" || echo "MU60 BUILDS FAILED"
ls -la "$OUT"
