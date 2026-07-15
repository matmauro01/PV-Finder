#!/bin/bash
# v5 corrected-widths campaign: rebuild the 8 training h5 files with the
# 'hllhc_corrected' resolution preset, then launch the v4b-recipe training.
# Run inside tmux from the repo root.
set -u
cd /data/home/matmauro/codice/PV-Finder
source venv/bin/activate

OUT=data/run4/PU200_corrected_h5
LOG=outputs/07_15_2026_pvf_v5
SRC=data/run4/PU200_withTiming
SRC6=$SRC/ATLAS_PVFinderData_601237e8481_s4494_r16633_PU200
mkdir -p "$OUT" "$LOG"

convert() {
    local in=$1 out=$2
    local name
    name=$(basename "$out" .h5)
    if [ -f "$out" ]; then echo "SKIP $out (exists)"; return 0; fi
    echo "[$(date +%H:%M:%S)] START $name"
    if python -u src/pv_finder/data/root_to_h5.py \
        --input "$in" --output "$out.tmp" \
        --resolution-preset hllhc_corrected --max-events 0 \
        > "$LOG/convert_$name.log" 2>&1; then
        mv "$out.tmp" "$out"
        echo "[$(date +%H:%M:%S)] DONE $name"
    else
        echo "[$(date +%H:%M:%S)] FAILED $name (see $LOG/convert_$name.log)"
        return 1
    fi
}

# 4 concurrent conversions, two batches
convert "$SRC/ATLAS_PVFinderData_601229_e8481_s4494_r16438_PU200.root" "$OUT/ATLAS_PVFinderData_601229_e8481_s4494_r16438_PU200.h5" &
convert "$SRC/ATLAS_PVFinderData_601229_e8481_s4494_r16633_PU200.root" "$OUT/ATLAS_PVFinderData_601229_e8481_s4494_r16633_PU200.h5" &
convert "$SRC6/ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200_1.root" "$OUT/ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200_1.h5" &
convert "$SRC6/ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200_2.root" "$OUT/ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200_2.h5" &
wait
convert "$SRC6/ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200_3.root" "$OUT/ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200_3.h5" &
convert "$SRC6/ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200_4.root" "$OUT/ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200_4.h5" &
convert "$SRC6/ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200_5.root" "$OUT/ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200_5.h5" &
convert "$SRC6/ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200_6.root" "$OUT/ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200_6.h5" &
wait

n=$(ls "$OUT"/*.h5 2>/dev/null | wc -l)
if [ "$n" -ne 8 ]; then
    echo "ABORT: only $n/8 h5 files present — not launching training"
    exit 1
fi
echo "ALL CONVERSIONS DONE ($n/8) — launching training"

python -u -m pv_finder.training.train_hllhc_e2e \
    -c configs/vertex_finding/config_hllhc_pu200_e2e_v5_corrected.yml \
    2>&1 | tee "$LOG/train.log"
echo "TRAINING EXITED (code $?)"
