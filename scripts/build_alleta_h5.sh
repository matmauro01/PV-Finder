#!/usr/bin/env bash
# Convert the extended-|eta| PU200 ROOTs to the training HDF5 pool.
#
# Differences from the |eta|<2.5 build (scripts/train_v5_corrected.sh):
#   * --max-tracks-per-sub 1536, not the 1024 default. A full z0 scan over
#     6.43 M sub-events of the new sample gives a max of 1066, and truncation
#     is z0-ordered: it drops the highest-z0 tracks while the target histogram
#     keeps their PVs. See docs/data/run_4_all_etas.md.
#   * A resolution preset refit on THIS sample. At fixed truth nTracks the
#     achievable resolution is 1.18-1.74x worse than at |eta|<2.5, because
#     nTracks now counts low-information forward tracks, so neither `hllhc`
#     nor `hllhc_corrected` describes it.
#   * Only the six nominal-mu=200 files. r16443 and r16638 carry a flat
#     mu ~0-210 spectrum and are NOT PU200 (mean truth PV/event 69.9 vs 139.5).
#
# Usage, from the repo root inside tmux:
#     PRESET=hllhc_alleta bash scripts/build_alleta_h5.sh
set -u
cd /data/home/matmauro/codice/PV-Finder || exit 1
source venv/bin/activate

PRESET=${PRESET:?set PRESET to the resolution preset registered for this sample}
MAX_TRACKS=${MAX_TRACKS:-1536}
CONCURRENCY=${CONCURRENCY:-6}

SRC=data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026
SRC4=$SRC/ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200
OUT=data/run4_all_etas/PU200_alleta_h5
LOG=outputs/07_31_2026_output/convert_alleta
mkdir -p "$OUT" "$LOG"

# Nominal mu=200 only. 5,173,600 events total.
INPUTS=(
    "$SRC/ATLAS_PVFinderData_601229_e8481_s4494_r16438_PU200.root"
    "$SRC/ATLAS_PVFinderData_601229_e8481_s4494_r16633_PU200.root"
    "$SRC4/ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200_1.root"
    "$SRC4/ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200_2.root"
    "$SRC4/ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200_3.root"
    "$SRC4/ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200_4.root"
)

convert() {
    local in=$1
    local name out
    name=$(basename "$in" .root)
    out="$OUT/$name.h5"
    if [ -f "$out" ]; then echo "[$(date +%H:%M:%S)] SKIP  $name (exists)"; return 0; fi
    echo "[$(date +%H:%M:%S)] START $name"
    # Write to .tmp and rename, so a killed job never leaves a truncated .h5
    # that a later run would "SKIP (exists)".
    if python -u src/pv_finder/data/root_to_h5.py \
        --input "$in" --output "$out.tmp" \
        --resolution-preset "$PRESET" \
        --max-tracks-per-sub "$MAX_TRACKS" --max-pv 300 \
        --compression lzf --max-events 0 \
        > "$LOG/convert_$name.log" 2>&1
    then
        mv "$out.tmp" "$out"
        echo "[$(date +%H:%M:%S)] DONE  $name"
    else
        echo "[$(date +%H:%M:%S)] FAIL  $name -- see $LOG/convert_$name.log"
        rm -f "$out.tmp"
        return 1
    fi
}

echo "preset=$PRESET  max_tracks=$MAX_TRACKS  concurrency=$CONCURRENCY"
running=0
for in in "${INPUTS[@]}"; do
    convert "$in" &
    running=$((running + 1))
    if [ "$running" -ge "$CONCURRENCY" ]; then wait -n; running=$((running - 1)); fi
done
wait

echo
echo "=== truncation warnings (must be empty) ==="
grep -h "WARNING" "$LOG"/convert_*.log || echo "  none"

echo
echo "=== verifying the pool ==="
python -u - "$OUT" <<'PY'
import glob
import os
import sys

import h5py

pool = sorted(glob.glob(os.path.join(sys.argv[1], "*.h5")))
total, bad = 0, []
attrs_seen = set()
for path in pool:
    name = os.path.basename(path)
    try:
        with h5py.File(path, "r") as fh:
            n_sub = fh["tracks"].shape[0]
            width = fh["tracks"].shape[-1]
            res = (
                round(float(fh.attrs["resolution_a_mm"]), 6),
                round(float(fh.attrs["resolution_b"]), 6),
                round(float(fh.attrs["resolution_c_mm"]), 6),
            )
            attrs_seen.add(res)
            assert fh["target_y_split"].shape[0] == n_sub, "tracks/target mismatch"
    except Exception as exc:
        bad.append(name)
        print(f"  FAIL {name:62s} {type(exc).__name__}: {exc}")
        continue
    total += n_sub
    print(
        f"  ok   {name:62s} {n_sub // 12:>9,} evts  "
        f"width={width}  ABC={res}"
    )

print(f"\n{len(pool)} files, {total:,} sub-events ({total // 12:,} events)")
if len(attrs_seen) > 1:
    print(f"ERROR: mixed resolution presets in the pool: {attrs_seen}")
    sys.exit(1)
if bad:
    print(f"UNREADABLE ({len(bad)}): " + ", ".join(bad))
    sys.exit(1)
print("Pool is consistent and readable.")
PY
