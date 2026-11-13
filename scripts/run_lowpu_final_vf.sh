#!/usr/bin/env bash
# Low pile-up production recompute (R1, R3): vertex finding at the frozen
# `lowpu` working point (PVF-e400, integral 0.50, centroid +-3, 0.17 mm).
#
# Samples and events (identical to the April 2026 evaluations):
#   mc   : Run 2 MC, the 2,550 h5 test events 48450-50999 (ROOT entry qibin[i])
#   run2 : Run 2 data 2018, file _000002, the first 2,500 events
#   run3 : Run 3 data, file_3.root, the first 2,500 events passing the loader
#
# Every working-point value comes from --regime lowpu (no overrides).
# Usage (repo root, inside tmux):  SAMPLE=mc DEVICE=2 bash scripts/run_lowpu_final_vf.sh
set -euo pipefail
cd /data/home/matmauro/codice/PV-Finder
source venv/bin/activate
export PYTHONPATH=src
ulimit -v "${VMEM_KB:-33554432}"
: "${SAMPLE:?mc, run2 or run3}"
DEVICE=${DEVICE:-0}
BASE=outputs/11_12_2026_output/lowpu_final
OUT=$BASE/vf_${SAMPLE}
EVAL=src/pv_finder/evaluation/vertex_finding
RUN2=data/run2/Run2_Data/user.rgarg.data18_13TeV.00364076.physics_ZeroBias.AOD1_EXT0/user.rgarg.49035490.EXT0._000002.ATLAS_PVFinderData_Run3Data.root
mkdir -p "$OUT"
case "$SAMPLE" in
  mc)
    python -u $EVAL/run_eval_pvf.py --regime lowpu --device "$DEVICE" \
        --indices configs/test_main_indices_2550evt.p \
        --output-dir "$OUT" --dataset-name "Run 2 MC" ;;
  run2)
    python -u $EVAL/run_eval_pvf_run3.py --regime lowpu --device "$DEVICE" \
        --root "$RUN2" --max-events 2500 --mu-min 20 --mu-max 35 \
        --save-histograms --output-dir "$OUT" --dataset-name "Run 2 data" ;;
  run3)
    python -u $EVAL/run_eval_pvf_run3.py --regime lowpu --device "$DEVICE" \
        --root data/run3/file_3.root --max-events 2500 --mu-min 50 --mu-max 65 \
        --save-histograms --output-dir "$OUT" --dataset-name "Run 3 data" ;;
  *) echo "unknown SAMPLE $SAMPLE"; exit 1 ;;
esac 2>&1 | tee "$OUT/eval.log"
