#!/usr/bin/env bash
# s8 at the frozen hllhc working point on r16443 entries 25,000-49,999 (off the
# 0-24,999 test set), with histograms saved for the offline peak-finding replay.
# Only the GNN-validation entries in this range are used by the study.
set -euo pipefail
cd /data/home/matmauro/codice/PV-Finder
source venv/bin/activate
ulimit -v 33554432
OUT=outputs/11_13_2026_output/peak_finding/eval_s8_r16443_e25000_50000
cmd=(python -u src/pv_finder/evaluation/vertex_finding/run_eval_pvf_run3.py --regime hllhc
  --root data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root
  --entry-start 25000 --entry-stop 50000 --mu-min 185 --mu-max 215
  --output-dir "$OUT" --device 3 --dataset-name "HL-LHC r16443 entries 25000-49999" --save-histograms)
{ printf '%q ' "${cmd[@]}"; echo; echo "# git $(git rev-parse HEAD)"; } > "$OUT/INVOCATION.txt"
"${cmd[@]}" > "$OUT/eval.log" 2>&1
touch "$OUT/DONE"
