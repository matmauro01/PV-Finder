#!/usr/bin/env bash
# Step 2a driver: matched-fake-rate scan, paired comparison and plots for the
# HL-LHC freeze candidates, on both held-out files.
#
# Needs the evaluations from scripts/eval_small_models_heldout.sh run with
# EXTRA=--save-histograms into $BASE/<model>/<file>/. v6 is the reference.
#
# Usage, from the repo root inside tmux:
#   MODELS="s5 s8" bash scripts/run_freeze_compare.sh
#   MODELS="s5 s8 s5_2p2" bash scripts/run_freeze_compare.sh   # once s5 2+2 is in
set -euo pipefail
cd /data/home/matmauro/codice/PV-Finder
source venv/bin/activate
ulimit -v 67108864

BASE=${BASE:-outputs/11_12_2026_output/freeze_candidates}
MODELS=${MODELS:-"s5 s8"}
ROOT_DIR=data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026

for tag in r16443 r16638; do
    root="$ROOT_DIR/ATLAS_PVFinderData_601229_e8481_s4494_${tag}_PU200.root"
    mflags=()
    cflags=(--model "v6=$BASE/v6/$tag")
    pflags=(--model "v6=$BASE/v6/$tag")
    for m in $MODELS; do
        mflags+=(--model "$m=$BASE/$m/$tag")
        cflags+=(--model "$m=$BASE/$m/$tag")
        pflags+=(--model "$m=$BASE/$m/$tag")
    done
    echo "=== $tag: matched fake rate"
    python -u scripts/matched_fp_scan.py --ref "v6=$BASE/v6/$tag" "${mflags[@]}" \
        --out "$BASE/matched_$tag.json"
    echo "=== $tag: paired comparison"
    python -u scripts/freeze_candidates_compare.py --tag "$tag" --root "$root" \
        --ref v6 "${cflags[@]}" --matched "$BASE/matched_$tag.json" \
        --out "$BASE/compare_$tag"
    echo "=== $tag: plots"
    python -u scripts/freeze_candidates_plots.py --tag "$tag" \
        --compare "$BASE/compare_$tag" --root "$root" "${pflags[@]}"
    git rev-parse HEAD > "$BASE/compare_$tag/GIT_HEAD.txt"
done
echo "=== done"
