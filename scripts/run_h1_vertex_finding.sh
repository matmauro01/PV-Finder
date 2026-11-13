#!/usr/bin/env bash
# H1/H2 driver: HL-LHC pre-GNN vertex-finding numbers and figures at the
# frozen working point (regime hllhc, s8), on both held-out PU200 files.
#
# Needs run_eval_pvf_run3.py --regime hllhc --save-histograms evaluations in
# $EVAL_BASE/<tag>/ (eval_results.pkl). Everything below reads saved outputs;
# no inference. Run from the repo root inside tmux:
#   EVAL_BASE=outputs/11_12_2026_output/hllhc_final/pre_gnn/s8 \
#   OUT=outputs/11_12_2026_output/hllhc_final/h1_vertex_finding \
#   GNN_CHUNKS=outputs/11_12_2026_output/gnn_v2_freeze/infer/test_s8 \
#   bash scripts/run_h1_vertex_finding.sh
set -euo pipefail
cd /data/home/matmauro/codice/PV-Finder
source venv/bin/activate
export PYTHONPATH=src
ulimit -v 33554432

EVAL_BASE=${EVAL_BASE:?set EVAL_BASE}
OUT=${OUT:?set OUT}
GNN_CHUNKS=${GNN_CHUNKS:-}
TAGS=${TAGS:-"r16443 r16638"}
ROOT_DIR=data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026
CKPT=$(python -c "from pv_finder.utils.working_points import get_regime; print(get_regime('hllhc').density.checkpoint)")
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/GIT_HEAD.txt"

h1=(--out "$OUT")
split=()
for tag in $TAGS; do
    root="$ROOT_DIR/ATLAS_PVFinderData_601229_e8481_s4494_${tag}_PU200.root"
    ev="$EVAL_BASE/$tag"
    echo "=== $tag: paired comparison (Eq. 1, fakes, HS, sigma, p68)"
    python -u scripts/freeze_candidates_compare.py --tag "$tag" --root "$root" \
        --ref s8 --model "s8=$ev" --out "$OUT/compare_$tag"
    python -u scripts/freeze_candidates_plots.py --tag "$tag" \
        --compare "$OUT/compare_$tag" --root "$root" --model "s8=$ev"
    echo "=== $tag: fair-comparison cache, one-to-one and window scan"
    python -u src/pv_finder/diagnostics/amvf_fair_comparison/run_fair_comparison.py \
        --build-cache --pkl "$ev/eval_results.pkl" --root "$root" \
        --cache "$OUT/fair_$tag/cache_$tag.npz" --outdir "$OUT/fair_$tag" --tag "$tag" \
        --w-min 0.10 --w-max 1.00 --w-step 0.01 --sensitivity-windows 0.17 0.1 0.5
    split+=(--cache "$tag=$OUT/fair_$tag/cache_$tag.npz")
    h1+=(--eval "$tag=$ev" --root "$tag=$root" --fair "$tag=$OUT/fair_$tag"
         --compare "$tag=$OUT/compare_$tag")
done
echo "=== split-inclusive efficiency"
python -u scripts/split_inclusive_efficiency.py --outdir "$OUT/split_inclusive" \
    "${split[@]}" --hs-events none --checkpoint "$CKPT"
if [[ -n "$GNN_CHUNKS" ]]; then h1+=(--gnn-chunks "r16443=$GNN_CHUNKS"); fi
echo "=== figures, per-mu numbers, peak-consistency gate"
python -u scripts/h1_vertex_finding.py "${h1[@]}"
python -u src/pv_finder/diagnostics/paper_figures/make_paper_figures.py \
    --out-dir "$OUT/plots/paper_figures_r16443" --sections hllhc --skip-diagrams \
    --hllhc-pkl "$EVAL_BASE/r16443/eval_results.pkl" \
    --hllhc-cache "$OUT/fair_r16443/cache_r16443.npz"
echo "=== summary, Table 3"
python -u scripts/h1_summary.py --out "$OUT"
echo "=== done"
