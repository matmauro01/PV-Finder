#!/usr/bin/env bash
# Held-out evaluation of the CPU small-model candidates (Nov 2026).
#
# Evaluates one checkpoint on both held-out flat-mu files (r16443, r16638) at
# the HL-LHC working point (--regime hllhc, pv_finder.utils.working_points:
# 0.17 mm window, sigma fitted on the counted peaks; the explicit peak flags
# below equal the regime and are kept for readability), mu in [185, 215],
# first 25000 entries. Before 11 Nov 2026 this ran at 0.1694 mm with the sigma
# fit at integral 0.50. Run production v6 through this same
# script so every model is measured with the same code version, then compare
# with scripts/compare_heldout_evals.py (paired, per event).
#
# Usage, from the repo root inside tmux:
#   NAME=v6 CKPT=model_weights/hllhc_alleta_v6_mse_2ep_phase2_epoch_2_fullstate.pth \
#     WIDTH=280 LATENT=4 DEVICE=1 bash scripts/eval_small_models_heldout.sh
set -euo pipefail
cd /data/home/matmauro/codice/PV-Finder
source venv/bin/activate
ulimit -v 33554432

: "${NAME:?}" "${CKPT:?}" "${WIDTH:?}" "${LATENT:?}"
DEVICE=${DEVICE:-1}
HIDDEN=${HIDDEN:-128}
OUT=${OUT:-outputs/11_09_2026_output/small_model_eval}
ROOT_DIR=data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026

for tag in r16443 r16638; do
    outdir="$OUT/$NAME/$tag"
    mkdir -p "$outdir"
    cmd=(python -u src/pv_finder/evaluation/vertex_finding/run_eval_pvf_run3.py --regime hllhc
        --root "$ROOT_DIR/ATLAS_PVFinderData_601229_e8481_s4494_${tag}_PU200.root"
        --e2e-model "$CKPT" --e2e-type v2
        --e2e-unet-channels "$WIDTH" --e2e-latent-channels "$LATENT"
        --e2e-hidden "$HIDDEN" "$HIDDEN" "$HIDDEN" "$HIDDEN" "$HIDDEN"
        --max-events 25000 --mu-min 185 --mu-max 215
        --peak-threshold 0.01 --integral-threshold 0.30 --min-height 0.03
        --centroid-halfwidth 3 --pairwise-bins 300
        --output-dir "$outdir" --device "$DEVICE"
        --dataset-name "HL-LHC PU200 held-out ($tag)")
    { printf '%q ' "${cmd[@]}"; echo; echo "# git $(git rev-parse HEAD)"; echo "# eval-code diff sha1 $(git diff HEAD -- src/pv_finder/evaluation | sha1sum | cut -c1-12)"; } > "$outdir/INVOCATION.txt"
    "${cmd[@]}" 2>&1 | tee "$outdir/eval.log" | grep -E "Eff=|Error|Traceback" || true
done
