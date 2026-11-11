#!/usr/bin/env bash
# Old-vs-new held-out comparison under ONE methodology (2026-08-05).
#
# Why this script exists
# ----------------------
# The deck wanted "the previous training run on the older data" beside the
# current one. The number that was circulating for v4b -- Eff 0.9327, FP 12.82,
# sigma 0.2900 -- is not usable for that:
#   * it was fitted at 60 pairwise bins, which JOURNAL 2026-07-20 identified as
#     a binning artefact that biases sigma high (and the matching window with
#     it, since sigma is fed back as the window);
#   * it was measured on r16438, which is INSIDE v4b's own training pool
#     (config_hllhc_pu200_e2e_v4b_stepwarmup.yml: r16438 + r16633 + 601237).
#
# r16443 is in NEITHER model's training pool -- v4b trained on the old
# production of r16438/r16633/601237, v6 on the all-|eta| production of the same
# three tags -- so the same 1920 mu-matched events of r16443 are genuinely
# held out for both. That is the cell this script fills in.
#
# The three cells
# ---------------
#   v4b_old   v4b on the |eta| < 2.5 production   -- old model on its own data
#   v4b_new   v4b on the all-|eta| production     -- isolates the DATA axis
#   (v6_new is not re-run: outputs/08_05_2026_output/eval_v6_mu_window/r16443)
#
# v4b_new is deliberately out of domain for v4b and is labelled as such; it is
# here only so the old->new change can be split into a data part and a model
# part, the same decomposition docs/research/pairwise_dz_bump.md makes for the
# satellite rate.
#
# Every cell uses the CURRENT methodology, identical to eval_v6_operating_point:
# integral 0.30, min-height 0.03, centroid half-width 3, 300 commensurate
# pairwise bins, sigma fitted on the mu window. Nothing here is tuned per model.
#
# The samples are NOT the same size: the all-|eta| re-production has ~111 truth
# PVs/event against ~99, and the added vertices are the hardest in the sample.
# Efficiencies across the two productions are therefore not like-for-like and
# must never be subtracted. See docs/evaluation/vertex_finding.md.
#
# Usage, from the repo root inside tmux:
#     DEVICE=2 bash scripts/eval_v4b_vs_v6_heldout.sh
set -u
cd /data/home/matmauro/codice/PV-Finder || exit 1
source venv/bin/activate

DEVICE=${DEVICE:-2}
MAX_EVENTS=${MAX_EVENTS:-25000}
OUT=${OUT:-outputs/08_05_2026_output/deck_update/eval_v4b_heldout}

V4B=model_weights/hllhc_pu200_e2e_v4b_3ep_280ch_4lat_stepwarmup_phase2_epoch_3_fullstate.pth
ROOT_OLD=data/run4/PU200_withTiming/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root
ROOT_NEW=data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root

run_cell () {
    local name=$1 root=$2 label=$3
    local outdir="$OUT/$name"
    mkdir -p "$outdir"

    local cmd=(python -u src/pv_finder/evaluation/vertex_finding/run_eval_pvf_run3.py
        --root "$root"
        --e2e-model "$V4B"
        --e2e-type v2
        --e2e-unet-channels 280
        --e2e-latent-channels 4
        --e2e-hidden 128 128 128 128 128
        --max-events "$MAX_EVENTS"
        --mu-min 185 --mu-max 215
        --peak-threshold 0.01
        --integral-threshold 0.30
        --min-height 0.03
        --centroid-halfwidth 3
        --pairwise-bins 300
        --output-dir "$outdir"
        --device "$DEVICE"
        --dataset-name "$label")

    printf '%q ' "${cmd[@]}" > "$outdir/INVOCATION.txt"
    printf '\n# git %s\n' "$(git rev-parse HEAD)" >> "$outdir/INVOCATION.txt"

    echo "[$(date +%H:%M:%S)] START $name -> $outdir"
    if "${cmd[@]}" > "$outdir/eval.log" 2>&1; then
        echo "[$(date +%H:%M:%S)] DONE  $name"
        grep -E "Eff=" "$outdir/eval.log" | tail -1
    else
        echo "[$(date +%H:%M:%S)] FAIL  $name -- see $outdir/eval.log"
        tail -25 "$outdir/eval.log"
    fi
}

run_cell v4b_old "$ROOT_OLD" "PU200 held-out r16443, |eta|<2.5 production (v4b, in domain)"
run_cell v4b_new "$ROOT_NEW" "PU200 held-out r16443, all-|eta| production (v4b, OUT of domain)"

echo
echo "=== summary ==="
grep -H "Eff=" "$OUT"/*/eval.log
echo "--- v6 reference (not re-run) ---"
grep -H "Eff=" outputs/08_05_2026_output/eval_v6_mu_window/r16443/eval.log
