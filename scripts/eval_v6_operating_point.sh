#!/usr/bin/env bash
# Held-out evaluation of the v6 finder at the chosen operating point.
#
# Operating point (docs/research/peak_position_estimator.md, selected on half A
# of r16443 and reported on half B):
#     local centroid, half-width 3 bins   (--centroid-halfwidth 3)
#     integral_threshold 0.30             (production was 0.20)
#     min_height 0.03
#
# Every argument is passed explicitly even where it matches the CLI default, and
# the full invocation is written to <output>/INVOCATION.txt. The previous
# production run recorded nothing, and its integral_threshold had to be
# reconstructed after the fact by replaying stored per-event peak lists.
#
# Reading the printed efficiency: this eval feeds the fitted sigma_vtx-vtx back
# in as the matching window, so a better sigma buys a tighter window and the
# printed efficiency drops for reasons that are not a loss of found vertices.
# Compare against the fixed-window numbers in the write-up, not only this one.
#
# Usage, from the repo root inside tmux:
#     DEVICE=3 bash scripts/eval_v6_operating_point.sh
set -u
cd /data/home/matmauro/codice/PV-Finder || exit 1
source venv/bin/activate

DEVICE=${DEVICE:-3}
MAX_EVENTS=${MAX_EVENTS:-25000}
INTEGRAL=${INTEGRAL:-0.30}
MIN_HEIGHT=${MIN_HEIGHT:-0.03}
HALFWIDTH=${HALFWIDTH:-3}

CKPT=model_weights/hllhc_alleta_v6_mse_2ep_phase2_epoch_2_fullstate.pth
SRC=data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026
OUT=outputs/08_04_2026_output/eval_v6_operating_point

for tag in r16443 r16638; do
    root="$SRC/ATLAS_PVFinderData_601229_e8481_s4494_${tag}_PU200.root"
    outdir="$OUT/$tag"
    mkdir -p "$outdir"

    cmd=(python -u src/pv_finder/evaluation/vertex_finding/run_eval_pvf_run3.py
         --root "$root"
         --e2e-model "$CKPT"
         --e2e-type v2
         --e2e-unet-channels 280
         --e2e-latent-channels 4
         --e2e-hidden 128 128 128 128 128
         --max-events "$MAX_EVENTS"
         --mu-min 185 --mu-max 215
         --peak-threshold 0.01
         --integral-threshold "$INTEGRAL"
         --min-height "$MIN_HEIGHT"
         --centroid-halfwidth "$HALFWIDTH"
         --pairwise-bins 240
         --output-dir "$outdir"
         --device "$DEVICE"
         --dataset-name "HL-LHC PU200 held-out ($tag)")

    printf '%q ' "${cmd[@]}" > "$outdir/INVOCATION.txt"
    printf '\n# git %s\n' "$(git rev-parse HEAD)" >> "$outdir/INVOCATION.txt"

    echo "[$(date +%H:%M:%S)] START $tag -> $outdir"
    if "${cmd[@]}" > "$outdir/eval.log" 2>&1; then
        echo "[$(date +%H:%M:%S)] DONE  $tag"
        grep -E "Eff=|sigma=" "$outdir/eval.log" | tail -2
    else
        echo "[$(date +%H:%M:%S)] FAIL  $tag -- see $outdir/eval.log"
        tail -20 "$outdir/eval.log"
    fi
done

echo
echo "=== summary ==="
grep -H "Eff=" "$OUT"/*/eval.log
