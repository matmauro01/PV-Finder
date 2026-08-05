#!/bin/bash
# TTVA v4 phase 1: build the PV-Finder v6 chain graphs and measure the
# chain-like augmentation parameters.
#
# Two DISJOINT held-out slices, both flat-mu files that PV-Finder v6 never
# trained on, both restricted to mu in [185, 215]:
#
#   r16638 -> augmentation-measurement set. The v3 campaign measured its
#             augmentation quantiles on the same events it later reported on;
#             using a separate file removes that self-reference.
#   r16443 -> the evaluation set. Its 1920 events / 213,738 truth PVs are
#             exactly the slice behind outputs/08_04_2026_output/eval_v6_heldout/r16443.
#
# Defaults are the FINAL v6 operating point, the one behind
# scripts/eval_v6_operating_point.sh and outputs/08_04_2026_output/eval_v6_heldout:
#
#   peak_threshold 0.01   integral_threshold 0.30   min_width 3
#   min_height 0.03       centroid_halfwidth 3
#
# centroid_halfwidth is the one to watch. The peak_finding library default is
# still 0 (historical full-region weighted mean) so that v3-era TTVA
# checkpoints keep reproducing, so every call site here passes it explicitly.
# It sets the PV-node z, which feeds every edge feature, so a mismatch between
# the graphs a checkpoint trains on and the graphs it is evaluated on degrades
# the result quietly and looks like a modelling problem.
#
# Usage (inside tmux):
#   DEVICE=3 bash scripts/ttva_v4_phase1_chain.sh
set -eu
cd /data/home/matmauro/codice/PV-Finder
source venv/bin/activate

PEAK_THRESHOLD=${PEAK_THRESHOLD:-0.01}
INTEGRAL_THRESHOLD=${INTEGRAL_THRESHOLD:-0.30}
MIN_WIDTH=${MIN_WIDTH:-3}
MIN_HEIGHT=${MIN_HEIGHT:-0.03}
CENTROID_HALFWIDTH=${CENTROID_HALFWIDTH:-3}
DEVICE=${DEVICE:-3}

RUN4=data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026
PVF=model_weights/hllhc_alleta_v6_mse_2ep_phase2_epoch_2_fullstate.pth
GRAPHS=data/run4_all_etas/ttva_graphs
OUT=${OUT:-outputs/08_05_2026_output/gnn_ttva_v4}
mkdir -p "$GRAPHS" "$OUT"

echo "=== peak operating point: thr=$PEAK_THRESHOLD integral=$INTEGRAL_THRESHOLD"\
     "width=$MIN_WIDTH floor=$MIN_HEIGHT halfwidth=$CENTROID_HALFWIDTH ==="

chain() {  # chain <tag> <root> <n_events_cap>
    local tag=$1 root=$2 cap=$3
    python -u -m gnn.data.pu200_chain_graphs \
        --root "$root" --pvf-weights "$PVF" \
        --entry-start 0 --entry-stop 25000 --mu-min 185 --mu-max 215 \
        --max-events "$cap" --knn 20 -d "$DEVICE" \
        --peak-threshold "$PEAK_THRESHOLD" \
        --integral-threshold "$INTEGRAL_THRESHOLD" \
        --min-width "$MIN_WIDTH" --min-height "$MIN_HEIGHT" \
        --centroid-halfwidth "$CENTROID_HALFWIDTH" \
        --unet-channels 280 --latent-channels 4 --hidden-nodes 128 \
        --output "$GRAPHS/pu200_chain_v6_k20_$tag.pt" \
        --output-dir "$OUT/chain_$tag/"
}

# 1. Augmentation-measurement chain (r16638), 1500 events is plenty for
#    quantile estimation and keeps the ROOT pass short.
chain augmeas "$RUN4/ATLAS_PVFinderData_601229_e8481_s4494_r16638_PU200.root" 1500

# 2. Evaluation chain (r16443), all 1920 mu-window events of the v6 slice.
chain test "$RUN4/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root" 0

# 3. Finder/associator gap decomposition + augmentation quantiles, measured
#    on the r16638 chain ONLY.
python -u -m gnn.diagnostics.chain_gap_decomposition \
    --graphs "$GRAPHS/pu200_chain_v6_k20_augmeas.pt" \
    --root "$RUN4/ATLAS_PVFinderData_601229_e8481_s4494_r16638_PU200.root" \
    --entry-indices "$OUT/chain_augmeas/entry_indices.npy" \
    --resolution-preset hllhc_alleta \
    --sigma-events 300 --pvf-weights "$PVF" -d "$DEVICE" \
    --peak-threshold "$PEAK_THRESHOLD" \
    --integral-threshold "$INTEGRAL_THRESHOLD" \
    --min-width "$MIN_WIDTH" --min-height "$MIN_HEIGHT" \
    --centroid-halfwidth "$CENTROID_HALFWIDTH" \
    -o "$OUT/ttva_gap_v6/"

# 4. The same decomposition on the EVALUATION chain, for its oracle and
#    finder-cap bounds (its augmentation_params.json is not used for training).
python -u -m gnn.diagnostics.chain_gap_decomposition \
    --graphs "$GRAPHS/pu200_chain_v6_k20_test.pt" \
    --root "$RUN4/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root" \
    --entry-indices "$OUT/chain_test/entry_indices.npy" \
    --resolution-preset hllhc_alleta \
    -o "$OUT/ttva_gap_v6_test/"

# 5. Re-verify the numba fast path AT THIS operating point. peak_finding.py and
#    peak_finding_fast.py are independent transcriptions, so a change to one
#    and not the other shows up here and nowhere else.
python -u -m gnn.evaluation.verify_fast_paths \
    --hists "$OUT/ttva_gap_v6/histograms_300ev.npz" \
    --peak-threshold "$PEAK_THRESHOLD" \
    --integral-threshold "$INTEGRAL_THRESHOLD" \
    --min-width "$MIN_WIDTH" --min-height "$MIN_HEIGHT" \
    --centroid-halfwidth "$CENTROID_HALFWIDTH" \
    -d "$DEVICE" -o "$OUT/fastpaths_post_peakchange/"

echo "=== PHASE 1 DONE ==="
