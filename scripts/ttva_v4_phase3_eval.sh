#!/bin/bash
# TTVA v4 phase 3: evaluate the trained associators against AMVF on the
# held-out r16443 mu-window slice (1920 events, 213,738 truth PVs with
# nTrk >= 2 - the same slice as the PV-Finder v6 held-out evaluation).
#
# Every model is scored through the identical classify_assignments core that
# produced the AMVF baseline in chain_<tag>/chain_info.json, so the rows are
# directly comparable.
#
# Usage (inside tmux):
#   bash scripts/ttva_v4_phase3_eval.sh <label>=<ckpt.pyt> [<label>=<ckpt.pyt> ...]
# e.g.
#   bash scripts/ttva_v4_phase3_eval.sh \
#     v4aug=model_weights/ttva_gnn_hllhc_v4_aug/ttva_gat_alleta_k20_v4_aug180k_epoch_312.pyt \
#     v4noaug=model_weights/ttva_gnn_hllhc_v4_noaug/ttva_gat_alleta_k20_v4_noaug180k_epoch_312.pyt \
#     v3zeroshot=model_weights/ttva_gnn_hllhc_v3/ttva_gat_pu200_k20_v3_aug180k_epoch_156.pyt
set -eu
cd /data/home/matmauro/codice/PV-Finder
source venv/bin/activate

DEVICE=${DEVICE:-3}
RUN4=data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026
ROOT_TEST=$RUN4/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root
GRAPHS=data/run4_all_etas/ttva_graphs
CHAIN=$GRAPHS/pu200_chain_v6_k20_test.pt
TRUTH=$GRAPHS/pu200_truth_k20_test_r16443.pt
OUT=${OUT:-outputs/08_05_2026_output/gnn_ttva_v4}
IDX=$OUT/chain_test/entry_indices.npy

[ $# -ge 1 ] || { echo "need at least one <label>=<ckpt>"; exit 2; }
for f in "$CHAIN" "$TRUTH" "$IDX"; do
    [ -f "$f" ] || { echo "missing input: $f (run phase 1 first)"; exit 2; }
done

for spec in "$@"; do
    ckpt=${spec#*=}
    [ -f "$ckpt" ] || { echo "missing checkpoint: $ckpt"; exit 2; }
done

# Stages run across ALL models before moving to the next stage, deliberately.
# The whole matrix takes a couple of hours on a loaded box, so if it has to be
# cut short the headline comparison should already be on disk. Stage 1 is the
# number the campaign exists to produce; stage 2 is a ceiling and stage 3 a
# secondary metric, both of which are still useful with stage 1 missing but not
# the other way round.

# --- Stage 1: chain threshold scan (the headline) -------------------------
# Vertex categories + track-level metrics, under both the all-peaks and
# drop-empty vertex conventions.
SCAN_ARGS=()
for spec in "$@"; do
    label=${spec%%=*}; ckpt=${spec#*=}
    echo "=== [1/3 chain scan] $label : $ckpt ==="
    python -u -m gnn.evaluation.chain_scan \
        -r "$CHAIN" -w "$ckpt" -d "$DEVICE" -o "$OUT/chain_scan_$label/"
    SCAN_ARGS+=(--scan "$label=$OUT/chain_scan_$label/chain_scan.json")
done

# --- Stage 2: ceiling, the same associator on TRUTH vertices, same events --
for spec in "$@"; do
    label=${spec%%=*}; ckpt=${spec#*=}
    echo "=== [2/3 truth ceiling] $label ==="
    python -u -m gnn.evaluation.evaluate_ttva_graphs \
        -r "$TRUTH" -w "$ckpt" -e MaxScore -t 0.5 -d "$DEVICE" \
        -o "$OUT/truth_eval_${label}_t050/"
    python -u -m gnn.evaluation.evaluate_ttva_graphs \
        -r "$TRUTH" -w "$ckpt" -e MaxScore -t 0.95 -d "$DEVICE" \
        -o "$OUT/truth_eval_${label}_t095/"
done

# --- Stage 3: hard-scatter identification, GNN vs AMVF on the same events --
for spec in "$@"; do
    label=${spec%%=*}; ckpt=${spec#*=}
    echo "=== [3/3 HS-ID] $label ==="
    python -u -m gnn.evaluation.hs_id_pu200 \
        --graphs "$CHAIN" --root "$ROOT_TEST" --entry-indices "$IDX" \
        -w "$ckpt" -d "$DEVICE" -o "$OUT/hs_id_$label/"
done

# --- Stage 4: publication plots from the measured JSONs (no re-evaluation) --
# SKIP_PLOTS=1 when evaluating arms one at a time as they finish training; the
# plots are meaningful only once every arm's scan exists, so draw them in a
# final pass with all --scan arguments rather than overwriting per arm.
if [ "${SKIP_PLOTS:-0}" != "1" ]; then
    python -u -m gnn.diagnostics.plot_ttva_pu200_pub \
        --gap "$OUT/ttva_gap_v6_test/gap_decomposition.json" \
        "${SCAN_ARGS[@]}" \
        --desc 'HL-LHC $t\bar{t}$, $\langle\mu\rangle=200$, ITk, $|\eta|<4$' \
        -o "$OUT/publication/"
else
    echo "SKIP_PLOTS=1: run stage 4 separately once all arms are scanned"
fi

echo "=== PHASE 3 DONE -> $OUT ==="
