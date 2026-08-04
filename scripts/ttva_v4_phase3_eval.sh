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

DEVICE=${DEVICE:-2}
RUN4=data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026
ROOT_TEST=$RUN4/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root
GRAPHS=data/run4_all_etas/ttva_graphs
CHAIN=$GRAPHS/pu200_chain_v6_k20_test.pt
TRUTH=$GRAPHS/pu200_truth_k20_test_r16443.pt
OUT=outputs/08_04_2026_output/gnn_ttva_v4
IDX=$OUT/chain_test/entry_indices.npy

[ $# -ge 1 ] || { echo "need at least one <label>=<ckpt>"; exit 2; }
for f in "$CHAIN" "$TRUTH" "$IDX"; do
    [ -f "$f" ] || { echo "missing input: $f (run phase 1 first)"; exit 2; }
done

SCAN_ARGS=()
for spec in "$@"; do
    label=${spec%%=*}; ckpt=${spec#*=}
    [ -f "$ckpt" ] || { echo "missing checkpoint: $ckpt"; exit 2; }
    echo "=== $label : $ckpt ==="

    # 1. Chain threshold scan (vertex categories + track-level metrics),
    #    under both the all-peaks and drop-empty vertex conventions.
    python -u -m gnn.evaluation.chain_scan \
        -r "$CHAIN" -w "$ckpt" -d "$DEVICE" -o "$OUT/chain_scan_$label/"

    # 2. Ceiling: the same associator on TRUTH vertices, same events.
    python -u -m gnn.evaluation.evaluate_ttva_graphs \
        -r "$TRUTH" -w "$ckpt" -e MaxScore -t 0.5 -d "$DEVICE" \
        -o "$OUT/truth_eval_${label}_t050/"
    python -u -m gnn.evaluation.evaluate_ttva_graphs \
        -r "$TRUTH" -w "$ckpt" -e MaxScore -t 0.95 -d "$DEVICE" \
        -o "$OUT/truth_eval_${label}_t095/"

    # 3. Hard-scatter identification, GNN vs AMVF on the same events.
    python -u -m gnn.evaluation.hs_id_pu200 \
        --graphs "$CHAIN" --root "$ROOT_TEST" --entry-indices "$IDX" \
        -w "$ckpt" -d "$DEVICE" -o "$OUT/hs_id_$label/"

    SCAN_ARGS+=(--scan "$label=$OUT/chain_scan_$label/chain_scan.json")
done

# 4. Publication plots from the measured JSONs (no re-evaluation).
python -u -m gnn.diagnostics.plot_ttva_pu200_pub \
    --gap "$OUT/ttva_gap_v6_test/gap_decomposition.json" \
    "${SCAN_ARGS[@]}" \
    --desc 'HL-LHC $t\bar{t}$, $\langle\mu\rangle=200$, ITk, $|\eta|<4$' \
    -o "$OUT/publication/"

echo "=== PHASE 3 DONE -> $OUT ==="
