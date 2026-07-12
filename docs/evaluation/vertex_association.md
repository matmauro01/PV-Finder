# Evaluation — Vertex Association

Evaluation of the GNN TTVA model on reconstructed primary vertices.

## Classification

Each reconstructed PV is classified as one of:

| Category | Definition |
|----------|-----------|
| **Clean** | Dominant truth PV contributes >= 70% of matched tracks |
| **Merged** | Dominant truth PV contributes < 70% (tracks from multiple truth PVs) |
| **Split** | Another reco PV already claimed the same truth PV with higher sum(pT²) |
| **Fake** | No matched tracks have truth associations |

## Evaluation Methods

- **MaxScore**: For each track, keep only the top-1 highest-scoring PV edge (if above threshold). Each track assigned to at most one PV.
- **Threshold**: Accept all edges with score above threshold. A track can be associated to multiple PVs.

## Usage

```bash
python -m gnn.evaluation.evaluate_ttva \
    -r /path/to/reco_graphs.pt \
    -f /share/lazy/qibinlei/recoTracks_incamvfassoc.h5 \
    -i configs/qibin_test_main_indices_v2.p \
    -w model_weights/gnn_ttva_epoch100.pyt \
    -e MaxScore \
    -t 0.5 \
    -d 0
```

Results are saved as `.npy` files with per-event classification counts and per-PV info.

## Baseline (Nov 2025, PVF e400 + GNN e100, MaxScore t=0.5)

**Reproduced bit-exactly on 2026-07-12** with the restored `gnn` pipeline:
per-event rows identical to the saved baseline for all 2,550 events
(`outputs/07_12_2026_ttva_reproduction/`, using the Nov 2025 pre-built
inference graphs).

From `~/codice/atlas_pvfinder/tracks_to_vertex/total_results_MaxScore.p`
(2,550 test events, `qibin_test_main_indices_v2.p`):

| Metric | Value |
|--------|-------|
| Reco PVs | 66,472 (truth 72,189 → 92.1% recovery) |
| Clean | 76.04% |
| Merged | 16.49% |
| Split | 7.05% |
| Fake | 0.42% |

PVF peaks were found with `threshold=1e-2`, `integral_threshold=0.2`,
`min_width=3` (see legacy `tracks_to_vertex/pvfinder_output_to_graph.py`).
PVF model: `e2e_mlpHist50_e2e400_1latent_mse_phase2_epoch_400.pyt`
(in `~/codice/atlas_pvfinder/model_weights/e2e_attempt4_mlp_hist/`).

## Code

`src/gnn/evaluation/evaluate_ttva.py` — restored from commit `51523df`
(deleted by mistake in the March 2026 eval cleanup `0da13fd`).
