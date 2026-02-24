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
python -m pv_finder.evaluation.evaluate_gnn_ttva \
    -r /path/to/reco_graphs.pt \
    -f /path/to/recoTracks_inctype.h5 \
    -i /path/to/test_indices.npy \
    -w /path/to/model_weights.pyt \
    -e MaxScore \
    -t 0.5 \
    -d 0
```

Results are saved as `.npy` files with per-event classification counts and per-PV info.

## Code

`src/pv_finder/evaluation/evaluate_gnn_ttva.py`
