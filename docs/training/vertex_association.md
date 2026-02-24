# Training — Vertex Association

Training procedure for the GNN TTVA model.

## Data Preparation

1. Build training graphs from MC HDF5 file:
```bash
python -m pv_finder.data.graph_construction \
    -f /path/to/recoTracks_inctype.h5 \
    -i /path/to/indices.npy \
    -o /path/to/ttva_graphs.pt
```

This produces a list of `HeteroData` objects saved with `torch.save`.

## Training

```bash
python -m pv_finder.training.train_gnn_ttva \
    -c configs/vertex_association/config_gnn_ttva.yml
```

**Loss:** `BCEWithLogitsLoss` with dynamic `pos_weight` computed per batch as `num_negative_edges / num_positive_edges`. This handles severe class imbalance (most track-PV pairs are not associated).

**Optimizer:** Adam (lr=0.001, betas=(0.9, 0.999)).

**Split:** Configurable via YAML (default 70/15/15 train/val/test).

**Checkpoints:** Saved every N epochs (configurable). Both `.pyt` (state_dict only) and `.pth` (full state with optimizer) are saved.

**Tracking:** MLflow, URI at `PV-Finder/mlruns`.

## Config

See `configs/vertex_association/config_gnn_ttva.yml` for all parameters.

## Code

- Training loop: `src/pv_finder/training/training_gnn.py`
- Training script: `src/pv_finder/training/train_gnn_ttva.py`
