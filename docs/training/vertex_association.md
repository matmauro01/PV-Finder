# Training — Vertex Association

Training procedure for the GNN TTVA model.

## Data Preparation

1. Build training graphs from MC HDF5 file:
```bash
python -m gnn.data.graph_construction \
    -f /share/lazy/qibinlei/recoTracks_incamvfassoc.h5 \
    -i /path/to/indices.npy \
    -o /path/to/ttva_graphs.pt
```

Indices can be a `.npy` array or a pickled list (e.g. the legacy
`configs/qibin_test_main_indices_v2.p`).

This produces a list of `HeteroData` objects saved with `torch.save`.

## Training

```bash
python -m gnn.training.train_ttva \
    -c configs/gnn/config_gnn_ttva.yml
```

**Loss:** `BCEWithLogitsLoss` with dynamic `pos_weight` computed per batch as `num_negative_edges / num_positive_edges`. This handles severe class imbalance (most track-PV pairs are not associated).

**Optimizer:** Adam (lr=0.001, betas=(0.9, 0.999)).

**Split:** Configurable via YAML (default 70/15/15 train/val/test).

**Checkpoints:** Saved every N epochs (configurable). Both `.pyt` (state_dict only) and `.pth` (full state with optimizer) are saved.

**Tracking:** MLflow, URI at `PV-Finder/mlruns`.

## Config

See `configs/gnn/config_gnn_ttva.yml` for all parameters.

## Reference training run (Nov 2025 baseline)

The checkpoint behind the ACAT/internal-note numbers was trained with
`atlas_pvfinder/tracks_to_vertex/configuration_GATConv_edgeattr_TTVA.yml`:
51,000-event graph dataset from `recoTracks_incamvfassoc.h5`, sequential split
[0.7, 0.25, 0.05], batch size 32, lr 0.001, 201 epochs, BCE with dynamic
pos_weight. Run name `test_GATConv_edgeattr_BCE` in MLflow experiment
"ATLAS 2025 GNN TTVA".

## Code

- Training loop: `src/gnn/training/training_loop.py`
- Training script: `src/gnn/training/train_ttva.py`
