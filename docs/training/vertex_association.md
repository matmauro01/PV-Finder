# Training — Vertex Association

Training procedure for the GNN TTVA model.

## Data Preparation

1. Build training graphs from MC HDF5 file:
```bash
python -m gnn.data.h5_to_graphs \
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

See `configs/gnn/config_gnn_ttva.yml` for all parameters. Variants:
- `config_gnn_ttva_repro.yml` — μ≈60 end-to-end reproduction (existing 51k
  fully-connected graph set from the Nov 2025 workspace).
- `config_gnn_ttva_hllhc.yml` — HL-LHC PU200 (30k kNN k=20 graphs built by
  `gnn.data.root_to_graphs`, hllhc resolution preset).

## HL-LHC PU200 training

```bash
# 1. Build truth graphs from ROOT (~15 min for 30k events)
python -u -m gnn.data.root_to_graphs \
    --input data/run4/Run4_MC21_ITk/ATLAS_PVFinderData_HLLHC_mc21_14TeV_ttbar_SingleLep_PU200.root \
    --output data/run4/ttva_graphs/pu200_truth_k20_30k.pt \
    --max-events 30000

# 2. Train (tmux; ~4-5 min/epoch on an A100, 201 epochs ≈ 15 h)
python -u -m gnn.training.train_ttva -c configs/gnn/config_gnn_ttva_hllhc.yml
```

Note: `train_ttva` materializes the lazy GATConv layers with a dummy forward
before creating the optimizer — do not reorder that block; fresh training
breaks without it.

**Completed run v1 (2026-07-12/13, `ttva_gat_pu200_k20`):** 201 epochs in
~3.5 h on one A100 (21,000 train / 7,500 val graphs), final train loss
0.2551 / val 0.2265. Checkpoints every 25 epochs in
`model_weights/ttva_gnn_hllhc/`. Training was unstable around epochs
50–75 (clean rate dipped to ~34%, lr=1e-3 too aggressive) before
recovering; best checkpoint epoch 175.

**Completed run v2 (2026-07-13, `ttva_gat_pu200_k20_v2_cosine`):** same
architecture + cosine LR (1e-3→1e-5) + grad clip 1.0
(`config_gnn_ttva_hllhc_v2.yml`; both knobs are opt-in config keys).
Final val **0.2050** (−9.5% vs v1), fully smooth learning curve, best
checkpoint **epoch 175** in `model_weights/ttva_gnn_hllhc_v2/`:
clean/truth 0.816 (t=0.5) / 0.9175 (t=0.95) on truth graphs — the new
best associator. NOTE: on full-chain (peak-node) graphs v1-e175 still
gives the better low-fake operating points — see the evaluation doc's
transfer-gap note before swapping checkpoints in the chain.

## v3: big data + chain-like augmentation (2026-07-14)

v3 attacks the truth→peak transfer gap and the data-volume ceiling at
once (`config_gnn_ttva_hllhc_v3.yml`, run `ttva_gat_pu200_k20_v3_aug180k`):

- **180k training graphs** from the all-hadronic 601237 r16633 PU200
  sample (file 1, entries 0–180k), built by `scripts/build_v3_shards.sh`
  as 9 shards × 20k with `root_to_graphs --augment-params`. Val = 5k
  augmented graphs from file 2. Test stays the SAME SingleLep slice as
  v1/v2 (entries 28500+), rebuilt with fixed heights as
  `pu200_truth_k20_test_fixedheights.pt` (+ the chain graphs, unchanged).
- **Chain-like augmentation** (p=0.7/event, `gnn.data.graph_augmentation`):
  finder-miss vertex dropping, peak-residual z jitter, measured peak
  sigmas/heights, junk-PV injection — all empirical quantiles measured on
  the v4b chain by `gnn.diagnostics.chain_gap_decomposition`.
- **PV heights fixed**: earlier trainings (μ60 baseline, v1, v2) had PV
  heights ≡ 0 from the double-Z_MIN bug; v3 is the first training with a
  live height feature.
- **Shard cycling** (`gnn.training.shard_loader.ShardCyclingLoader`): one
  20k shard resident at a time, advanced per epoch round-robin
  (`data_files` + `val_file` config keys). 162 epochs = 18 dataset
  passes, cosine 1e-3→1e-5 (T_max = 162), grad clip 1.0, checkpoints
  every 6 epochs.

Build gotcha: `torch.save` of a 20k-graph shard takes minutes while
worker thread pools spin at high CPU — the process is NOT hung; wait for
the "Saved N graphs" line (a first-attempt kill at that stage truncated
all 8 shards, which had to be rebuilt).

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
