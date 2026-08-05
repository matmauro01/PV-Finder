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
- `config_gnn_ttva_hllhc.yml` — HL-LHC PU200 |η|<2.5 (30k kNN k=20 graphs
  built by `gnn.data.root_to_graphs`, `hllhc` resolution preset).
- `config_gnn_ttva_hllhc_v4_aug.yml` / `_v4_noaug.yml` — HL-LHC PU200
  **extended |η|** on `data/run4_all_etas`, **`hllhc_alleta` preset**.

> **Resolution preset — read before building any graphs.** `hllhc` is
> superseded and is correct *only* for the old |η|<2.5 production. Anything on
> `data/run4_all_etas` must use **`hllhc_alleta`**: at fixed truth nTracks the
> AMVF-truth residual is 13–26% wider at extended |η|, because nTracks now
> counts forward tracks with σ(z0) up to 2.8 mm. The preset sets PV-node
> heights and edge significances — features the GAT trains on — so the wrong
> preset does not crash, it silently trains on mis-scaled inputs.
> `--resolution-preset` is now a required argument (there is no
> `DEFAULT_RESOLUTION_PRESET` any more). Note that
> `scripts/build_v3_shards.sh` hard-codes `hllhc` and is correct only for the
> v3 |η|<2.5 shards; `scripts/build_ttva_v4_shards.sh` uses `hllhc_alleta`.

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

**Completed run v3 (2026-07-14, `ttva_gat_pu200_k20_v3_aug180k`):** 162
epochs in 3.2 h on one A100 (~70 s/shard-epoch incl. load), final
train/val 0.2115/0.2120 (val on *augmented* graphs — not comparable to
v1/v2), perfectly smooth. **Best = epoch 156, the new production
checkpoint**: truth graphs 0.823 clean/truth @ t=0.5 / 0.9155 @ t=0.95;
full chain **0.716 @ 0.05% fakes (t=0.98)** vs v1 0.647 / AMVF 0.573.
Improved every metric incl. HS-ID (98.1% > AMVF). The transfer gap is
effectively closed (96% of the oracle bound on peaks).

## v4: extended-|η| retrain on the PV-Finder v6 chain (2026-08-05)

v4 moves the associator onto `data/run4_all_etas` (the July-2026 extended-|η|
re-production) and onto PV-Finder v6, and runs augmentation as a **controlled
A/B** rather than assuming the v3 finding carries over. The v6 chain has a much
higher junk-peak rate than the v4b chain the v3 augmentation was tuned on.

| | arm A | arm B |
|---|---|---|
| config | `config_gnn_ttva_hllhc_v4_aug.yml` | `config_gnn_ttva_hllhc_v4_noaug.yml` |
| shards | `v4_shards_augmented` | `v4_shards_pristine` |
| augmentation | chain-like, p=0.7 | none (control) |
| GPU | 2 | 3 |

Everything else is identical: same events, same shard boundaries, same per-shard
seeds (100+i), same architecture, same schedule, and **both validate on the same
pristine 5k val shard** so the two learning curves share one yardstick. Arm A
therefore validates slightly out of its training distribution by design — the
chain evaluation, not this loss, decides between the arms.

Differences from the v3 recipe that matter:

- **`hllhc_alleta` resolution preset, not `hllhc`.** See the box above.
- **18 shards × 10k, not 9 × 20k.** Extended-|η| events carry ~1366 tracks
  against ~927, so a 20k shard no longer fits the 32 GB per-process limit.
- **Augmentation quantiles measured on r16638 and reported on r16443**, two
  disjoint held-out files. v3 measured and reported on the same events.
- **Augmentation measured at centroid half-width 3**, matching the v6 operating
  point, so arm A's peak statistics match the deployment condition exactly.

### Epoch budget: 108, not 324

The design target was 324 epochs (18 dataset passes, matching v3's
graph-presentation count). The campaign had a hard 4–5 h training budget and
sneezy was at load ≈296, where an epoch costs 85–165 s rather than the ~28 s it
costs on a quiet box. **108 is the largest multiple of 18 whose cosine anneal
completes inside that window.**

Two constraints worth remembering for any future resize:

1. **The epoch count must be a multiple of the shard count.** `ShardCyclingLoader`
   advances one shard per epoch round-robin, so any other count shows some
   shards one more time than others. In an A/B whose whole point is that the
   arms differ only in augmentation, that is an uncontrolled variable.
2. **Shorten the schedule; do not truncate a long one.** With cosine annealing
   to 1e-5, an early-stopped checkpoint has not annealed and does not represent
   the recipe. `T_max` must equal the epoch budget.

`save_frequency: 1` (every epoch), so a wall-clock guard always has a loadable
state_dict. `scripts/ttva_v4_deadline_guard.sh` enforces the budget with
SIGTERM — never SIGKILL, which on this kernel creates unkillable core-spinners.

**108 epochs is 6 dataset passes against v3's 18.** Read the two arms as a
like-for-like A/B against each other, not against v3's absolute numbers.

### Verifying a shard build before training on it

Both builds were checked before use (a 3-hour run on a silently broken shard set
is the expensive failure mode here). Sampling 300 graphs/shard:

| | pristine | augmented |
|---|---|---|
| tracks/event | 1366.4 | 1366.4 (same events) |
| PV nodes/event | 138.9 | 132.3 |
| track→pv edges | 27,327 | 27,327 (= knn 20 × tracks) |
| positive-edge fraction | 0.0490 | 0.0455 |
| PV heights identically 0 | 0.0000 | 0.0000 |

The PV-node drop is the augmentation working, and it is quantitatively right:
of ~138.9 truth vertices, ~111.3 have nTrk≥2 and are droppable at the measured
~22% miss rate, giving ~114.4 kept, plus junk at 0.1397/kept ≈ 16.0, i.e. ~130.4
per augmented event. With augmentation applied to 70% of events that predicts
132.4 overall against the 132.3 measured. The lower positive-edge fraction is
the same effect from the label side: junk nodes contribute only negative edges.

**Always confirm PV heights are not identically zero.** The double-`Z_MIN` bug
(fixed 2026-07-14) left them at exactly 0 in every pre-fix truth graph, and
legacy files deliberately retain that for reproduction, so a stale file mixed
into a new campaign trains a dead feature.

## mu60 v2: fixed heights + augmentation (2026-07-14)

`config_gnn_ttva_mu60_v2.yml`, run `ttva_gat_mu60_v2_aug`: augmented
48,450-event rebuild in the exact legacy ordering (test excluded by
construction; params from `gnn.diagnostics.mu60_aug_params`), 2 train
shards + 12,750-graph val, 120 cosine epochs in 3.3 h (~97 s/epoch),
final val 0.1057, smooth. **Best = epoch 115, the mu60 production
checkpoint**: chain 0.7743 @ 0.02% fakes (t=0.99), truth graphs 0.857
(t=0.5) / 0.892 (t=0.98), track F1 0.874, HS-ID 98.98%. See the
evaluation doc for the full ladder.

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
