# Vertex Association (TTVA) — Overview and Model

**Start here** for the GNN track-to-vertex association system. Companion pages:
[training](../training/vertex_association.md) ·
[evaluation](../evaluation/vertex_association.md) ·
[diagnostics](../diagnostics/vertex_association.md).

## What TTVA is and where it sits

PV-Finder's vertex-finding stage turns tracks into a 12,000-bin z-histogram and
peak-finds it — it answers "*where* are the primary vertices?" but not "*which
tracks belong to which vertex?*". Track-to-Vertex Association (TTVA) answers
the second question. It matters because physics analyses need per-vertex track
lists (hard-scatter identification via max sum-pT², pileup jet rejection), and
because assigning tracks lets us classify reconstructed vertices the same way
AMVF results are classified (Clean/Merged/Split/Fake) for apples-to-apples
comparison.

```
tracks ──► PVF (MLP+UNet) ──► 12000-bin histogram ──► peak finding ──► candidate PVs
   │                                                                       │
   └────────────────────────► TTVA GNN ◄──────────────────────────────────┘
                                  │
                     per-(track, PV) association score
```

**Key design decision** (colleague guidance, matches Nov 2025 baseline):
**train on MC truth vertices** (labels exist), **evaluate on PV-Finder
reconstructed vertices** (the deployment condition).

## Problem formulation

Each event is a **heterogeneous bipartite graph**:

- **Track nodes** — 8 features: `[d0, z0, d0_err, z0_err, cov_d0z0, theta, phi, pT/50000]`
- **PV nodes** — 2 features: `[z_position, peak_height]`
  - truth graphs: true z; height synthesized from the same Gaussian-CDF recipe
    that builds PVF target histograms, with σ(N) = A·N^(−B) + C
  - inference graphs: peak-finder output (z, height) from the PVF histogram
- **Edges** — fully connected track↔PV (every track paired with every PV),
  made undirected via `T.ToUndirected()` (adds `(pv, rev_to, track)` copies)
- **Edge attributes** — 3 physics quantities:
  1. longitudinal significance `(z0 − z_PV) / sqrt(σ_z0² + σ_PV²)`
  2. horizontal significance `d0 / σ_d0`
  3. absolute distance `|z0 − z_PV|`
- **Labels** (training only) — edge y=1 iff MC says the track came from that PV

Task: **binary edge classification**. The GNN outputs one logit per edge;
sigmoid gives an association score in [0,1].

## Architecture: TTVAGATModel

**Code:** `src/gnn/models/ttva_gat.py` (~26k parameters)

```
track.x (n_t, 8) ──► Linear(8,32)+LeakyReLU ──┐
                                               ├─► 2 × HeteroConv{GATConv(4 heads,
pv.x    (n_p, 2) ──► Linear(2,32)+LeakyReLU ──┘      edge_dim=3, concat=False)}
                                               │     + residual + ReLU
                                               ▼
                          edge (t→p): cat(track_emb[t], pv_emb[p])  (64)
                          Linear(64,32)+LeakyReLU → Linear(32,32)+LeakyReLU
                          → Linear(32,1) → edge logit
```

- GATConv attention uses the 3 edge attributes (edge_dim=3), so the network
  sees the significances both as message-passing biases and implicitly via
  node geometry.
- HeteroConv aggregation is `max`; both directions (`track→pv`, `pv→rev_to→track`)
  have their own GATConv per layer.
- State-dict keys are backward-compatible with the original
  `TTVA_GATGraphConv_Model` (verified 38/38 on the baseline checkpoint).

## Graph construction

**Code:** `src/gnn/data/graph_construction.py`

- `create_training_graph(event_data)` — truth vertices + truth edge labels.
  CLI: builds a full graph dataset from an event-keyed HDF5.
- `create_inference_graph(...)` — PV nodes from pre-computed PVF peaks
  (caller supplies `pred_z/pred_heights/pred_sigmas`); no labels.
- `load_event_indices(path)` — .npy or pickled index lists.

**Code:** `src/gnn/data/pvf_to_graphs.py` — driver from PVF **histograms** to
inference graphs: peak-finds each histogram with
`pv_finder.utils.peak_finding.pv_locations_updated_res` (defaults
`threshold=1e-2`, `integral_threshold=0.2`, `min_width=3` = Nov 2025 baseline)
and calls `create_inference_graph`. Accepts .npy / pickle / HDF5 histogram
inputs.

PV resolution model used for truth-graph heights and edge significances:
`σ_PV(N) = 0.238·N^(−0.495) − 0.0008` (Run-3 fit; see
`pv_finder/utils/constants.py`).

## Data

| Dataset | Path | Truth assoc? | Role |
|---|---|---|---|
| MC ttbar μ≈60 (event-keyed) | `/share/lazy/qibinlei/recoTracks_incamvfassoc.h5` (51k events) | ✅ `pv_assoc_tracks` (+ AMVF `reco_pv_assoc_tracks`) | train + eval |
| Test indices | `configs/qibin_test_main_indices_v2.p` (2,550 events, shuffled; last 5% of pubnote ordering) | — | baseline eval split |
| Run 2 / Run 3 real data | ROOT files (see data docs) | ❌ (AMVF `RecoVertex_assocTracks` as reference only) | future eval |
| Run 4 PU200 | ROOT has `TruthVertex_assocTracks` | ✅ | future; fully-connected graphs scale as tracks×PVs ≈ 117k edges/event at μ=200 → needs kNN construction |

## Trained weights

- `model_weights/gnn_ttva_epoch100.pyt` — **the baseline checkpoint**,
  byte-identical (md5) to `test_GATConv_edgeattr_BCE_100.pyt` (Nov 2025).
  Loads into `TTVAGATModel()` defaults with `strict=True`.
- Full series epochs 0–200 (every 25):
  `~/codice/atlas_pvfinder/tracks_to_vertex/model_weights/`.
- MLflow: experiments "ATLAS 2025 GNN TTVA" (training) and "ATLAS 2025 Reco
  Vtx GNN TTVA" at `/share/lazy/qibinlei/trackstoHists` (training losses only,
  no eval rates).

## Status (2026-07-12)

- Nov 2025 baseline **reproduced bit-exactly** end-to-end with this package:
  graphs regenerated from saved PVF histograms are tensor-identical, and the
  2,550-event Clean/Merged/Split/Fake rows match the saved baseline exactly.
  See [evaluation](../evaluation/vertex_association.md) for numbers and the
  Qi Bin UNet-85 comparison.
- Not yet done: retraining in this repo, ground-truth-vertex eval refresh,
  Run 2/3 agreement-with-AMVF eval, Run 4 PU200.

## Known limitations / next steps (from internal note §7)

1. Fully-connected graphs scale poorly to PU200 → kNN in z or (z, d0).
2. Hard-scatter identification (max sum-pT²) not yet addressed.
3. pT underused — weighting/attention on high-pT tracks expected to help.
4. TTVA is the natural home for HGTD timing features (z ambiguity breaking).
