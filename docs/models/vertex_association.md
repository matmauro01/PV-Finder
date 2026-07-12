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

**Code:** `src/gnn/models/ttva_gat.py` (39,521 parameters once the lazy
GATConv layers are materialized)

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

**Code:** `src/gnn/data/root_to_graphs.py` — driver from a **PVFinderData
ROOT ntuple** to truth-training graphs (for samples without an event-keyed
h5, i.e. HL-LHC PU200). Reads `RecoTrack_*` + jagged
`TruthVertex_assocTracks` with uproot; also stores per-track
`data['track'].truth_pv` for exact evaluation.

**kNN edge construction** (`create_training_graph(knn=...)`): at μ=200 a
fully-connected event has ~118k edges (927 tracks × 128 PVs), 8× the μ≈60
scale. With `knn=k` each track connects only to its k nearest PVs in |Δz|.
Coverage measured on PU200 (200 events, 187k true edges): k=20 keeps 99.50%
of true edges (k=10: 98.8%, k=50: 99.9% — the tail is badly-measured tracks
whose z0 sits tens of mm from their true vertex; those are unlearnable for
any z-based method). `knn=None` (default) = fully connected, bit-exact with
earlier builds.

PV resolution model for truth-graph heights and edge significances:
`σ_PV(N) = A·N^(−B) + C` with presets from
`pv_finder/data/resolution_presets.py` — `run3` (0.238, 0.495, −0.0008,
default in `create_training_graph`) and `hllhc` (0.179, 0.727, 0; default in
`root_to_graphs`).

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

- Nov 2025 baseline **reproduced bit-exactly** end-to-end with this package
  (graphs tensor-identical, eval rows identical). AMVF comparison done: same
  clean rate, PVF+GNN finds 16% more vertices (clean-vertex efficiency 70.0%
  vs 60.9%). See [evaluation](../evaluation/vertex_association.md).
- **PU200 zero-shot**: the μ≈60 checkpoint collapses at μ=200 (clean
  ~45% vs ~77% in-domain; edge purity ~64%) → retraining on PU200 required.
- **PU200 training launched 2026-07-12** (`ttva_gat_pu200_k20`): 30k truth
  graphs (k=20, hllhc preset), split [0.7, 0.25, 0.05], 201 epochs — config
  `configs/gnn/config_gnn_ttva_hllhc.yml`, checkpoints in
  `model_weights/ttva_gnn_hllhc/`, MLflow experiment "TTVA_GNN".
- Training-leg bug fixed on the way: lazy GATConv params must be materialized
  by a dummy forward before the optimizer is created (train_ttva.py); fresh
  training was broken before, only weight-loading paths had been exercised.
- Not yet done: Run 2/3 agreement-with-AMVF eval, PU200 full-chain
  (PVF peaks → GNN) eval, track-level TTVA metrics.

## Known limitations / next steps (from internal note §7)

1. ~~Fully-connected graphs scale poorly to PU200~~ → kNN in |Δz| implemented
   (k=20 default); ranking by significance or (z, d0) still worth exploring.
2. Hard-scatter identification (max sum-pT²) not yet addressed.
3. pT underused — weighting/attention on high-pT tracks expected to help.
4. TTVA is the natural home for HGTD timing features (z ambiguity breaking);
   the PU200_withTiming samples carry the needed branches.
