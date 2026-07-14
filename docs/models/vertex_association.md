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

## Status (2026-07-13)

- Nov 2025 baseline **reproduced bit-exactly** end-to-end with this package
  (graphs tensor-identical, eval rows identical). AMVF comparison done. See
  [evaluation](../evaluation/vertex_association.md) for the full results
  matrix.
- **mu60 CURRENT MODEL (2026-07-14)**: v2 retrain (fixed heights +
  augmentation, `ttva_gnn_mu60_v2/ttva_gat_mu60_v2_aug_epoch_115.pyt`) —
  chain clean/truth **0.774 @ 0.02% fakes** (t=0.99, drop-empty) vs AMVF
  0.609; oracle bound 0.819, finder cap 0.852, truth-graph bound 0.892;
  track F1 0.874; **HS-ID 98.98% > AMVF 98.75%**; edge AUC 0.998. The
  legacy e100 checkpoint is kept for reproduction only.
- **PU200 retraining DONE, twice** — v1 (fixed lr, unstable mid-training):
  `model_weights/ttva_gnn_hllhc/ttva_gat_pu200_k20_epoch_175.pyt`; v2
  (cosine + clip, smooth, val −9.5%):
  `model_weights/ttva_gnn_hllhc_v2/ttva_gat_pu200_k20_v2_cosine_epoch_175.pyt`.
  Truth-graph ceiling clean/truth 0.9175 (v2, t=0.95). **v1-e175 is still
  the chain production checkpoint** (transfer gap: v2's fake floor on
  peak-node graphs is ~4.5%).
- **PU200 FULL CHAIN — FINAL (v3, 2026-07-14)**: PVF v4b peaks + GNN v3
  reaches **clean/truth 0.716 @ 0.05% fakes** (t=0.98, drop-empty) vs
  AMVF 0.573 @ 0.91% — **+14.3 pts at 18× lower fakes, 96% of the
  oracle bound** (0.748; finder cap 0.810). HS-ID **98.1% > AMVF
  97.5%**. Production checkpoint:
  `model_weights/ttva_gnn_hllhc_v3/ttva_gat_pu200_k20_v3_aug180k_epoch_156.pyt`
  (train with fixed heights + chain-like augmentation + 180k all-hadronic
  events). v1/v2 kept for reference. Optimized chain latency ≈ 11 ms/event
  (numba peak finder + vectorized selection). See evaluation doc.
- **PV-height bug found + fixed (2026-07-14)**: the Gaussian-CDF height
  recipe added Z_MIN twice (inherited from Nov 2025), so every
  truth-training graph ever built had PV heights ≡ 0 while inference
  graphs carry real peak heights — a major truth→peak transfer-gap
  mechanism. Fixed in `graph_construction.py`; pre-fix graph files keep
  zero heights for reproduction.
- **Chain-like augmentation (2026-07-14)**: `gnn.data.graph_augmentation`
  makes truth graphs statistically chain-like (measured miss-prob vertex
  dropping, dz jitter, peak sigmas/heights, junk-node injection; all
  empirical quantiles from `chain_gap_decomposition`). v3 training uses
  it on 180k all-hadronic events with shard cycling.
- **Run 3 real-data eval (truth-free)**: full PVF+GNN chain vs AMVF
  associations — see evaluation doc, section "Run 3".
- GPU forward is nondeterministic at the ~1e-5 level (GATConv scatter
  atomics): events with near-degenerate top-2 scores can flip between runs
  (~1 event in 2,550). The regression guard in `threshold_scan` allows
  knife-edge flips explicitly.
- Not yet done: PU200 full-chain (PVF peaks → GNN) eval, hard-scatter ID.

## Known limitations / next steps (from internal note §7)

1. ~~Fully-connected graphs scale poorly to PU200~~ → kNN in |Δz| implemented
   (k=20 default); ranking by significance or (z, d0) still worth exploring.
2. Hard-scatter identification (max sum-pT²) not yet addressed.
3. pT underused — weighting/attention on high-pT tracks expected to help.
4. TTVA is the natural home for HGTD timing features (z ambiguity breaking);
   the PU200_withTiming samples carry the needed branches.
