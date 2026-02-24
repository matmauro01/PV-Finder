# Vertex Association Model

Graph Attention Network (GNN) for Track-to-Vertex Association (TTVA).

## Purpose

Given candidate primary vertices (from PVF peak finding or MC truth) and reconstructed tracks, the GNN predicts which tracks belong to which vertex via binary edge classification on a bipartite graph.

## Architecture: TTVAGATModel

**Type:** Heterogeneous bipartite GAT with edge attributes.

**Code:** `src/pv_finder/models/ttva_gnn.py`

```
Track features (8-dim) ──► Linear(8,32) + LeakyReLU ──┐
                                                        ├──► 2x HeteroConv(GATConv, 4 heads) + residual
PV features (2-dim) ────► Linear(2,32) + LeakyReLU ──┘
                                                        │
                                                        ▼
                                              Edge prediction MLP:
                                              cat(track_emb, pv_emb) → 64
                                              Linear(64,32) + LeakyReLU
                                              Linear(32,32) + LeakyReLU
                                              Linear(32,1) → edge logit
```

**Graph structure:**
- Track nodes: `[d0, z0, d0_err, z0_err, cov_d0z0, theta, phi, pT/50000]` — 8 features
- PV nodes: `[z_position, peak_height]` — 2 features
- Edges: fully connected bipartite (every track ↔ every PV)
- Edge attributes: `[longitudinal_significance, horizontal_significance, |dz|]` — 3 features

**Output:** Per-edge logits. Apply sigmoid for association scores in [0,1].

## Graph Construction

**Code:** `src/pv_finder/data/graph_construction.py`

Two modes:
- `create_training_graph()`: from MC truth HDF5 — includes truth edge labels
- `create_inference_graph()`: from PVF peak finding output — no labels

Edge attributes:
- Longitudinal significance: `(z0_track - z_pv) / sqrt(sigma_z0² + sigma_pv²)`
- Horizontal significance: `d0 / sigma_d0`
- Absolute distance: `|z0_track - z_pv|`

PV resolution modeled as: `sigma_pv(N) = 0.238 * N^(-0.495) - 0.001` (fitted from MC).

## Trained Weights

Existing checkpoint: `test_GATConv_edgeattr_BCE_100.pyt` (100 epochs, BCE loss).
State dict keys are backward-compatible with the original `TTVA_GATGraphConv_Model`.
