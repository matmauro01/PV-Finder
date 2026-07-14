# Diagnostics — Vertex Association

Diagnostic and visualization tools for the GNN TTVA model.

## Plotting infrastructure (2026-07-13)

All TTVA figures share `src/gnn/diagnostics/plot_style.py`: mplhep ATLAS
style, Okabe-Ito colourblind-safe palette, common category/algorithm
colours, and a `save_figure` helper (300-dpi PNG + vector PDF). The ATLAS
label defaults to **"Simulation Internal"** (flip `ATLAS_STATUS` once
results are approved for a different label).

Plotting scripts (each reads the JSON/npz written by the matching eval):

| Script | Input | Figures |
|---|---|---|
| `plot_threshold_scan.py` | `threshold_scan` + `edge_metrics` outputs | ROC, score distributions, vertex rates vs t, clean-vertex efficiency vs t, track-level metrics vs t |
| `plot_ttva_performance.py` | t* eval outputs + AMVF + GNN-on-truth | publication bars, clean-vertex efficiency, efficiency vs truth-PV nTracks |
| `plot_ttva_pu200.py` | `evaluate_checkpoints` learning curve | PU200 learning curve, zero-shot vs retrained |
| `plot_ttva_run3.py` | `evaluate_ttva_run3` outputs | Run 3 vs MC score overlay, multiplicity, AMVF agreement |

## PU200 chain diagnostics (2026-07-14)

- `gnn.diagnostics.chain_gap_decomposition` — finder-vs-associator gap
  decomposition on the chain graphs: greedy peak↔truth matching at
  0.3/0.5/1.0 mm, finder cap, oracle-association ceiling (both vertex
  conventions), junk/missed statistics, and the empirical quantile
  distributions that drive `gnn.data.graph_augmentation`
  (`--sigma-events N` re-runs PVF on N events for peak sigmas + saves
  histograms for `verify_fast_paths`). Outputs:
  `outputs/07_14_2026_ttva_gap/{gap_decomposition,augmentation_params}.json`.
- `gnn.diagnostics.plot_ttva_pu200_pub` — publication plots from the
  measured JSONs: yield ladder (algorithms + bounds), chain threshold
  scan (clean/truth + drop-empty fakes), finder-miss vs nTrk. Feeds the
  technical note (`final_paper/.../figures/pu200_*.png`).
- `gnn.diagnostics.hgtd_timing_coverage` — valid-HGTD-time fraction vs
  η on the withTiming samples (PU200: 3.95% overall, 44.6% at |η|>2.3).

## Run3 Track Probability Distribution

**Script:** `src/gnn/diagnostics/run3_track_probability.py`

Runs the full PVF → peak-finding → GNN pipeline on Run3 events (NPZ cache) and
plots the distribution of per-edge GNN association scores.

### Pipeline

1. Load Run3 events from the pre-extracted NPZ cache
   (`data/run3/cache_file3_2000ev_seed42.npz`)
2. For each event, run the PVF model on subevent tensors (100 tracks/chunk,
   accumulated per subevent) to get the 12 000-bin histogram
3. Peak-find with `pv_locations_updated_res` → `pred_z, pred_heights, pred_sigmas`
4. Build a fully-connected bipartite inference graph with `create_inference_graph`
5. Run the GNN → sigmoid edge scores

### Outputs

- `run3_track_probability.{png,pdf}`: two-panel figure
  - Panel 1: distribution of all (track → PV) edge scores (log scale)
  - Panel 2: distribution of max score per track (best PV match per track)
- `run3_track_probability_summary.json`: mean/std/fraction-above-threshold

### Usage

```bash
python -m gnn.diagnostics.run3_track_probability \
    --cache data/run3/cache_file3_2000ev_seed42.npz \
    --pvf-weights model_weights/tracks2kde_KDE_A_z_epoch180.pyt \
    --gnn-weights model_weights/gnn_ttva_epoch100.pyt \
    --legacy-model-path /path/to/atlas_pvfinder/mattia_finder \
    --output-dir outputs/run3_track_probability \
    --nevents 500
```

**`--legacy-model-path`** must point to the `mattia_finder/` directory from the
old codebase, because the PVF weights were pickled with the legacy
`model.autoencoder_models` module path.

### Notes

- ~8–9 s/event on CPU with 300 tracks and ~20 PVF peaks (fully connected graph)
- The score distribution is expected to peak sharply near 0 (most track–PV pairs
  are non-associations) with a secondary peak near 1 (true associations)
- **Bug fixed 2026-07-13**: the `create_inference_graph` call passed
  `(d0_err, z0_err)` where the signature expects `(sig_z0, sig_d0)`, so
  score distributions produced before that date used transposed errors in
  the edge attributes. The truth-free Run 3 evaluation
  (`gnn.evaluation.evaluate_ttva_run3`) supersedes this diagnostic for
  quantitative statements.

## MC Track Probability Distribution

**Script:** `src/gnn/diagnostics/mc_track_probability.py`

Same idea as the Run3 script but on MC events: truth PV positions are used
directly as PV nodes (`create_training_graph`), so no PVF inference step is
needed and the setup matches training exactly.

## Not Yet Migrated

- **event_display.py** (610 lines): Visualizes track-vertex associations per event. Needs splitting to meet 500-line limit.
- **MakePlots_GNN_TTVA.py**: Bar chart summaries of Clean/Merged/Split/Fake rates. Uses ROOT/atlasplots; would need rewrite to matplotlib.
