# Diagnostics — Vertex Association

Diagnostic and visualization tools for the GNN TTVA model.

## Run3 Track Probability Distribution

**Script:** `src/pv_finder/diagnostics/run3_track_probability.py`

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
python -m pv_finder.diagnostics.run3_track_probability \
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

## Not Yet Migrated

- **event_display.py** (610 lines): Visualizes track-vertex associations per event. Needs splitting to meet 500-line limit.
- **MakePlots_GNN_TTVA.py**: Bar chart summaries of Clean/Merged/Split/Fake rates. Uses ROOT/atlasplots; would need rewrite to matplotlib.
