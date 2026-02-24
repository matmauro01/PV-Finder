# Evaluation — Vertex Finding

Two-step evaluation of the PV-Finder UNet on MC test data.

## Pipeline

```
Step 1 (Resolution): PVF model → inference → peak finding → pairwise distances → fit σ_vtx-vtx
Step 2 (Classification): peaks + truth PVs + σ_vtx-vtx → match → Clean/Merged/Split/Fake/Missed
```

## Usage

```bash
# Full run (inference + resolution + classification):
PYTHONPATH=src python -m pv_finder.evaluation.evaluate_pvf \
    --pvf-weights model_weights/pvf_e2e_epoch400.pyt \
    --pvf-h5 data/monte_carlo/training_data.h5 \
    --track-h5 data/monte_carlo/track_associations.h5 \
    --n-events 2550 \
    --output-dir outputs/evaluation/pvf_e400_2550evt \
    --device 0 \
    --threshold 0.01 --integral-threshold 0.5 --min-width 3 --prominence 0.85

# Classify-only (reuse saved histograms):
PYTHONPATH=src python -m pv_finder.evaluation.evaluate_pvf \
    --histograms outputs/evaluation/pvf_e400_2550evt/pvf_histograms.npy \
    --track-h5 data/monte_carlo/track_associations.h5 \
    --sigma-vtx-vtx 0.34 \
    --output-dir outputs/evaluation/pvf_rerun \
    --threshold 0.01 --integral-threshold 0.5 --min-width 3 --prominence 0.85
```

## Peak-Finding Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.01 | Min bin value to start a peak |
| `integral_threshold` | 0.5 | Min sum of bin values in a region |
| `min_width` | 3 | Min consecutive above-threshold bins |
| `prominence` | 0.85 | Reserved for future conjoined peak splitting |

Shared algorithm: `pv_finder.utils.peak_finding.pv_locations_updated_res`.

## Resolution Fitting (Step 1)

For each event, find peaks in the predicted histogram and collect all pairwise z-distances between predicted PVs. Histogram these distances in [-6, 6] mm (61 bins) and fit a sigmoid:

```
f(x) = a / (1 + exp(b * (rcc - |x|))) + c
```

The parameter `rcc` = σ_vtx-vtx is the vertex-vertex resolution. This is then used as the matching window in Step 2.

## Vertex Classification (Step 2)

Each predicted PV is matched to truth PVs (from `track_associations.h5`, filtered to nTracks >= 2) using the fitted σ_vtx-vtx as the matching window (converted to bins):

| Category | Definition |
|----------|-----------|
| **Clean** | Exactly one truth PV within the matching window |
| **Merged** | Multiple truth PVs within the matching window |
| **Split** | Multiple reco PVs match the same truth PV (closest kept, others split) |
| **Fake** | No truth PV within the matching window |
| **Missed** | Truth PV not matched by any reco PV |

## Test Split

Events 48450–50999 (2,550 events, never seen during training).
Subevent indices: 581400–611999 (12 subevents per event).

## Output Files

```
outputs/evaluation/pvf_e400_2550evt/
├── pvf_histograms.npy          # (2550, 12000) float32 predictions
├── pvf_truth_histograms.npy    # (2550, 12000) float32 truth
├── deltaz_resolution.png/pdf   # Vertex distance fit plot
├── pvf_results.json            # Summary metrics
├── pvf_per_event.npy           # (2550, 5) [clean, merged, split, fake, n_truth]
└── pvf_category_bar.png        # Category distribution bar chart
```

## Code

- Evaluation script: `src/pv_finder/evaluation/evaluate_pvf.py`
- Vertex matching: `src/pv_finder/evaluation/vertex_matching.py`
- Peak finding: `src/pv_finder/utils/peak_finding.py`
