# Evaluation — Vertex Finding

Two-step evaluation of the PV-Finder UNet on MC test data.

## Pipeline

```
Step 1 (Resolution): PVF model → inference → peak finding → pairwise distances → fit σ_vtx-vtx
Step 2 (Classification): peaks + truth histogram peaks + σ_vtx-vtx → match → Clean/Merged/Split/Fake
```

## Usage

```bash
# Full run (inference + resolution + classification):
PYTHONPATH=src python -m pv_finder.evaluation.evaluate_pvf \
    --pvf-weights model_weights/pvf_e2e_epoch400.pyt \
    --pvf-h5 data/monte_carlo/training_data.h5 \
    --n-events 2550 \
    --output-dir outputs/evaluation/pvf_e400_2550evt \
    --device 0 \
    --threshold 0.01 --integral-threshold 0.5 --min-width 3

# Classify-only (reuse saved histograms, let sigma be fitted):
PYTHONPATH=src python -m pv_finder.evaluation.evaluate_pvf \
    --histograms outputs/evaluation/pvf_e400_2550evt/pvf_histograms.npy \
    --output-dir outputs/evaluation/pvf_rerun \
    --threshold 0.01 --integral-threshold 0.5 --min-width 3

# Classify-only with a fixed sigma (skip resolution fit):
PYTHONPATH=src python -m pv_finder.evaluation.evaluate_pvf \
    --histograms outputs/evaluation/pvf_e400_2550evt/pvf_histograms.npy \
    --sigma-vtx-vtx 0.34 \
    --output-dir outputs/evaluation/pvf_sigma034 \
    --threshold 0.01 --integral-threshold 0.5 --min-width 3
```

`--track-h5` is accepted for backwards compatibility but is no longer used.

## Peak-Finding Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.01 | Min bin value to start a peak |
| `integral_threshold` | 0.5 | Min sum of bin values in a region |
| `min_width` | 3 | Min consecutive above-threshold bins |

Shared algorithm: `pv_finder.utils.peak_finding.pv_locations_updated_res`.

### Conjoined peak splitting

`pv_locations_updated_res` includes a **conjoined-peak split**: when two nearby peaks
overlap and the histogram never dips below `threshold` between them, a local minimum is
detected (histogram starts rising again after having already fallen) and the region is
split into two separate PV candidates.

This is critical for the resolution plot. Without it, pairs of true vertices at
separations of 0.3–1 mm that produce overlapping KDE peaks are merged into one predicted
PV and never contribute a pair to the Δz histogram. The fitted σ_vtx-vtx then reflects
the KDE overlap scale (~0.8 mm) rather than the true detector resolution (~0.34 mm).

## Resolution Fitting (Step 1)

For each event, find peaks in the predicted histogram and collect all pairwise
z-distances between predicted PVs (peaks are shuffled before computing pairs so
differences are symmetric around zero). Histogram these distances in [−6, 6] mm
(61 bins) and fit a sigmoid:

```
f(x) = a / (1 + exp(b * (rcc - |x|))) + c
```

The parameter `rcc` = σ_vtx-vtx is the vertex-vertex resolution: the scale at which
two nearby true vertices can be resolved into two separate predicted peaks. Used as the
matching window in Step 2.

## Vertex Classification (Step 2)

Truth positions come from **peak-finding on the truth KDE histograms**
(`pvf_truth_histograms.npy`), using the same algorithm and parameters as prediction
peak-finding. Raw MC generator-level positions (`pv_loc_z`) are not used: the model
predicts KDE peaks, which spatially blur nearby vertices, so comparing against individual
MC positions with a small window gives a spurious ~70% fake rate.

Each predicted PV is matched to truth peaks using σ_vtx-vtx (converted to bins) as
the matching window:

| Category | Definition |
|----------|-----------|
| **Clean** | Exactly one truth PV within the matching window |
| **Merged** | Multiple truth PVs within the matching window |
| **Split** | Multiple reco PVs match the same truth PV (closest kept, rest become Split) |
| **Fake** | No truth PV within the matching window |

All four are reco-side quantities; bar-chart percentages sum to 100% of predicted PVs.

## Test Split

Events 48450–50999 (2,550 events, never seen during training).
Subevent indices: 581400–611999 (12 subevents per event).

## Output Files

```
outputs/evaluation/pvf_e400_2550evt/
├── pvf_histograms.npy          # (2550, 12000) float32 predictions
├── pvf_truth_histograms.npy    # (2550, 12000) float32 truth KDE
├── deltaz_resolution.png/pdf   # Vertex distance fit plot
├── pvf_results.json            # Summary metrics
├── pvf_per_event.npy           # (2550, 4) [clean, merged, split, fake]
└── pvf_category_bar.png        # Category distribution bar chart
```

## Code

- MC evaluation script: `src/pv_finder/evaluation/evaluate_pvf.py`
- Run 3 evaluation script: `src/pv_finder/evaluation/evaluate_pvf_run3.py`
- Vertex matching: `src/pv_finder/evaluation/vertex_matching.py`
- Peak finding: `src/pv_finder/utils/peak_finding.py`

---

# Run 3 Evaluation

Evaluation of the PVF UNet on real Run 3 proton–proton collision data.
Unlike the MC pipeline, there are no truth KDE histograms; truth comes from
AMVF reconstructed vertices (beam-spot-corrected).

## Pipeline

```
Step 1 (Resolution): ROOT tracks → inference → peak finding → PVF pairwise Δz  ┐
                     AMVF reco vertices → AMVF pairwise Δz                      ├ overlay plot
                     Fit sigmoid to each → σ_PVF, σ_AMVF                       ┘
Step 2 (Classification): PVF peaks + AMVF truth + σ_PVF → match → Clean/Merged/Split/Fake
```

## Usage

```bash
PYTHONPATH=src python -m pv_finder.evaluation.evaluate_pvf_run3 \
    --model model_weights/pvf_e2e_epoch400.pyt \
    --root-file data/run3/pvfinder_data.root \
    --output-dir outputs/evaluation/pvf_run3_e400 \
    --n-events 2000 \
    --device 0 \
    --threshold 0.01 --integral-threshold 0.5 --min-width 3

# Skip resolution fit (use known sigma):
PYTHONPATH=src python -m pv_finder.evaluation.evaluate_pvf_run3 \
    --model model_weights/pvf_e2e_epoch400.pyt \
    --root-file data/run3/pvfinder_data.root \
    --output-dir outputs/evaluation/pvf_run3_sigma034 \
    --sigma-vtx-vtx 0.34 \
    --n-events 2000
```

## Key Differences from MC Evaluation

| Aspect | MC (`evaluate_pvf.py`) | Run 3 (`evaluate_pvf_run3.py`) |
|--------|----------------------|-------------------------------|
| Input format | H5 subevents | ROOT file via `uproot` |
| Truth source | KDE histogram peaks | AMVF reco vertices (nTracks ≥ 2) |
| Track loading | Pre-batched subevents | Built on-the-fly from raw branches |
| Resolution plot | PVF only | PVF + AMVF overlaid |
| Efficiency metric | LHCb-style (S, MT, FP) | AMVF efficiency (clean+merged)/n_amvf |

## ROOT Branches Used

| Branch | Description |
|--------|-------------|
| `RecoTrack_z0` | Track z₀ impact parameter |
| `RecoTrack_d0` | Track d₀ impact parameter |
| `RecoTrack_ErrD0` | d₀ error |
| `RecoTrack_ErrZ0` | z₀ error |
| `RecoTrack_ErrD0Z0` | d₀–z₀ covariance |
| `RecoVertex_z` | AMVF vertex z position |
| `RecoVertex_nTracks` | Tracks per AMVF vertex |
| `BeamPosZ` | Beam-spot z (subtracted from vertex z) |

## Output Files

```
outputs/evaluation/pvf_run3_e400/
├── deltaz_resolution.png/pdf   # PVF + AMVF pairwise distance overlay
├── pvf_run3_results.json       # Summary metrics
└── pvf_category_bar.png        # Category distribution bar chart
```

### `pvf_run3_results.json` fields

| Field | Description |
|-------|-------------|
| `n_events` | Events with valid AMVF truth |
| `n_amvf_truth` | Total AMVF vertices used as truth |
| `avg_amvf_per_event` | Mean AMVF vertices per event |
| `clean/merged/split/fake` | PVF reco-side category counts |
| `n_reco` | Total predicted PVs |
| `amvf_efficiency` | (clean + merged) / n_amvf_truth |
| `sigma_pvf_mm` | Fitted PVF resolution σ |
| `sigma_amvf_mm` | Fitted AMVF resolution σ |
