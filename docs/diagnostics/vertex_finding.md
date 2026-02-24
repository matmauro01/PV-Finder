# Diagnostics — Vertex Finding

Source: `src/pv_finder/diagnostics/`

## Feature Distribution Comparison (MC vs Run 3)

Compares track parameter distributions between Monte Carlo training data and
real ATLAS Run 3 data to quantify domain shift.

### Files

| File | Lines | Purpose |
|------|-------|---------|
| `compare_feature_distributions.py` | 356 | Entry point: CLI, summary, JSON export |
| `feature_plots_1.py` | 387 | Figures 1--3: core params, 2D correlations, tensor distributions |
| `feature_plots_2.py` | 369 | Figures 4--6: CDF/QQ, per-subevent stats, beam spot |

Data loading and feature extraction live in `src/pv_finder/data/feature_loading.py` (389 lines).

### Figures Produced

1. **Core Track Parameters** (4x2): d0, z0, d0_err, z0_err, d0_z0_cov histograms + track count + log-scale tails
2. **2D Correlations** (2x3): hexbin plots of feature pairs (MC vs Run 3)
3. **Tensor Distributions** (4x2): per-channel model-input distributions + OOD fraction bar chart
4. **Tails & Quantiles** (2x3): CDF comparisons + QQ plot for d0_err
5. **Per-Subevent Statistics** (2x2): mean d0, d0_err, track count, std(d0) vs z-position
6. **Beam Spot Investigation** (2x2): z0 shift analysis between MC origin and Run 3 beam spot

### Usage

```bash
PYTHONPATH=src python src/pv_finder/diagnostics/compare_feature_distributions.py \
    --run3-cache data/run3/cache_file3_2000ev_seed42.npz \
    --mc-h5 data/monte_carlo/training_data.h5 \
    --output-dir outputs/domain_shift_investigation \
    --n-events 200
```

### Critical Note on Feature Mapping

The MC H5 channels are:
- Ch0: d0 (RAW, mm) -- NOT d0*2
- Ch1: z0 (mm)
- Ch2: d0_err (mm) -- NOT theta
- Ch3: z0_err (mm) -- NOT phi
- Ch4: d0_z0_cov (mm^2)
- Ch5: z_start (mm)
- Ch6: z_end (mm)

Previous scripts had bugs decoding Ch0/2/3. This code uses the correct mapping.

## KDE Model vs Analytical Comparison

Compares the T2KDE neural network predictions against analytically computed KDE
on both MC validation data and Run 3 ATLAS data. Measures how well the model
reproduces the ground-truth KDE, and whether domain shift degrades performance.

### Files

| File | Lines | Purpose |
|------|-------|---------|
| `compare_kde_model_vs_analytical.py` | 389 | Entry point: CLI, metrics, JSON export |
| `analytical_kde.py` | 288 | Per-track 2D Gaussian KDE (numpy, no numba) |
| `kde_model_inference.py` | 223 | Model loading (pickle alias fix) and batched inference |
| `kde_comparison_plots.py` | 413 | All visualizations: overlays, per-vertex, summaries |

### Algorithm

The analytical KDE replicates the ACTS_KDE_generation.ipynb algorithm:
1. For each z-bin (1000 bins per subevent), filter active tracks (z0 ± 3σ_z)
2. Coarse scan: 60×60 grid in [-0.6, 0.6] mm, evaluate 2D Gaussian PDFs
3. Fine scan: 7×7 refinement around best coarse point (step = 0.01 mm)
4. KDE value = maximum sum of PDFs across all scan points

### Model

Best T2KDE model: `tracks2kde_KDE_A_z_epoch180.pyt` (MaskedDNN, 7→1000, 5×100 hidden).
Loaded via pickle with module alias fix (`model.autoencoder_models` → `pv_finder.models.autoencoder_models`).

### Outputs

```
outputs/kde_comparison/
├── mc_event_overlays/        # Full-event KDE overlays (analytical + model + truth)
├── run3_event_overlays/      # Full-event KDE overlays (analytical + model)
├── mc_per_vertex/            # Per-vertex zoom plots (MC)
├── run3_per_vertex/          # Per-vertex zoom plots (Run 3)
├── mc_agreement_summary.*    # MC agreement scatter/histogram plots
├── run3_agreement_summary.*  # Run 3 agreement plots
├── mc_residual_distributions.*  # MC residual histograms and CDFs
├── run3_residual_distributions.*
├── mc_vs_run3_comparison.*   # Side-by-side MC vs Run 3 degradation
└── kde_comparison_summary.json
```

### Usage

```bash
PYTHONPATH=src venv/bin/python3 -m pv_finder.diagnostics.compare_kde_model_vs_analytical \
    --n-events 200 --n-viz-events 3 --output-dir outputs/kde_comparison
```

### Key Findings (200 events)

The analytical KDE matches the pre-computed H5 truth (kde_split) to Pearson r = 1.0000,
confirming the algorithm is correct.

| Metric | MC (val) | Run 3 |
|--------|----------|-------|
| Pearson r (mean ± std) | 0.911 ± 0.103 | 0.912 ± 0.112 |
| Event RMSE | 10.47 | 5.87 |
| Integral ratio | 0.916 | 0.908 |
| Peak matching rate | 93.3% (277/297) | 77.9% (710/911) |

The model generalises well to Run 3 data (similar Pearson r), but misses more peaks
(77.9% vs 93.3%), likely due to the domain shift in track multiplicity and errors.

## Data Exploration Notebook

Interactive Jupyter notebook for quick exploration of MC and Run 3 features.

Location: `src/pv_finder/scratch/data_exploration.ipynb`

Covers: basic stats, 1D distributions, 2D correlations, track multiplicity, beam spot,
and direct ROOT file exploration (requires `uproot`).
