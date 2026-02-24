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
