# Diagnostics — Vertex Finding

Source: `src/pv_finder/diagnostics/domain_shift_investigation/`

## Feature Distribution Comparison (MC vs Run 3)

Compares track parameter distributions between Monte Carlo training data and
real ATLAS Run 3 data to quantify domain shift.

### Files

| File | Lines | Purpose |
|------|-------|---------|
| `domain_shift_investigation/feature_distribution/compare_feature_distributions.py` | 356 | Entry point: CLI, summary, JSON export |
| `domain_shift_investigation/feature_distribution/feature_plots_1.py` | 387 | Figures 1--3: core params, 2D correlations, tensor distributions |
| `domain_shift_investigation/feature_distribution/feature_plots_2.py` | 369 | Figures 4--6: CDF/QQ, per-subevent stats, beam spot |

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
PYTHONPATH=src python -m pv_finder.diagnostics.domain_shift_investigation.feature_distribution.compare_feature_distributions \
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
| `domain_shift_investigation/kde_study/compare_kde_model_vs_analytical.py` | 389 | Entry point: CLI, metrics, JSON export |
| `domain_shift_investigation/kde_study/analytical_kde.py` | 288 | Per-track 2D Gaussian KDE (numpy, no numba) |
| `domain_shift_investigation/kde_study/kde_model_inference.py` | 223 | Model loading (pickle alias fix) and batched inference |
| `domain_shift_investigation/kde_study/kde_comparison_plots.py` | 413 | All visualizations: overlays, per-vertex, summaries |

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
PYTHONPATH=src venv/bin/python3 -m pv_finder.diagnostics.domain_shift_investigation.kde_study.compare_kde_model_vs_analytical \
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

**Note on old atlas_pvfinder results**: The ~15x amplitude suppression on Run 3 reported
in `atlas_pvfinder/clean_run3` was caused by feature encoding bugs in the old scripts
(compare_kde_theory_vs_model.py, investigate_domain_shift.py). They fed theta/3 and
(phi+pi)/3 into channels 2--3 instead of d0_err and z0_err, and used d0/2 instead of
raw d0. For MC this cancelled out (round-trip encoding); for Run 3 it fed the model
angular values where it expected uncertainties, causing output collapse. The "60% OOD"
finding was comparing d0_err against theta/3. With correct feature encoding, the model
shows no catastrophic domain shift.

## Per-Vertex Histogram Visualization

Visualizes e2e model histogram predictions vs analytical KDE on a per-vertex basis,
for both MC and Run 3 data. Shows histogram overlays zoomed around individual
truth/AMVF vertices, full-event overviews, and track scatter panels.

### Files

| File | Lines | Purpose |
|------|-------|---------|
| `per_vertex_visualization/run_per_vertex.py` | ~275 | Entry point: CLI, orchestration |
| `per_vertex_visualization/inference.py` | ~163 | Load e2e model, run batched inference (all tracks, no truncation) |
| `per_vertex_visualization/peak_matching.py` | ~120 | Peak finding (shared algorithm), vertex-window matching, truth vertex loading |
| `per_vertex_visualization/vertex_plots.py` | ~464 | Two-panel overview and three-panel per-vertex zoom figures |

Shared dependency: `src/pv_finder/utils/peak_finding.py` (~110 lines) — peak-finding
algorithm shared with evaluation.

### Models

**e2e (tracks→histogram):** per-track MLP (7→1000 bins) + masked sum over all tracks
+ UNet refinement → (1000,) histogram per subevent.
Weights: `model_weights/e2e_mlpHist50_e2e400_1latent_mse_phase2_epoch_130.pyt`

**T2KDE (tracks→KDE):** MaskedDNN that predicts the analytical KDE from raw tracks.
Weights: `model_weights/tracks2kde_KDE_A_z_epoch180.pyt`. Optional — if the model
file is not found, the T2KDE curve is simply omitted from the plots.

All tracks per subevent are fed without truncation. Batch-internal padding to max
N_tracks per batch, maskVal = -240.0.

### Figures Produced

**Overview figure** (per event, PNG + PDF, `event{N:04d}/event{N:04d}_overview.{png,pdf}`):
- Two panels: full z-range histogram overlay (e2e, analytical KDE, T2KDE model, MC truth target) + residual
- ±0.5mm shaded bands around each truth/AMVF vertex
- Filled dots = peaks within a band; open circles = peaks outside

**Per-vertex zoom figure** (per vertex, PNG, `event{N:04d}/event{N:04d}_vtx{V:02d}_z{z:.1f}mm.png`):
- Panel 1: histogram overlay ±8mm around truth vertex, ±0.5mm band, peak markers,
  all visible truth vertices (focused=black, others=grey)
  - Red solid: e2e predicted histogram
  - Blue dashed: analytical KDE (rescaled)
  - Orange dash-dot: T2KDE model (rescaled)
  - Green dotted (MC only): truth target histogram (rescaled)
- Panel 2: residual strip (e2e − analytical KDE)
- Panel 3: track scatter z₀ vs |d₀|/σ_d₀, coloured by log₁₀(σ_d₀)

All panels share x-axis for proper alignment. Y-axis shows raw e2e histogram values;
KDE, T2KDE, and truth target are rescaled to the e2e global peak for shape comparison.

### Usage

```bash
PYTHONPATH=src venv/bin/python3 -m pv_finder.diagnostics.per_vertex_visualization.run_per_vertex \
    --n-events 3 --output-dir outputs/per_vertex [--device cpu] [--window-mm 8] \
    [--t2kde-model-path model_weights/tracks2kde_KDE_A_z_epoch180.pyt]
```

Output tree:
```
outputs/per_vertex/
├── mc/
│   ├── event0000/
│   │   ├── event0000_overview.{png,pdf}
│   │   ├── event0000_vtx00_z-42.9mm.png
│   │   └── ...
│   └── event0001/
└── run3/
    ├── event0000/
    │   ├── event0000_overview.{png,pdf}
    │   └── ...
    └── event0001/
```

### Truth Vertex Sources

- **MC**: generator-level z-positions from H5 `pv` dataset (shape (51000, 92), filter ≤ −500 padding)
- **Run 3**: beam-corrected AMVF vertices: `RecoVertex_z − BeamPosZ`, filter `nTracks ≥ 2`

Note: Run 3 track z0 values are in the detector frame while AMVF vertices are
beam-corrected. The offset is typically O(1 mm) and within the ±0.5mm matching window.

### Peak Finding

Uses `pv_locations_updated_res` from `src/pv_finder/utils/peak_finding.py` — the same
algorithm used by evaluation metrics. Scans contiguous above-threshold regions with
integral and width criteria (no conjoined-peak splitting):
- `threshold = 0.01` (minimum bin value)
- `integral_threshold = 0.5` (minimum region integral)
- `min_width = 3` (minimum consecutive bins)

Each region yields exactly one peak at the weighted-mean z-position.
A vertex is considered "matched" if at least one histogram peak falls within ±0.5mm.

## Data Exploration Notebook

Interactive Jupyter notebook for quick exploration of MC and Run 3 features.

Location: `src/pv_finder/scratch/data_exploration.ipynb`

Covers: basic stats, 1D distributions, 2D correlations, track multiplicity, beam spot,
and direct ROOT file exploration (requires `uproot`).

## AMVF Resolution vs N_Tracks (Figure 12 reproduction)

Reproduces the AMVF z-resolution-vs-N_Tracks plot from ATL-PHYS-PUB-2023-011 Fig. 12
on HL-LHC PU200 ttbar MC, and extracts the (a, b, c) parameters of
`sigma_z(n) = a / n^b + c` for setting target-histogram Gaussian widths during
PV-Finder training.

Script: `src/pv_finder/diagnostics/amvf_resolution_vs_ntracks.py`

### Method

1. Load HL-LHC PU200 ttbar ROOT via `pv_finder.data.run3_io.load_run3_from_root`
   (exposes AMVF reco vertices + `TruthVertex_z` + `TruthVertex_nTracks`).
2. Per event: greedy closest-first 1-to-1 match between AMVF reco and truth
   within `--match-window` (default 2 mm). ~90% AMVF vertices match.
3. Bin matched pairs by truth N_Tracks (26 bins, 2 to 140). Per bin: build a
   `TH1F` of `dz = z_AMVF - z_truth` and fit a Gaussian (`TF1 "gaus"`,
   range ±2.5·RMS) to extract sigma(n).
4. Fit `sigma(n) = a / n^b + c` to the (centre, sigma) points using
   `TGraphErrors.Fit(TF1)` with parameter limits.
5. Plot with PyROOT + atlasplots mimicking Qi Bin's `sample_plotting_code.py`
   style (red star marker `MARKER_AMVF=29`, dashed power-law fit,
   `atlasplots.atlas_label`, TLatex tags).

Supports `--replot-from-npz <vertex_data.npz>` for fast iteration without
re-walking ROOT.

### Dependencies

`PyROOT 6.24` and `atlasplots` are not in the venv -- they live in the system
anaconda. The script prepends `/usr/local/anaconda3/lib/python3.8/site-packages`
to `sys.path` automatically. You only need to add `src` for the `pv_finder.*`
package imports:

```bash
source venv/bin/activate
PYTHONPATH=src python -u src/pv_finder/diagnostics/amvf_resolution_vs_ntracks.py \
    --max-events 20000
```

### Latest fit (HL-LHC PU200 ttbar, 20 000 events, 1.71M matched pairs)

| Param | Value | Units | Interpretation |
|-------|-------|-------|----------------|
| `a` | 171.99 ± 8.96 | μm | sigma at n=1 (extrapolated) |
| `b` | 0.7241 ± 0.0179 | -- | power-law exponent |
| `c` | 0.00 ± 0.14 | μm | irreducible floor |

Output: `outputs/06_01_2026_output/amvf_resolution_residuals/`
