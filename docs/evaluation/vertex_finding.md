# Evaluation — Vertex Finding

Evaluation of PV-Finder vertex finding on MC and Run 3 data.
Ported from `atlas_pvfinder/mattia_finder` (Feb 2026).

## Code

| File | Description |
|------|-------------|
| `src/pv_finder/evaluation/vertex_matching.py` | Peak finding (6-tuple), vertex categorization, resolution fitting |
| `src/pv_finder/evaluation/evaluate_pvf.py` | MC evaluation (pre-computed pickles + ROOT truth) |
| `src/pv_finder/evaluation/evaluate_pvf_run3.py` | Run 3 evaluation (ROOT input, AMVF truth, on-the-fly inference) |
| `src/pv_finder/utils/peak_finding.py` | Shared peak finding (4-tuple), used by diagnostics |

### Peak finding: two versions

The eval pipeline uses `_pv_locations_updated_res` from `vertex_matching.py`, which
returns a 6-tuple `(z_positions, peak_values, peak_bins, conjoined_left, conjoined_right, sigmas)`.
The diagnostics code uses `pv_locations_updated_res` from `utils/peak_finding.py`, which
returns a 4-tuple `(z_positions, peak_heights, peak_bins, sigmas)`. Both implement
conjoined-peak splitting. They will be unified after testing.

---

## MC Evaluation (`evaluate_pvf.py`)

Classifies pre-computed PV-Finder outputs against MC truth from ROOT files.

**Prerequisites:** Run `TestModel.py` first to generate pickled inputs, labels, and outputs.

### Usage

```bash
PYTHONPATH=src python -m pv_finder.evaluation.evaluate_pvf \
    -o outputs/evaluation/pvf_results \
    -m unet \
    -f data/monte_carlo/training_data.h5 \
    -r data/monte_carlo/pvfinder_data.root \
    -i test_indices.p \
    -s 0.34
```

### CLI Arguments

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `-o`, `--dirname` | yes | — | Directory with pickled PVF outputs |
| `-m`, `--modelname` | yes | — | Model name (e.g. `unet`, `unetplusplus`) |
| `-f`, `--path_hdf5` | yes | — | Path to input H5 file |
| `-r`, `--path_root` | yes | — | Path to ROOT file (same data as H5) |
| `-i`, `--indices` | yes | — | Pickled index array for train/test split |
| `-s`, `--sigma` | yes | — | σ_vtx-vtx matching window (mm) |
| `-n`, `--nevents` | no | 51000 | Total events in input |
| `--use_label_truth` | no | off | Use KDE label peaks instead of ROOT truth |
| `--label_peak_thresh` | no | 0.1 | Peak height threshold for label truth |

### Truth Sources

- **Default (ROOT):** Reads `TruthVertex_x/y/z/nTracks` from the ROOT file,
  filtered to nTracks ≥ 2.
- **Label-based (`--use_label_truth`):** Finds peaks in the KDE labels using
  `scipy.signal.find_peaks`. All peaks are valid (dummy nTracks = 10).

### Peak-Finding Parameters (hardcoded)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `threshold` | 0.01 | Min bin value to start a peak |
| `integral_threshold` | 0.2 | Min integral for a valid peak |
| `min_width` | 3 | Min consecutive above-threshold bins |

### Efficiency Metrics

Two metrics are computed:

1. **Custom efficiency:** fraction of truth vertices (nTracks ≥ 2) classified
   as "clean" or "merged" by `compare_res_reco`.

2. **LHCb-style efficiency:** computed by `pv_finder.utils.efficiency.efficiency`
   with parameters `difference=5.0`, `threshold=1e-2`, `integral_threshold=0.2`,
   `min_width=3`. Returns (S, Sp, MT, FP).

### Output Files

Pickled dictionaries keyed by pileup (rounded `ActualNumOfInt`):

```
{dirname}/
├── separated_all_pvf_{model}.p      # total reco per event by pileup
├── separated_clean_pvf_{model}.p
├── separated_merged_pvf_{model}.p
├── separated_split_pvf_{model}.p
├── separated_fake_pvf_{model}.p
├── separated_eff_pvf_{model}.p      # per-event efficiency by pileup
├── truth_correct_pvf_{model}.p      # per-truth-PV correct flag
├── truth_ntrks_pvf_{model}.p        # per-truth-PV nTracks
├── total_reco_z_{model}.p           # all reco z positions (mm)
└── predicted_pv_distances_pvf_{model}.p  # all pairwise Δz
```

---

## Run 3 Evaluation (`evaluate_pvf_run3.py`)

End-to-end evaluation on real Run 3 data. Builds subevent tensors from ROOT
tracks, runs the PVF model, and classifies against AMVF reconstructed vertices.

### Usage

```bash
PYTHONPATH=src python -m pv_finder.evaluation.evaluate_pvf_run3 \
    --model model_weights/pvf_e2e_epoch400.pyt \
    --input data/run3/pvfinder_data.root \
    --output-dir outputs/evaluation/pvf_run3 \
    --nevents 2000 \
    --device 0
```

### CLI Arguments

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--model` | yes | — | Path to trained model (.pyt) |
| `--input` | yes | — | Input file (.pkl or .root) |
| `--output-dir` | yes | — | Output directory |
| `--nevents` | no | 1000 | Number of events to process |
| `--seed` | no | 42 | Random seed |
| `--sigma-vtx-vtx` | no | 0.34 | Resolution for matching (mm) |
| `--threshold` | no | 0.02 | Peak finding threshold |
| `--integral-threshold` | no | 0.4 | Integral threshold |
| `--min-width` | no | 2 | Min peak width (bins) |
| `--device` | no | 0 | GPU device (-1 for CPU) |
| `--batch-size` | no | 600 | GPU batch size (sub-events) |

### Pipeline

1. Load tracks from ROOT/pkl (`Track_z0/d0/theta/phi/ErrD0Z0/ErrZ0`)
2. Build 12 subevent tensors (7 features, 100 tracks each, padded to -999999)
3. Run PVF model on GPU in batches
4. Stitch 12 subevent outputs into full 12000-bin histogram
5. Peak-find with conjoined splitting
6. Match against AMVF truth (beam-corrected, nTracks ≥ 2) using `compare_res_reco`
7. Generate resolution plot (PVF + AMVF overlay) and category bar chart

### Known Issue: Feature Engineering

The current code uses the **mattia_finder feature mapping** which is known to
be wrong for the PV-Finder training setup:

| Channel | Current (wrong) | Correct |
|---------|----------------|---------|
| 0 | d0 / 2 | d0 |
| 1 | z0 | z0 |
| 2 | theta / 3 | d0_err |
| 3 | (phi + π) / 3 | z0_err |
| 4 | sig_d0_z0 | d0_z0_cov |

This will be fixed as a follow-up after testing the matching/classification logic.

### Output Files

```
{output-dir}/
├── run3_eval_summary.json
├── pvfinder_deltaz_resolution.png/pdf   # PVF + AMVF overlay
├── pvfinder_vs_amvf_bar.png/pdf         # Category bar chart
├── run3_eval_separated_clean.p
├── run3_eval_separated_merged.p
├── run3_eval_separated_split.p
├── run3_eval_separated_fake.p
├── run3_eval_truth_ntrks.p
├── run3_eval_truth_correct.p
├── run3_eval_total_reco_z.p
└── run3_eval_all_pred_hists.p
```

---

## Vertex Matching (`vertex_matching.py`)

Core matching logic, ported from `efficiency_res_optimized_atlas.py`.

### Functions

| Function | Description |
|----------|-------------|
| `_pv_locations_updated_res` | Peak finding with conjoined splitting (6-tuple return) |
| `filter_nans_res` | Filter predicted peaks that land on NaN regions in labels |
| `get_reco_resolution` | Per-peak FWHM-based resolution measurement |
| `compare_res_reco` | Classify predicted PVs as Clean/Merged/Split/Fake |
| `compare_res_reco2` | Extended version with per-peak resolution support |
| `fit_sigma_vtx_vtx` | Fit sigmoid to pairwise Δz distribution |
| `make_resolution_plot` | Generate Δz histogram + sigmoid fit plot |

### Vertex Categories

| Category | Definition |
|----------|-----------|
| **Clean** | Exactly one truth PV within the σ_vtx-vtx matching window |
| **Merged** | Multiple truth PVs within the matching window |
| **Split** | Multiple reco PVs match the same truth PV (closest kept, rest = Split) |
| **Fake** | No truth PV within the matching window |

### Resolution Fitting

Histogram all pairwise z-distances between predicted PVs in [−6, 6] mm
(61 bins) and fit a sigmoid:

```
f(x) = a / (1 + exp(b * (rcc - |x|))) + c
```

The parameter `rcc` = σ_vtx-vtx is the vertex-vertex resolution.

### Bugs Fixed During Port

| Bug | Location | Fix |
|-----|----------|-----|
| `[[]]*n` shallow copy | `compare_res_reco` | `[[] for _ in range(n)]` |
| `MT_total+=eff[3]` | `evaluate_pvf.py` | `FP_total+=eff[3]` (index 3 is FP, not MT) |
