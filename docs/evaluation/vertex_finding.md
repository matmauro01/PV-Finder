# Evaluation — Vertex Finding

Evaluation of PV-Finder vertex finding on MC test data.

## Script

`src/pv_finder/evaluation/vertex_finding/run_eval_pvf.py`
Plotting helpers: `src/pv_finder/evaluation/vertex_finding/plots_pvf.py`

Two pipeline modes (mutually exclusive):

| Flag | Pipeline |
|------|----------|
| _(default)_ | Analytical KDE_A_z (h5) → K2H (UNet_1000) |
| `--e2e-model` | Raw tracks → trackstoHists_UNet_1000 (no KDE stage) |

## CLI Options

All options have sensible defaults — only the model checkpoint is required.

| Flag | Default | Description |
|------|---------|-------------|
| `--e2e-model` | — | E2E checkpoint (mutually exclusive with `--k2h-model`) |
| `--k2h-model` | — | K2H checkpoint (mutually exclusive with `--e2e-model`) |
| `--h5` | standard data path | HDF5 file |
| `--root-truth` | standard ROOT path | ROOT truth file; also loads qibin automatically |
| `--indices` | `configs/test_main_indices_2550evt.p` | Test event indices |
| `--output-dir` | `outputs/eval_pvf` | Output directory |
| `--device` | `0` | CUDA device (-1 = CPU) |

## How to Run

```bash
source venv/bin/activate

# Canonical — E2E v1 Run 2 MC model, all defaults:
python src/pv_finder/evaluation/vertex_finding/run_eval_pvf.py \
    --e2e-model model_weights/03_24_2026/reproduction_T2HIST_400ep_T2KDE100_K2H150_epoch_300_fullstate.pth \
    --e2e-type v1 \
    --output-dir outputs/eval_mc_T2HIST_400ep_ep300 \
    --title "Run 2 MC — T2HIST 400ep ep300" --device 1

# K2H stage-2, custom output dir:
python src/pv_finder/evaluation/vertex_finding/run_eval_pvf.py \
    --k2h-model model_weights/reproduction_KDE2HIST_matmauro_200epochs_epoch_190_fullstate.pth \
    --output-dir outputs/eval_k2h

# Without ROOT truth (faster, no nTracks filter):
python src/pv_finder/evaluation/vertex_finding/run_eval_pvf.py \
    --e2e-model model_weights/... --root-truth "" --output-dir outputs/eval_no_root
```

## Test Set

- **File:** `configs/test_main_indices_2550evt.p`
- **Events:** h5 indices 48450–50999 (2550 events = last 5% of 51000)
- **Subevents:** 581400–611999 (30600 subevents, split [0.7, 0.25, 0.05])
- Matches `mattia_finder` test set exactly.

## Key Design Decisions

### Peak-finding thresholds

| Parameter | Value (default) | Used for |
|-----------|-------|---------|
| `threshold` | `1e-2` | Min bin height to start a peak |
| `INTEGRAL_THRESHOLD` | `0.5` | Min peak area — performance counts (clean/merged/split/fake, efficiency, FP, all peak-count plots) |
| `INTEGRAL_THRESHOLD_RES` | `0.5` | Min peak area — σ_vtx_vtx pairwise Δz fit |
| `min_width` | `3` bins | Min peak width |

**Unified-threshold (2026-04-16):** Both performance and resolution use `0.5`
by default on Run 2 / Run 3 / MC. This is consistent and filters small sidelobe
peaks out of both metrics. Rationale:

- Dual-threshold (`0.2` perf + `0.5` res) was misleading: sidelobes counted as
  fakes in performance but were hidden from the resolution plot. Using `0.5`
  for both ensures consistent accounting.
- The `clean_run3` reference eval uses a single threshold (`0.4`) — same spirit.
- History: original sidelobe investigation thought E2E training had fixed the
  problem, but it was actually the stricter resolution threshold hiding them.

**HL-LHC PU200 override:** PU200 peaks are shallower (track density spread across
more vertices). Pass `--integral-threshold 0.2 --integral-threshold-res 0.2` to
avoid losing real vertices. Full scan results in `outputs/04_15_2026_output/thr_scan_hllhc/`.

### ROOT truth vs h5 truth

Without `--root`: truth PVs from h5 `pv` field — **no nTracks filter**. All truth PVs included, which inflates `reco_merged` count.

With `--root` + `--qibin`: truth PVs from `ATLAS_PVFinderData_TruthMatched.root`, filtered to **nTracks ≥ 2**. Matches `mattia_finder` exactly.

### h5 ↔ ROOT event index mapping

The h5 file uses reindexed ("pubindices") event ordering — h5 event `i` ≠ ROOT event `i`. The correct ROOT index for the `k`-th sequential test event is `qibin[k]`, stored in `configs/qibin_test_main_indices_v2.p` (copied from `mattia_finder/config/`).

### Pileup filter for summary

Summary statistics (clean/merged/split/fake averages) are computed only over events with **55 ≤ ActualNumOfInt ≤ 65** (from ROOT). This matches `mattia_finder`'s `plot_tracks2hist.py` convention. Overall efficiency across all events is also printed.

### Pairwise Δz for σ_vtx_vtx

`pv_locations_updated_res` returns PVs sorted ascending in z, so all pairwise differences `pvs[i]-pvs[j]` for `i<j` are negative. Both `+dz` and `-dz` are added to make the distribution symmetric before fitting the sigmoid.

## Outputs

| File | Contents |
|------|----------|
| `resolution_plot.png` | Pairwise Δz histogram + sigmoid fit → σ_vtx_vtx |
| `performance_plot.png` | Clean/merged/split/fake fractions and efficiency vs pileup |
| `stats_histogram.png` | **Total reconstructed PVs/event vs pileup — PV-Finder (Σ clean+merged+split+fake) vs AMVF** (nTracks≥2). Two curves, SEM error bars, overall-mean annotation box. AMVF source: `n_amvf` (MC eval via `RecoVertex_nTracks`) or `n_truth` (real-data eval, where truth already is AMVF). |
| `reco_vs_mu.png` | Same idea as `stats_histogram` but also overlays the MC truth reference (dashed gray). MC eval only (requires `--root-truth`). |
| `category_counts_hist.png` | **5-bar summary** of per-event reco counts in the high-pileup window `μ ∈ [55, 65]`: **Total, Clean, Merged, Split, Fake**. Bars labeled with mean value on top, SEM error bars, pileup window + n_events + checkpoint metadata in the corner box. |
| `eval_results.pkl` | All per-event results, pred/truth PV positions, fit params |

## Model Checkpoints

| Model | File | Notes |
|-------|------|-------|
| **Run 2 MC — canonical (Model B, ep300)** | `model_weights/03_24_2026/reproduction_T2HIST_400ep_T2KDE100_K2H150_epoch_300_fullstate.pth` | E2E v1, 400-epoch Qi Bin reproduction, initialized from T2KDE ep100 + K2H ep150. `trackstoHists_UNet_1000` with default width (64 UNet ch, [100]×5 MLP). **Default for all Run 2 MC / Run 2 data / Run 3 data evals.** Previously used ep150 — moved to ep300 on 2026-04-15 (later in the 400-epoch schedule, more converged). |
| **HLLHC PU200 — canonical (v2 wide)** | `model_weights/hllhc_pu200_mlp50_e2e400_v2_phase2_epoch_100_fullstate.pth` | E2E v1 **wide** variant (`n_UNetChannels=96`, `l_HiddenNodes=[128]×5`, 680K params). Load via `--e2e-wide`. Phase 2 epoch 100, which is **150 effective epochs** counting the 50-epoch MLP warmup in Phase 1. LR-stable recipe: 1e-4 + 5-ep warmup + cosine decay + grad-clip. |
| E2E v1 ep130 (Strategy B, older) | `model_weights/e2e_mlpHist50_e2e400_1latent_mse_phase2_epoch_130_fullstate.pth` | 50-ep MLP warmup + 400-ep E2E (`train_mlp_hist_then_e2e.py`). The "old Run 2 model" reference used in the 2026-04-09 HLLHC-vs-Run2 comparison. Default-width v1. |
| E2E v1 ep191 (tracks→hist) | `model_weights/tracks2hist_1channel_200epochs_epoch_191_fullstate.pth` | Manually extracted from a mattia_finder `.pyt` artifact (see Outstanding Issues). |
| E2E v2 ep90 (TracksToHist_v2) | `model_weights/T2HIST_v2_100epochs_epoch_90_fullstate.pth` | |
| K2H v1 ep190 | `model_weights/reproduction_KDE2HIST_matmauro_200epochs_epoch_190_fullstate.pth` | |
| K2H v2 ep190 | `model_weights/K2H_v2_interp_200epochs_epoch_190_fullstate.pth` | |
| T2KDE ep130 | `model_weights/reproduction_KDE_A_z_matmauro_run1_200_epoch_130_fullstate.pth` | |

The E2E checkpoint was extracted from the mattia_finder MLflow artifact (`.pyt` full model → state dict) using the `pvfinder` conda env, since the `.pyt` format embeds the `model` module path.

## Differences vs mattia_finder evaluate_model.py

This table is the ground truth for what is and isn't matched:

| Aspect | mattia_finder | Our script | Status |
|--------|--------------|-----------|--------|
| Truth source | ROOT `TruthVertex_z`, nTracks≥2 | Same (via `--root-truth`) | ✅ Matched |
| h5↔ROOT index mapping | `qibin_test_main_indices_v2.p` | Same | ✅ Matched |
| Peak finder thresholds (performance) | threshold=0.01, int=0.2, width=3 | Same | ✅ Matched |
| Peak finder thresholds (σ) | threshold=0.01, int=0.5, width=3 | Same | ✅ Matched |
| Pileup variable | `ActualNumOfInt` (float, rounded) | Same when ROOT available | ✅ Matched |
| Summary pileup filter | μ∈[55,65] | Same | ✅ Matched |
| NaN filter on predicted PVs | Called but disabled (dead code) | Not applied | ✅ Equivalent |
| Matching window units | Bins | Bins (when ROOT) | ✅ Matched |
| **Pairwise Δz for σ** | **One sign only (negative)** | **Both ±dz** | ⚠️ Intentional difference |
| **Sigmoid fit range** | All bins | All bins | ✅ Matched (clean_run3 excludes ±0.3mm centre — we don't) |

The **one intentional difference**: we add both `+dz` and `-dz` for each pair, giving a symmetric distribution. mattia_finder only stores the negative direction (PVs are sorted ascending so `pvs[i]-pvs[j]` is always negative). Our approach is more correct; the fitted σ may differ slightly in value.

## Moving Parts — Things to Be Aware Of

### qibin mapping
`configs/qibin_test_main_indices_v2.p` maps sequential test event position → ROOT event index. It has **exactly 2550 entries**, one per test event (h5 indices 48450–50999). If you change the test set or h5 file, this mapping is **invalid** and needs to be regenerated from mattia_finder.

### ActualNumOfInt
- A **float** from ROOT (e.g. 58.3), **rounded** to the nearest integer for pileup binning.
- Used for: (1) summary table filter μ∈[55,65], (2) x-axis of performance and stats histogram plots.
- Not available without ROOT — falls back to N truth PVs.
- Distinct from `NumRecoVtx` (number of reconstructed AMVF vertices, also in ROOT) and from N truth PVs (from h5/ROOT after nTracks≥2 filter).

### σ_vtx_vtx is both output and input
σ_vtx_vtx is computed from the pairwise Δz distribution, then **used as the matching window** in `compare_res_reco`. This creates a mild circular dependency: a very different model will give a different σ, which changes how clean/merged/fake are counted. Keep this in mind when comparing numbers across very different models.

### Integral threshold — 0.5 for both (2026-04-16)

Unified threshold: both performance and resolution use `0.5` by default
(`INTEGRAL_THRESHOLD = 0.5`, `INTEGRAL_THRESHOLD_RES = 0.5`). This is
consistent: peaks counted as fakes in performance also appear in the
resolution plot's pairwise Δz. Small sidelobe peaks (integral < 0.5)
are filtered out of both. Overridable via `--integral-threshold` and
`--integral-threshold-res`.

**HL-LHC PU200 override:** peaks are shallower due to PU200 track density
spread. Pass `--integral-threshold 0.2 --integral-threshold-res 0.2`
explicitly for HL-LHC evals to avoid losing real vertices.

For reference: `clean_run3` uses `0.4` for both. Our 0.5 is slightly
stricter but same single-threshold property.

### Pileup filter scope
μ∈[55,65] applies **only to the printed summary table**. The performance plot and stats histogram use **all events**. Both bounds are overridable via `--mu-min` / `--mu-max` (e.g. `--mu-min 195 --mu-max 205` for HLLHC PU200).

### Adaptive sigmoid fit initial guess
The pairwise-Δz sigmoid fit uses initial parameters computed from the actual histogram
(`a = baseline − min(counts)`, `c = median(counts)`, `b = 10`, `rcc = 0.5`). This
adapts to widely different count scales — Run 2 (~1000/bin), HLLHC PU200 (~10000/bin) —
without hand-tuning. The older fixed `FIT_P0 = [1000, 10, 30, 0.8]` failed on HLLHC.

### E2E checkpoint format
mattia_finder saves full model objects (`.pyt`) embedding the `model` module path. These cannot be loaded directly from PV-Finder's venv. Extraction procedure: load with `conda run -n pvfinder python -c "... ckpt.state_dict() ..."` then `torch.save({"model_state": sd, "epoch": N}, "...fullstate.pth")`.

---

## Real Data Evaluation (Run 2 / Run 3)

`src/pv_finder/evaluation/vertex_finding/run_eval_pvf_run3.py`
Data loading: `src/pv_finder/data/run3_io.py`

Evaluates PV-Finder on real collision data (Run 2 or Run 3), using AMVF reconstructed vertices (nTracks >= 2) as the reference baseline. There is no MC truth on real data. The same script handles both Run 2 and Run 3 — the ROOT format is identical.

### Modes

| Flags | Pipeline |
|-------|----------|
| `--t2kde-model` + `--k2h-model` | Tracks → T2KDE (MaskedDNN) → K2H (UNet_1000) |
| `--e2e-model` + `--e2e-type v1` | Tracks → trackstoHists_UNet_1000 end-to-end |
| `--e2e-model` + `--e2e-type v1 --e2e-wide` | Same class, but wider (96 UNet ch, [128]×5 MLP) — for the HLLHC v2 checkpoint |
| `--e2e-model` + `--e2e-type v2` | Tracks → TracksToHist_v2 end-to-end |

The same script also runs on **HLLHC PU200** ROOT files (Run 4) — the tree layout
is identical. Pass `--mu-min`/`--mu-max` to move the summary window from the Run 2/3
default of `[55, 65]` to something like `[195, 205]` for PU200.

### Data Sources (mutually exclusive)

| Flag | Source |
|------|--------|
| `--root` | ROOT file directly via uproot (supports `--entry-start`/`--entry-stop`) |
| `--npz` | Pre-extracted NPZ cache (faster, 2000 events) |

### How to Run

```bash
source venv/bin/activate

# Full pipeline (T2KDE + K2H), from ROOT file, 500 events:
python src/pv_finder/evaluation/vertex_finding/run_eval_pvf_run3.py \
    --root data/run3/file_3.root \
    --t2kde-model model_weights/reproduction_KDE_A_z_matmauro_run1_200_epoch_130_fullstate.pth \
    --k2h-model model_weights/reproduction_KDE2HIST_matmauro_200epochs_epoch_190_fullstate.pth \
    --max-events 500 --entry-stop 10000 --output-dir outputs/eval_run3_pipeline

# E2E model, from NPZ cache:
python src/pv_finder/evaluation/vertex_finding/run_eval_pvf_run3.py \
    --npz data/run3/cache_file3_2000ev_seed42.npz \
    --e2e-model model_weights/e2e_mlpHist50_e2e400_1latent_mse_phase2_epoch_130_fullstate.pth \
    --max-events 300 --output-dir outputs/eval_run3_e2e

# Run 2 real data (same script — identical ROOT format):
python src/pv_finder/evaluation/vertex_finding/run_eval_pvf_run3.py \
    --root data/run2/Run2_Data/.../user.rgarg.49035490.EXT0._000002.ATLAS_PVFinderData_Run3Data.root \
    --t2kde-model model_weights/reproduction_KDE_A_z_matmauro_run1_200_epoch_130_fullstate.pth \
    --k2h-model model_weights/reproduction_KDE2HIST_matmauro_200epochs_epoch_190_fullstate.pth \
    --max-events 2500 --output-dir outputs/eval_pvf_run2

# HLLHC PU200 — same script, custom pileup window:
python src/pv_finder/evaluation/vertex_finding/run_eval_pvf_run3.py \
    --root data/run4/ATLAS_PVFinderData_HLLHC_mc21_14TeV_ttbar_SingleLep_PU200.root \
    --e2e-model model_weights/hllhc_pu200_mlp50_e2e400_phase2_epoch_100_fullstate.pth \
    --e2e-type v1 --mu-min 195 --mu-max 205 \
    --max-events 2550 --output-dir outputs/eval_hllhc_ep100 \
    --title "PVF — HLLHC PU200 — ep100"
```

### Real Data vs MC Differences

| Aspect | MC (`run_eval_pvf.py`) | Real data (`run_eval_pvf_run3.py`) |
|--------|----------------------|-------------------------------|
| Data format | Flat HDF5 with pre-split subevents | ROOT or NPZ with variable-length track arrays |
| Pre-computed KDEs | Available (Stage 2 shortcut) | Not available — must run full pipeline |
| Ground truth | MC truth PVs | AMVF vertices (nTracks >= 2) |
| Beam correction | Not needed (MC beam at origin) | **Applied by default** (subtracts BeamPosZ from AMVF z) |
| Pileup (μ) | ActualNumOfInt from ROOT | ActualNumOfInt from ROOT; unavailable from NPZ |
| Subevent building | Pre-split in HDF5 | Built on-the-fly from track z0 positions |
| Run 2 specifics | — | μ ≈ 25–30 peak, BeamPosZ ≈ -2.5 mm |
| Run 3 specifics | — | μ ≈ 60 peak, BeamPosZ varies |

### Beam Correction

Beam correction is **on by default** (`--no-correct-beam` to disable). AMVF vertex z positions are beam-corrected while PVFinder operates in the detector frame (from track z0). Subtracting BeamPosZ from AMVF z aligns the two coordinate systems. Without correction, you'll see a systematic ~BeamPosZ offset and artificially low efficiency.

### Outputs

Same as MC eval: `resolution_plot.png`, `performance_plot.png`, `stats_histogram.png`, `eval_results.pkl`.

### Post-Processing: Smoothing + NMS

Two optional post-processing steps reduce fake sidelobe peaks on Run 3 data.
The E2E model produces UNet deconvolution sidelobes — small spurious peaks
0.5–0.85 mm from real vertex peaks, caused by over-resolving broad KDE features
in high track-density regions. These inflate the fake rate and contaminate the
resolution plot.

| Flag | Default | Description |
|------|---------|-------------|
| `--smooth-sigma` | `0` (off) | Gaussian sigma (bins) applied to histogram before peak finding only |
| `--nms-min-sep` | `0` (off) | Remove shorter peak if pair closer than this (mm) |
| `--nms-max-ratio` | `0.3` | Only suppress if short/tall height ratio < this |

**How it works:**

1. **Gaussian pre-smoothing** blurs the predicted histogram before peak finding
   (original preserved for all other purposes). Narrow sidelobe fluctuations get
   absorbed into their parent peak. Kills ~0.75 fake/evt, ~0.05 real/evt.

2. **NMS** (`suppress_neighbor_peaks()` in `efficiency_res_optimized_atlas.py`)
   scans pairs of peaks within `min_sep` mm. If the shorter peak's height is
   < `max_ratio` × the taller, the shorter is suppressed (tallest-first ordering).
   Preserves genuine close vertex pairs (similar heights) while killing sidelobe
   fakes (3–5× shorter than parent). Kills ~1.4–2.1 fake/evt depending on ratio.

**Two operating points:**

| Config | Eff | Fake/evt | Real lost/evt | Fake:real | sigma |
|--------|:---:|:---:|:---:|:---:|:---:|
| No PP (baseline) | 97.6% | 4.80 | — | — | 0.347 mm |
| s=2 + NMS(0.85, **0.3**) | 97.4% | 2.70 | 0.14 | **10:1** | 0.487 mm |
| s=2 + NMS(0.85, **0.5**) | 96.9% | 2.00 | 0.50 | **3.7:1** | 0.592 mm |

- **NMS 0.3** (conservative): 90.9% of removed peaks are fake; 98% purity in the
  0.55–0.65 mm sidelobe core. Barely touches real vertices.
- **NMS 0.5** (aggressive): halves the fake rate but removes 0.50 real/evt.

Sigma increases with PP because fake close pairs that artificially pulled it
down get removed — the post-PP sigma is a more honest measure.

**Example (conservative):**

```bash
python src/pv_finder/evaluation/vertex_finding/run_eval_pvf_run3.py \
    --root data/run3/file_3.root \
    --e2e-model model_weights/e2e_mlpHist50_e2e400_1latent_mse_phase2_epoch_130_fullstate.pth \
    --smooth-sigma 2.0 --nms-min-sep 0.85 --nms-max-ratio 0.3 \
    --output-dir outputs/eval_run3_s2_nms03
```

### NMS Diagnostic Script

`src/pv_finder/evaluation/vertex_finding/nms_diagnostic.py` — re-runs inference
on a subset of events, identifies which peaks NMS removes, classifies them as
real (truth-matched) or fake, and generates per-vertex zoom plots.

```bash
python src/pv_finder/evaluation/vertex_finding/nms_diagnostic.py \
    --root data/run3/file_3.root \
    --e2e-model model_weights/e2e_mlpHist50_e2e400_1latent_mse_phase2_epoch_130_fullstate.pth \
    --entry-start 300000 --entry-stop 300200 \
    --output-dir outputs/nms_diagnostic --device 0
```

Outputs: `removal_stats.png` (4-panel summary), `zoom_plots/` (~40 per-vertex
3-panel plots with analytical KDE + track scatter), `removed_peaks_summary.pkl`.

---

## Outstanding Issues

1. **E2E checkpoint extraction** — `tracks2hist_1channel_200epochs_epoch_191_fullstate.pth` was manually extracted. Other epoch checkpoints have not been extracted. Automate if needed.

2. **σ_vtx_vtx fit differences vs clean_run3** — clean_run3 excludes central |x|≤0.3 mm bins from the sigmoid fit and tries a Gaussian notch fit first; PV-Finder fits all bins with a sigmoid only. clean_run3 uses different peak-finding thresholds (threshold=0.02, integral=0.4, width=2 vs our 0.01/0.2 counts / 0.5 resolution). Our dual-threshold design (0.2 for counts, 0.5 for resolution) is intentional — the scan on 2026-04-15 confirmed 0.2 is the right operating point for the peak-count plots while 0.5 keeps the pairwise-Δz sigmoid fit clean.

3. **No nTracks in h5** — the flat h5 `pv` field has only z positions. The nTracks≥2 filter requires ROOT. Running without `--root-truth` gives unfiltered truth (more merged, lower clean counts).
