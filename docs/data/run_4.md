# Run 4 Data (HLLHC PU200)

Simulated HLLHC ttbar + single lepton dataset at μ ≈ 200, used for training the
next-generation PV-Finder model.

## Source file

```
ATLAS_PVFinderData_HLLHC_mc21_14TeV_ttbar_SingleLep_PU200.root
```

- Tree: `PVFinderData`
- 99,800 events
- ~927 tracks/event, ~126 truth primary vertices/event
- **Pileup is discrete — μ ∈ {190, 210}** (not a continuous distribution around 200).
  Use `--mu-min 185 --mu-max 215` on the eval scripts to catch both values in the
  high-pileup summary / category-counts window.

## Converted HDF5

Built by `src/pv_finder/data/root_to_h5.py`:

```
data/run4/hllhc_pu200_training.h5
```

Layout (same convention as the Run 2 / MC flat HDF5):

| Dataset          | Shape                     | dtype   | Notes                                            |
|------------------|---------------------------|---------|--------------------------------------------------|
| `tracks`         | `(N_sub, 7, MAX_TRACKS)`  | float32 | `[d0, z0, d0_err, z0_err, d0_z0_cov, z_start, z_end]`, padded with `-999999` |
| `target_y_split` | `(N_sub, 2, 1000)`        | float16 | Per-subevent truth histogram (ch0: nTracks≥2, ch1: nTracks<2) |
| `target_y`       | `(N_evt, 2, 12000)`       | float16 | Full-event truth histogram, same channels        |
| `pv`             | `(N_evt, MAX_PV)`         | float32 | Truth PV z (mm), padded with `-999999`           |

`N_sub = 12 * N_evt`. No `kde_split` — HLLHC training is end-to-end only; computing
KDEs at μ≈200 is infeasible (~15–30s/event).

## Converter

`src/pv_finder/data/root_to_h5.py` is a two-pass converter:

1. **Pass 1** scans the tree and records `MAX_TRACKS` (per subevent) and `MAX_PV`
   (per event). This lets Pass 2 pre-allocate the HDF5 datasets.
2. **Pass 2** converts in chunks (default 1000 events) and writes them in-place.

### Target histogram generation

Truth histograms are generated on the fly using the LHCb Gaussian-CDF method
(ported from `CreatingTargetHistogram.py`):

- Resolution per truth PV: `σ = A · ntrks^(-B) + C  (mm)` for ntrks ≥ 2,
  `σ = BIN_WIDTH` for ntrks < 2.
- (A, B, C) come from a **resolution preset** selected on the CLI
  (`--resolution-preset`) and are written to the output `h5.attrs` so the
  file self-documents which constants it used.
- Amplitude: Gaussian CDF evaluated at ±5 bins around the PV z position.
- Scaling: `populate = where((0.15/σ) > 1, (0.15/σ)*populate, populate)`.
- Channel 0 aggregates PVs with nTracks ≥ 2, channel 1 aggregates the rest.

#### Resolution presets

| Preset  | A (mm)     | B        | C (mm)       | Source |
|---------|------------|----------|--------------|--------|
| `hllhc` (default) | 0.17898 | 0.7274 | 0.0 | AMVF↔truth residual fit on HL-LHC PU200 ttbar (ITk), 99 800 events, 2026-06-01 (`src/pv_finder/diagnostics/amvf_resolution_vs_ntracks.py`) |
| `run3`  | 0.23817443 | 0.49491396 | -0.000787436 | Legacy `ResolutionFit_ATLAS.ipynb` (Run-3 ID, ⟨μ⟩≈60) |

Override individual parameters with `--a-res`, `--b-res`, `--c-res` to use
custom values without editing the file. To register a new preset for a
recurring sample, add it to `RESOLUTION_PRESETS` in `root_to_h5.py`.

### Usage

```bash
# HL-LHC PU200 ttbar (default preset is 'hllhc')
python -u src/pv_finder/data/root_to_h5.py \
    --input  data/run4/Run4_MC21_ITk/ATLAS_PVFinderData_HLLHC_mc21_14TeV_ttbar_SingleLep_PU200.root \
    --output data/run4/hllhc_pu200_training_v2.h5 \
    --max-events 0

# Re-generate with the legacy Run-3 widths
python -u src/pv_finder/data/root_to_h5.py \
    --input  <run3.root> \
    --output <run3_training.h5> \
    --resolution-preset run3

# One-off override
python -u src/pv_finder/data/root_to_h5.py \
    --input <root> --output <h5> \
    --a-res 0.20 --b-res 0.65 --c-res 0.0
```

Optional: `--max-tracks-per-sub` to cap track padding (default: from Pass 1 scan).

## Training

HLLHC uses the MLP warmup + E2E recipe:

- Phase 1 (50 epochs): MLP-only histogram supervision, UNet disabled.
- Phase 2 (400 epochs): full E2E, MSE loss, Adam lr=1e-3.

Config: `configs/vertex_finding/config_hllhc_pu200_e2e.yml`.

## Beam spot

Simulation is already centered; no beam-spot subtraction required (unlike Run 3).

## PU200 *with timing* samples (`data/run4/PU200_withTiming/`)

A larger re-production (~2.94 M events) that adds **HGTD timing** branches
(`RecoTrack_Time`, `RecoTrack_TimeResolution`) on top of the standard branch set.
Same `PVFinderData` tree and 7-channel track layout as the original PU200 file.

| Sample (DSID) | r-tag(s) | files | events | pileup |
|---|---|---|---|---|
| 601229 ttbar SingleLep | r16438 | 1 | 99,800 | fixed μ = 200 |
| 601229 ttbar SingleLep | r16443 | 1 | 100,000 | **flat μ ≈ 0–210** (mean ~100) |
| 601229 ttbar SingleLep | r16633 | 1 | 99,800 | fixed μ = 200 |
| 601229 ttbar SingleLep | r16638 | 1 | 99,800 | **flat μ ≈ 0–210** (mean ~100) |
| 601237 ttbar all-hadronic | r16633 | 6 (`_1`..`_6`) | ~2,541,000 | fixed μ = 200 |

**Use the fixed-μ=200 tags (r16438, r16633, 601237) for PU200 training.**
r16443 / r16638 carry a flat pileup spectrum (a broad-μ sample, not "PU200").
The per-*track* parameter shapes are identical across all tags; only the
per-event pileup/multiplicity differs.

### Timing branches

- `-1` is the **no-timing sentinel**. Mask it (`Time > -0.999`) before using.
- Only **~3.9 %** of tracks carry real timing — the HGTD covers the forward
  region, and tracks here are capped at |η|<2.5, so only the |η| ≈ 2.4–2.5 edge
  slice gets a time. Acceptance is 0 below |η|≈2.25, rising to ~70–100 % at 2.5.
- Real `Time` ≈ 0 ± 0.2 ns; `TimeResolution` is discrete at ~20 / 25 / 35 ps
  (HGTD hit-multiplicity layers).

### QA

`src/pv_finder/diagnostics/timing_data_qa.py` produces overlaid tracking-parameter
distributions across all samples (+ the old no-timing file as reference). The
2026-06-02 QA confirmed the data is healthy (see
`outputs/06_02_2026_output/timing_data_qa/README.md`). Note: `RecoTrack_chisq`
and `RecoVertex_chisq` are empty in these ntuples (pre-existing upstream
non-fill, also empty in the original PU200 file).

## Scaling the converter for multi-sample training

`root_to_h5.py` now supports two changes designed for the with-timing scale-up
(~2.74 M fixed-μ=200 events across 8 ROOT files):

- **`--compression {lzf,gzip,none}`** (default `lzf`). LZF compresses the
  padded `tracks` tensor to roughly 1/8 of its raw size — most of the
  bytes are the MASK_VAL constant in the unused tail of each subevent,
  which compresses to near-nothing. Lossless; no measurable read-time
  overhead.
- **`--keep-target-y`** (off by default). The HL-LHC end-to-end trainer
  only reads `target_y_split`; the full-event `target_y (N_evt, 2, 12000)`
  is the largest single chunk of disk and isn't used. Skipping it cuts
  another ~120 GB at the 2.74 M-event scale.

Compression and the `has_target_y` flag are written to `h5.attrs` so each
output file self-documents.

### Multi-file training pool

The training pipeline now accepts a list of HDF5 paths as `data_file`:

```yaml
# config_hllhc_pu200_e2e_v4.yml
data_file:
  - data/run4/PU200_withTiming_h5/ATLAS_PVFinderData_601229_e8481_s4494_r16438_PU200.h5
  - data/run4/PU200_withTiming_h5/ATLAS_PVFinderData_601229_e8481_s4494_r16633_PU200.h5
  - data/run4/PU200_withTiming_h5/ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200_1.h5
  # ...
```

`make_tracksHists_dataset` in `src/pv_finder/data/h5_dataset.py` reads each
file's `max_tracks_per_subevent` attribute, takes the global maximum, and
right-pads tracks with the mask sentinel at `__getitem__` time so batches
stack cleanly. Output is a single `torch.utils.data.ConcatDataset`. The
train/val/test split logic is unchanged; it operates on the combined length.

A single-string `data_file:` still works for legacy configs (no padding,
no factory overhead).
