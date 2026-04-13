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

- Resolution per truth PV:
  ```
  σ = 0.23817443 · ntrks^(-0.49491396) − 0.000787436    (ntrks ≥ 2)
  σ = BIN_WIDTH                                          (ntrks < 2)
  ```
- Amplitude: Gaussian CDF evaluated at ±5 bins around the PV z position.
- Scaling: `populate = where((0.15/σ) > 1, (0.15/σ)*populate, populate)`.
- Channel 0 aggregates PVs with nTracks ≥ 2, channel 1 aggregates the rest.

### Usage

```bash
python -u src/pv_finder/data/root_to_h5.py \
    --input  data/run4/ATLAS_PVFinderData_HLLHC_mc21_14TeV_ttbar_SingleLep_PU200.root \
    --output data/run4/hllhc_pu200_training.h5 \
    --max-events 0                    # 0 = all events
```

Optional: `--max-tracks-per-sub` to cap track padding (default: from Pass 1 scan).

## Training

HLLHC uses the MLP warmup + E2E recipe:

- Phase 1 (50 epochs): MLP-only histogram supervision, UNet disabled.
- Phase 2 (400 epochs): full E2E, MSE loss, Adam lr=1e-3.

Config: `configs/vertex_finding/config_hllhc_pu200_e2e.yml`.

## Beam spot

Simulation is already centered; no beam-spot subtraction required (unlike Run 3).
