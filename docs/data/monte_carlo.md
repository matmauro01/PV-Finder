# Monte Carlo Data

Source: `src/pv_finder/data/`

## HDF5 Format

Training data is stored in HDF5 with these datasets:

| Dataset | Shape | Description |
|---------|-------|-------------|
| `tracks` | (N, 7, max_tracks) | Track features |
| `kde_split` | (N, 4, 1000) | 4 KDE channels, 1000 bins |
| `target_y_split` | (N, 2, 1000) | Target histogram + mask |

Total: ~600,000 sub-events.

## Feature Channels

`[d0, z0, d0_err, z0_err, d0_z0_cov, z_start, z_end]`

## Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| N_SUBEVENTS | 12 | Sub-events per event |
| SUBEVENT_WIDTH | 40.0 mm | Z-range per sub-event |
| N_TRACKS_PER_SUBEVENT | 100 | Max tracks per chunk |
| MASK_VAL | -999999 | Empty track slot padding |
| BIN_WIDTH | 0.04 mm | Histogram bin width |

## Data Loading Pipeline

`collectdata_poca_KDE.py` dispatches to the correct H5Dataset class based on the training phase:

| Pipeline string | Dataset class | Returns |
|----------------|---------------|---------|
| `tracks-to-KDE` | H5Dataset_tracksKDE | (tracks, kde) |
| `KDE-to-hist` | H5Dataset_kdeHists | (kde, histogram) |
| `tracks-to-hist` | H5Dataset_tracksHists | (tracks, histogram) |

Train/val/test split: 70% / 15% / 5% (sequential, not random). Indices saved to `.npy` for reproducibility.

## Files

| File | Contents |
|------|----------|
| `collectdata_poca_KDE.py` | Data loading dispatcher |
| `h5_dataset.py` | H5Dataset classes |
