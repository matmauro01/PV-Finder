# Run 3 Data

Real ATLAS Run 3 proton-proton collision data.

## Formats

- ROOT files — raw ATLAS data
- HDF5 — preprocessed track features and target histograms
- NPZ — cached event data for fast evaluation

## Beam Spot

On real data, subtract `BeamPosZ` from AMVF vertex z but do NOT subtract it from track z0.

(to be expanded as data code is migrated)
