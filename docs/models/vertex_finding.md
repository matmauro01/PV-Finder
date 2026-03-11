# Vertex Finding Models

Source: `src/pv_finder/models/`

## Architecture

Three-phase pipeline, each with its own model:

```
Phase 1: Tracks → KDE        (MaskedDNN)
Phase 2: KDE → Histogram     (UNet_1000)
Phase 3: Tracks → Histogram  (trackstoHists_UNet_1000, combines Phase 1 + 2)
```

### MaskedDNN (Phase 1)

Fully-connected network. Maps 7 track features to a 1000-bin KDE.
Masks out invalid tracks (z_0 <= -240). Output is summed per-track contributions.

- Input: (batch, 7, num_tracks)
- Output: (batch, 1000)
- Layers: 5 FC layers, 100 hidden nodes each, LeakyReLU + Softplus output

### UNet_1000 (Phase 2)

1-D UNet with skip connections. Refines a KDE into the final vertex histogram.

- Input: (batch, n_features, 1000)
- Output: (batch, 1000)
- 4-level encoder-decoder with ConvBNrelu blocks

### trackstoHists_UNet_1000 (Phase 3)

End-to-end model: MLP (same as MaskedDNN) feeding into UNet (same as UNet_1000).
Can be initialized from pretrained Phase 1 + 2 weights via `initialize_combined_weights.py`.

- Input: (batch, 7, num_tracks)
- Output: (batch, 1000)

### UNet_1000_v2 (Phase 2 — improved)

Source: `src/pv_finder/models/unet_v2.py`

Redesigned K2H model targeting the sidelobe artifact (spurious peaks 0.25–1.0 mm
from real peaks). Three architectural changes vs UNet_1000:

| | UNet_1000 | UNet_1000_v2 |
|---|---|---|
| Upsampling | ConvTranspose1d(k=2, s=2) | F.interpolate(nearest) + Conv1d(k=3) |
| Bottleneck | 125 bins (8x, 3 pool stages) | 250 bins (4x, 2 pool stages) |
| First kernel | k=25 (1.0 mm) | k=15 (0.6 mm) |
| Skip connections | Concat (channel doubling) | Additive |
| Encoder blocks | ConvBNrelu | ResConvBNrelu (residual) |
| Output head | Conv(k=5) → Conv(k=5) | Conv(k=5) → Conv(k=3) → Conv(k=1) |
| Parameters | ~221K | ~156K |

**Rationale**: The original UNet produces sidelobes because (1) ConvTranspose with
kernel=stride creates seam artifacts, (2) 8x downsampling makes sharp peaks
sub-resolution at bottleneck, (3) k=25 correlates bins across the full sidelobe range.
See JOURNAL.md 2026-03-11 entry for investigation details.

## Loss Functions

- **MSELoss**: standard, used in Phase 3 (recommended)
- **Asymmetric loss** (`alt_loss_A.py`): from LHCb PV-Finder, emphasizes false positive suppression. Used in Phase 2.

## Standard Configuration

- KDE channel: **A_z only** (out_idx=0, n_latent_channels=1)
- 4-channel variant exists but A_z is the production standard

## Files

| File | Contents |
|------|----------|
| `autoencoder_models.py` | All model classes (1628 lines, needs splitting) |
| `unet_v2.py` | UNet_1000_v2 — sidelobe-free K2H model |
| `alt_loss_A.py` | Asymmetric loss function |
