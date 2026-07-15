"""Vertex z-resolution presets used to set per-PV Gaussian widths.

Sigma model: ``sigma_z(n) = A * n^(-B) + C  [mm]`` where ``n`` is the truth
N_Tracks of a primary vertex. The same model is consumed by
``root_to_h5.py`` when building target histograms.

To add a preset, add an entry to both ``RESOLUTION_PRESETS`` and
``RESOLUTION_PRESET_SOURCES`` so that the source attribution is recorded.
The chosen (A, B, C) end up in the HDF5's ``h5.attrs`` so any file
self-documents which resolution model produced it.
"""

from __future__ import annotations

RESOLUTION_PRESETS: dict[str, tuple[float, float, float]] = {
    # name: (A_mm, B, C_mm)
    "hllhc": (0.17898, 0.7274, 0.0),
    "hllhc_corrected": (0.1239, 0.4583, -0.0073),
    "run3": (0.23817443, 0.49491396, -0.000787436),
}

RESOLUTION_PRESET_SOURCES: dict[str, str] = {
    "hllhc": (
        "AMVF<->truth fit on HL-LHC PU200 ttbar (ITk), 99 800 events, "
        "produced 2026-06-01 by amvf_resolution_vs_ntracks.py "
        "(see outputs/06_01_2026_output/amvf_resolution_residuals/fit_params.json). "
        "SUPERSEDED 2026-07-15: per-bin widths were inflated by wrong-match "
        "background inside the 2 mm matching window; see 'hllhc_corrected'."
    ),
    "hllhc_corrected": (
        "Background-corrected refit of the same residuals (Gaussian core + flat "
        "background per bin), produced 2026-07-15 by resolution_fit_v2.py "
        "(outputs/07_15_2026_note_figs/resolution_fit_v2.json). Statistical "
        "1/sqrt(n) scaling at half the Run 3 amplitude."
    ),
    "run3": (
        "Run-3 fit from ResolutionFit_ATLAS.ipynb / "
        "CreatingTargetHistogram.py upstream (ATLAS Inner Detector, mu~60)."
    ),
}

DEFAULT_RESOLUTION_PRESET = "hllhc"
