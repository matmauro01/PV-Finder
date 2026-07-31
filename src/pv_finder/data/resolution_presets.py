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
    "hllhc_alleta": (0.14318, 0.36200, -0.01797),
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
    "hllhc_alleta": (
        "Background-corrected fit on the EXTENDED-|eta| July-2026 re-production "
        "(data/run4_all_etas, 601229 r16633, 99 800 events, 2.31M truth vertices "
        "with nTrk>=2), produced 2026-07-31 by amvf_resolution_vs_ntracks.py then "
        "resolution_fit_v2.py (outputs/07_31_2026_output/amvf_resolution_alleta/). "
        "Use this for any |eta|<4 sample: at fixed truth nTracks the AMVF-truth "
        "residual is 13-26% WIDER than at |eta|<2.5, because nTracks now counts "
        "forward tracks whose sigma(z0) is up to 2.8 mm. Do NOT reuse "
        "'hllhc_corrected' on this data. Naive (uncorrected) fit for reference: "
        "A=0.2218, B=0.7203, C=0. chi2/ndf=60 is a large-N artifact (0.18 um "
        "errors on the n=2 bin); relative accuracy is 1-3% over n=2-100. sigma "
        "reaches the 2 um floor only at n=231, above the observed max of 163."
    ),
    "run3": (
        "Run-3 fit from ResolutionFit_ATLAS.ipynb / "
        "CreatingTargetHistogram.py upstream (ATLAS Inner Detector, mu~60)."
    ),
}

DEFAULT_RESOLUTION_PRESET = "hllhc"
