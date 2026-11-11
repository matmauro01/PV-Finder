"""Optional GBT peak filter for run_eval_pvf.py (histogram-only features).

Moved out of run_eval_pvf.py unchanged in behaviour, to keep that script
under the 500-line limit. The classifier is the ``gbt_hist_model`` saved by
``pv_finder.diagnostics.peak_classifier_v2``.
"""

from __future__ import annotations

import pickle

import numpy as np

Z_MIN = -240.0  # mm, full-event histogram lower edge
BIN_WIDTH = 0.04  # mm
N_BINS_FULL = 12000


def compute_hist_features(
    peak_z: float,
    peak_h: float,
    hist: np.ndarray,
    all_peaks_z: np.ndarray,
    all_peaks_h: np.ndarray,
) -> np.ndarray:
    """Eight histogram-shape and neighbourhood features for one peak."""
    bin_idx = int((peak_z - Z_MIN) / BIN_WIDTH)
    bin_idx = max(0, min(bin_idx, N_BINS_FULL - 1))

    # Local integral (+-0.5 mm = +-12.5 bins)
    hw = 13
    lo_i, hi_i = max(0, bin_idx - hw), min(N_BINS_FULL, bin_idx + hw + 1)
    local_integral = float(np.sum(hist[lo_i:hi_i]))

    # Asymmetry: right vs left integral within 0.5 mm
    left_int = float(np.sum(hist[lo_i:bin_idx]))
    right_int = float(np.sum(hist[bin_idx + 1 : hi_i]))
    hist_skewness = (right_int - left_int) / max(right_int + left_int, 1e-6)

    # FWHM approximation
    half_max = peak_h / 2.0
    fwhm_bins = 0
    for direction in [-1, 1]:
        j = bin_idx
        while 0 <= j < N_BINS_FULL and hist[j] >= half_max:
            fwhm_bins += 1
            j += direction
    fwhm_mm = fwhm_bins * BIN_WIDTH

    # Curvature at the peak (more negative = sharper)
    if 1 <= bin_idx <= N_BINS_FULL - 2:
        curvature = float(hist[bin_idx - 1] + hist[bin_idx + 1] - 2 * hist[bin_idx])
    else:
        curvature = 0.0

    # Height relative to the local background
    bg_lo = max(0, bin_idx - 50)
    bg_hi = min(N_BINS_FULL, bin_idx + 50)
    bg_median = float(np.median(hist[bg_lo:bg_hi]))
    rel_height = peak_h / max(bg_median, 1e-6)

    # Nearest other peak
    others_mask = np.abs(all_peaks_z - peak_z) > 0.01
    others = all_peaks_z[others_mask]
    if len(others) > 0:
        nearest_peak_dz = float(np.min(np.abs(others - peak_z)))
        nearest_idx = np.argmin(np.abs(others - peak_z))
        nearest_peak_ratio = peak_h / max(
            float(all_peaks_h[others_mask][nearest_idx]),
            1e-6,
        )
    else:
        nearest_peak_dz = 999.0
        nearest_peak_ratio = 1.0

    return np.array(
        [
            peak_h,
            local_integral,
            hist_skewness,
            fwhm_mm,
            curvature,
            rel_height,
            nearest_peak_dz,
            nearest_peak_ratio,
        ],
        dtype=np.float32,
    )


def load_gbt_model(path: str):  # noqa: ANN201 - sklearn estimator
    """Load ``gbt_hist_model`` from a peak_classifier_v2 results pickle."""
    with open(path, "rb") as f:
        model = pickle.load(f).get("gbt_hist_model")
    if model is None:
        raise ValueError(
            "The provided pkl does not contain 'gbt_hist_model'. Did you run "
            "peak_classifier_v2.py with the updated save logic?"
        )
    return model


def filter_peaks(
    p_pvs: np.ndarray,
    p_hts: np.ndarray,
    hist: np.ndarray,
    model,  # noqa: ANN001 - sklearn estimator
    threshold: float,
) -> np.ndarray:
    """Keep the peak positions whose GBT probability is >= ``threshold``."""
    if len(p_pvs) == 0:
        return p_pvs
    feats = np.stack(
        [
            compute_hist_features(p_pvs[j], p_hts[j], hist, p_pvs, p_hts)
            for j in range(len(p_pvs))
        ]
    )
    return p_pvs[model.predict_proba(feats)[:, 1] >= threshold]
