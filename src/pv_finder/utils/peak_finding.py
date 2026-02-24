"""
Histogram peak finding for PV-Finder predicted histograms.

Scans contiguous above-threshold regions in a 12000-bin histogram, recording
a PV candidate when the region meets width and integral criteria.  Each region
yields exactly one PV at the weighted-mean z-position.

Used by both evaluation and diagnostics (shared logic).
"""

from __future__ import annotations

import numpy as np

from pv_finder.utils.constants import Z_MAX, Z_MIN

# Bin geometry (must match model output: 12 subevents x 1000 bins = 12000)
_N_BINS = 12000
_BIN_WIDTH = (Z_MAX - Z_MIN) / _N_BINS  # 0.04 mm


def pv_locations_updated_res(
    targets: np.ndarray,
    threshold: float = 0.01,
    integral_threshold: float = 0.5,
    min_width: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract PV z-positions from a 12000-bin histogram.

    Scans bins left-to-right, accumulating contiguous above-threshold regions.
    A region is recorded as a PV if it meets the width and integral criteria.
    Each region produces exactly one PV at the weighted-mean position.

    Parameters
    ----------
    targets:
        1-D array of histogram values (length 12000).
    threshold:
        Minimum bin value to be considered "on".
    integral_threshold:
        Minimum sum of bin values in a contiguous region to record a PV.
    min_width:
        Minimum number of consecutive above-threshold bins.

    Returns
    -------
    z_positions : np.ndarray (float32)
        Weighted-mean z-positions in mm for each detected PV.
    peak_heights : np.ndarray (float32)
        Maximum bin value within each detected region.
    peak_bins : np.ndarray (int32)
        Bin index of the maximum within each region.
    pv_sigmas : np.ndarray (float32)
        Weighted standard deviation of the region, converted to mm.
    """
    # Accumulator state
    state = 0
    integral = 0.0
    sum_wl = 0.0  # sum of (bin_value * bin_index)
    sum_wl2 = 0.0  # sum of (bin_value * bin_index^2)
    currentmax = 0

    # Pre-allocate output arrays (resized dynamically if needed)
    cap = 500
    items = np.empty(cap, np.float32)
    peakvals = np.empty(cap, np.float32)
    peakpos = np.empty(cap, np.int32)
    sigmas = np.empty(cap, np.float32)
    n = 0  # number of recorded PVs

    for i in range(len(targets)):
        if state == 0:
            currentmax = i

        # Accumulate above-threshold bins
        if targets[i] >= threshold:
            state += 1
            integral += targets[i]
            sum_wl += i * targets[i]
            sum_wl2 += (i * i) * targets[i]

            if targets[i] > targets[currentmax]:
                currentmax = i

        # End of region: below threshold or last bin
        if (targets[i] < threshold or i == len(targets) - 1) and state > 0:
            if state >= min_width and integral >= integral_threshold:
                # Resize if capacity exceeded
                if n >= cap:
                    cap += 1
                    items = np.resize(items, cap)
                    peakvals = np.resize(peakvals, cap)
                    peakpos = np.resize(peakpos, cap)
                    sigmas = np.resize(sigmas, cap)

                wmean = sum_wl / integral
                wvar = (sum_wl2 / integral) - wmean * wmean
                if wvar < 0:
                    wvar = 0.0

                items[n] = wmean * _BIN_WIDTH + Z_MIN
                peakvals[n] = targets[currentmax]
                peakpos[n] = currentmax
                sigmas[n] = np.sqrt(wvar) * _BIN_WIDTH

                n += 1

            # Reset accumulator
            state = 0
            integral = 0.0
            sum_wl = 0.0
            sum_wl2 = 0.0

    return (
        items[:n],
        peakvals[:n],
        peakpos[:n],
        sigmas[:n],
    )
