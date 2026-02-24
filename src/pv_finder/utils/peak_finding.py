"""
Histogram peak finding for PV-Finder predicted histograms.

Ported from atlas_pvfinder/clean_run3/scripts/utils/efficiency_res_optimized_atlas.py.
Algorithm unchanged -- only style, type hints, and constant references updated.

Used by both evaluation and diagnostics (shared logic).
"""

from __future__ import annotations

import numpy as np

from pv_finder.data.feature_loading import Z_MAX, Z_MIN

# Bin geometry (must match model output: 12 subevents x 1000 bins = 12000)
_N_BINS = 12000
_BIN_WIDTH = (Z_MAX - Z_MIN) / _N_BINS  # 0.04 mm


def pv_locations_updated_res(
    targets: np.ndarray,
    threshold: float = 0.02,
    integral_threshold: float = 0.4,
    min_width: int = 2,
    min_prominence: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract PV z-positions from a 12000-bin histogram.

    Scans bins left-to-right, accumulating contiguous above-threshold regions.
    A region is recorded as a PV if it meets the width and integral criteria.
    Conjoined peaks (two maxima sharing one above-threshold region) are split
    when the valley between them drops by at least ``min_prominence`` fraction
    of the peak height.

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
    min_prominence:
        Fractional valley depth required to split conjoined peaks.
        0.0 (default) recovers original behaviour (split on any rise).
        Recommended: 0.3 (valley must drop by >= 30% of peak height).

    Returns
    -------
    z_positions : np.ndarray (float32)
        Weighted-mean z-positions in mm for each detected PV.
    peak_heights : np.ndarray (float32)
        Maximum bin value within each detected region.
    peak_bins : np.ndarray (int32)
        Bin index of the maximum within each region.
    conjoined_left : np.ndarray (bool)
        True if this PV was split from the left side of a conjoined pair.
    conjoined_right : np.ndarray (bool)
        True if the *previous* PV was conjoined_left (i.e. this PV is the
        right member of a split pair).
    pv_sigmas : np.ndarray (float32)
        Weighted standard deviation of the region, converted to mm.
    """
    # Accumulator state
    state = 0
    integral = 0.0
    sum_wl = 0.0  # sum of (bin_value * bin_index)
    sum_wl2 = 0.0  # sum of (bin_value * bin_index^2)
    currentmax = 0
    peak_passed = False
    valley_min = float("inf")

    # Pre-allocate output arrays (resized dynamically if needed)
    cap = 500
    items = np.empty(cap, np.float32)
    peakvals = np.empty(cap, np.float32)
    peakpos = np.empty(cap, np.int32)
    conjoined_l = np.empty(cap, bool)
    conjoined_r = np.empty(cap, bool)
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

            if targets[i - 1] > targets[i]:
                peak_passed = True

            if peak_passed:
                valley_min = min(valley_min, targets[i])

        # Prominence gate: decide whether a rising edge after a dip
        # justifies splitting the current region into two PVs.
        should_split = False
        if targets[i - 1] < targets[i] and peak_passed:
            peak_height = targets[currentmax]
            if peak_height > 0:
                prominence = (peak_height - valley_min) / peak_height
            else:
                prominence = 0.0
            if prominence >= min_prominence:
                should_split = True
            else:
                peak_passed = False
                valley_min = float("inf")

        # End of region: below threshold, last bin, or valid split point
        end_of_region = targets[i] < threshold or i == len(targets) - 1
        if (end_of_region or should_split) and state > 0:
            if state >= min_width and integral >= integral_threshold:
                # Resize if capacity exceeded
                if n >= cap:
                    cap += 1
                    items = np.resize(items, cap)
                    peakvals = np.resize(peakvals, cap)
                    peakpos = np.resize(peakpos, cap)
                    conjoined_l = np.resize(conjoined_l, cap)
                    conjoined_r = np.resize(conjoined_r, cap)
                    sigmas = np.resize(sigmas, cap)

                wmean = sum_wl / integral
                wvar = (sum_wl2 / integral) - wmean * wmean
                if wvar < 0:
                    wvar = 0.0

                items[n] = wmean * _BIN_WIDTH + Z_MIN
                peakvals[n] = targets[currentmax]
                peakpos[n] = currentmax
                sigmas[n] = np.sqrt(wvar) * _BIN_WIDTH

                conjoined_l[n] = should_split
                conjoined_r[n] = n > 0 and conjoined_l[n - 1]

                n += 1

            # Reset accumulator
            state = 0
            integral = 0.0
            sum_wl = 0.0
            sum_wl2 = 0.0
            peak_passed = False
            valley_min = float("inf")

    return (
        items[:n],
        peakvals[:n],
        peakpos[:n],
        conjoined_l[:n],
        conjoined_r[:n],
        sigmas[:n],
    )
