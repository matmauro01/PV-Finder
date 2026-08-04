"""
Histogram peak finding for PV-Finder predicted histograms.

Scans contiguous above-threshold regions in a 12000-bin histogram, recording
a PV candidate when the region meets width and integral criteria.  Each region
yields exactly one PV.

Conjoined-peak splitting: when two nearby peaks overlap and the histogram never
dips below threshold between them, the algorithm detects the local minimum and
splits the region into two separate PV candidates.  This restores the original
behaviour from efficiency_res_optimized_atlas.py and is essential for correctly
measuring the vertex-vertex resolution.

Position estimator: the z-position is the value-weighted centroid of the bins
within ``centroid_halfwidth`` of the region maximum, clipped to the region.
The historical estimator was the centroid of the *whole* region; after a
conjoined split the region is only one side of an overlapping pair, so its
full-region centroid is dragged away from the peak.  See
``docs/research/peak_position_estimator.md``.

Used by both evaluation and diagnostics (shared logic).
"""

from __future__ import annotations

import numpy as np

from pv_finder.utils.constants import Z_MAX, Z_MIN

# Bin geometry (must match model output: 12 subevents x 1000 bins = 12000)
_N_BINS = 12000
_BIN_WIDTH = (Z_MAX - Z_MIN) / _N_BINS  # 0.04 mm

# Recommended half-width (bins) of the local-centroid window: 3 bins = 0.12 mm
# either side of the maximum, chosen on held-out data (see the module docstring
# reference).  This is the default of the PV-finder *evaluation* entry points.
#
# The library default below stays 0 (historical full-region weighted mean) on
# purpose: the GNN graph builders in src/gnn/data feed these positions and
# pv_sigmas to TTVA checkpoints that were trained on them, so changing what they
# see is a separate, deliberate decision.  Opt in explicitly.
RECOMMENDED_CENTROID_HALFWIDTH = 3
LEGACY_CENTROID_HALFWIDTH = 0


def _centroid(targets: np.ndarray, lo: int, hi: int, fallback: int) -> float:
    """Value-weighted mean bin index over the inclusive bin range [lo, hi]."""
    total = 0.0
    weighted = 0.0
    for j in range(lo, hi + 1):
        v = targets[j]
        if v > 0.0:
            total += v
            weighted += j * v
    return weighted / total if total > 0.0 else float(fallback)


def pv_locations_updated_res(
    targets: np.ndarray,
    threshold: float = 0.01,
    integral_threshold: float = 0.5,
    min_width: int = 3,
    min_height: float = 0.0,
    centroid_halfwidth: int = LEGACY_CENTROID_HALFWIDTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract PV z-positions from a 12000-bin histogram.

    Scans bins left-to-right, accumulating contiguous above-threshold regions.
    A region is recorded as a PV if it meets the width and integral criteria.

    Conjoined-peak splitting: if the histogram starts rising again after having
    already passed a local maximum (indicating two overlapping peaks that never
    dip below threshold), the current region is flushed and a new one starts.
    Without this, overlapping peaks are merged into one candidate, which inflates
    the fitted sigma_vtx-vtx.

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
    min_height:
        Minimum peak amplitude (max bin value in the region) to record a PV.
        Default 0.0 keeps all peaks; the production eval sets this to the
        operating point (~0.03) to drop the lowest-amplitude fakes.
    centroid_halfwidth:
        Half-width, in bins, of the window centred on the region maximum used
        to compute the position.  The window is clipped to the region, so it
        never reads bins belonging to a neighbouring peak.  ``0`` restores the
        historical full-region weighted mean.

    Returns
    -------
    z_positions : np.ndarray (float32)
        Peak z-positions in mm.
    peak_heights : np.ndarray (float32)
        Maximum bin value within each detected region.
    peak_bins : np.ndarray (int32)
        Bin index of the maximum within each region.
    pv_sigmas : np.ndarray (float32)
        Weighted standard deviation of the **whole region** about the whole-region
        weighted mean, in mm.  This is deliberately independent of
        ``centroid_halfwidth``: it is consumed as a vertex feature by the GNN
        graph builders, so its definition must not move with the position
        estimator.  It is therefore a region-width estimate, not the spread of
        the window the position was computed from.
    """
    # Accumulator state
    state = 0
    integral = 0.0
    sum_wl = 0.0  # sum of (bin_value * bin_index)
    sum_wl2 = 0.0  # sum of (bin_value * bin_index^2)
    currentmax = 0
    region_start = 0  # first above-threshold bin of the current region
    peak_passed = False  # True once the histogram has started falling within a region

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
            if state == 0:
                region_start = i
            state += 1
            integral += targets[i]
            sum_wl += i * targets[i]
            sum_wl2 += (i * i) * targets[i]

            if targets[i] > targets[currentmax]:
                currentmax = i

            # Track whether we have passed the local maximum of this region
            if i > 0 and targets[i - 1] > targets[i]:
                peak_passed = True

        # End of region: below threshold, last bin, or rising again after a peak.
        # The third condition is the conjoined-peak split: two peaks that overlap
        # without the histogram falling below threshold are separated here.
        conjoined_split = i > 0 and (targets[i - 1] < targets[i]) and peak_passed
        if (
            targets[i] < threshold or i == len(targets) - 1 or conjoined_split
        ) and state > 0:
            if (
                state >= min_width
                and integral >= integral_threshold
                and targets[currentmax] >= min_height
            ):
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

                if centroid_halfwidth > 0:
                    # Bin i closes the region; it belongs to the region only if
                    # it was itself above threshold (true for a conjoined split
                    # and for the final bin, false for a fall below threshold).
                    region_end = i if targets[i] >= threshold else i - 1
                    lo = max(region_start, currentmax - centroid_halfwidth)
                    hi = min(region_end, currentmax + centroid_halfwidth)
                    pos = _centroid(targets, lo, hi, currentmax)
                else:
                    pos = wmean

                items[n] = pos * _BIN_WIDTH + Z_MIN
                peakvals[n] = targets[currentmax]
                peakpos[n] = currentmax
                sigmas[n] = np.sqrt(wvar) * _BIN_WIDTH

                n += 1

            # Reset accumulator
            state = 0
            integral = 0.0
            sum_wl = 0.0
            sum_wl2 = 0.0
            peak_passed = False

    return (
        items[:n],
        peakvals[:n],
        peakpos[:n],
        sigmas[:n],
    )
