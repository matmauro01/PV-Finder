"""Numba-compiled peak finding, bit-identical to pv_locations_updated_res.

The pure-Python scan in peak_finding.py costs ~61 ms per 12000-bin
histogram at PU200 — 74% of the full PVF+GNN chain latency. This module
JIT-compiles the *identical* algorithm (same statements, same accumulator
types: float64 accumulation over float32/float64 bins), so outputs are
bit-identical by construction; gnn.evaluation.verify_fast_paths checks
that on real PU200 histograms.

Use pv_locations_updated_res_fast as a drop-in replacement. First call
per input dtype pays a one-off JIT compilation (~1 s).
"""

from __future__ import annotations

import numba
import numpy as np

from pv_finder.utils.constants import Z_MAX, Z_MIN
from pv_finder.utils.peak_finding import LEGACY_CENTROID_HALFWIDTH

_N_BINS = 12000
_BIN_WIDTH = (Z_MAX - Z_MIN) / _N_BINS  # 0.04 mm


@numba.njit(cache=True)
def _scan(
    targets: np.ndarray,
    threshold: float,
    integral_threshold: float,
    min_width: int,
    min_height: float,
    centroid_halfwidth: int,
    items: np.ndarray,
    peakvals: np.ndarray,
    peakpos: np.ndarray,
    sigmas: np.ndarray,
) -> int:
    """Region scan; fills pre-allocated outputs, returns the peak count."""
    state = 0
    integral = 0.0
    sum_wl = 0.0
    sum_wl2 = 0.0
    currentmax = 0
    region_start = 0
    peak_passed = False
    n = 0

    for i in range(len(targets)):
        if state == 0:
            currentmax = i

        if targets[i] >= threshold:
            if state == 0:
                region_start = i
            state += 1
            integral += targets[i]
            sum_wl += i * targets[i]
            sum_wl2 += (i * i) * targets[i]

            if targets[i] > targets[currentmax]:
                currentmax = i

            if i > 0 and targets[i - 1] > targets[i]:
                peak_passed = True

        conjoined_split = i > 0 and (targets[i - 1] < targets[i]) and peak_passed
        if (
            targets[i] < threshold or i == len(targets) - 1 or conjoined_split
        ) and state > 0:
            if (
                state >= min_width
                and integral >= integral_threshold
                and targets[currentmax] >= min_height
            ):
                wmean = sum_wl / integral
                wvar = (sum_wl2 / integral) - wmean * wmean
                if wvar < 0:
                    wvar = 0.0

                pos = wmean
                if centroid_halfwidth > 0:
                    region_end = i if targets[i] >= threshold else i - 1
                    lo = max(region_start, currentmax - centroid_halfwidth)
                    hi = min(region_end, currentmax + centroid_halfwidth)
                    total = 0.0
                    weighted = 0.0
                    for j in range(lo, hi + 1):
                        v = targets[j]
                        if v > 0.0:
                            total += v
                            weighted += j * v
                    pos = weighted / total if total > 0.0 else float(currentmax)

                items[n] = pos * _BIN_WIDTH + Z_MIN
                peakvals[n] = targets[currentmax]
                peakpos[n] = currentmax
                sigmas[n] = np.sqrt(wvar) * _BIN_WIDTH

                n += 1

            state = 0
            integral = 0.0
            sum_wl = 0.0
            sum_wl2 = 0.0
            peak_passed = False

    return n


def pv_locations_updated_res_fast(
    targets: np.ndarray,
    threshold: float = 0.01,
    integral_threshold: float = 0.5,
    min_width: int = 3,
    min_height: float = 0.0,
    centroid_halfwidth: int = LEGACY_CENTROID_HALFWIDTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Drop-in numba replacement for pv_locations_updated_res.

    Same parameters, same return values; see peak_finding.py for the full
    docstring. Outputs are bit-identical to the Python implementation.
    """
    targets = np.ascontiguousarray(targets)
    # A region needs >= 1 bin, so len(targets) peaks is the hard ceiling.
    cap = len(targets)
    items = np.empty(cap, np.float32)
    peakvals = np.empty(cap, np.float32)
    peakpos = np.empty(cap, np.int32)
    sigmas = np.empty(cap, np.float32)
    n = _scan(
        targets,
        threshold,
        integral_threshold,
        min_width,
        min_height,
        centroid_halfwidth,
        items,
        peakvals,
        peakpos,
        sigmas,
    )
    return items[:n], peakvals[:n], peakpos[:n], sigmas[:n]
