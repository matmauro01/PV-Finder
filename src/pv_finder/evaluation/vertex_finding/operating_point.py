"""Operating-point helpers: compare models at a matched fake rate.

Two models evaluated with the same peak finder can sit at different points of
the efficiency / fake-rate curve. To compare them fairly, one model's
peak-finding threshold is moved until its fakes per event match the other's.
These helpers are pure functions so the matching logic can be tested without
inference; ``scripts/matched_fp_scan.py`` drives them on saved evaluations.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def filter_by_height(
    pvs: np.ndarray, heights: np.ndarray, min_height: float
) -> np.ndarray:
    """Keep the peaks whose height is at least ``min_height``.

    Exact replay of a higher ``min_height`` in ``pv_locations_updated_res``:
    that cut acts on each region's maximum independently, and the saved
    ``pred_heights`` are those maxima.
    """
    pvs = np.asarray(pvs)
    heights = np.asarray(heights)
    if pvs.shape != heights.shape:
        raise ValueError(f"shape mismatch {pvs.shape} vs {heights.shape}")
    return pvs[heights >= min_height]


def bisect_to_target(
    fp_of: Callable[[float], float],
    target: float,
    lo: float,
    hi: float,
    tol: float = 0.02,
    max_iter: int = 30,
) -> tuple[float, float]:
    """Find a threshold in [lo, hi] whose fakes per event match ``target``.

    ``fp_of`` must decrease (weakly) as the threshold rises, as it does for
    the integral and height cuts. Returns ``(threshold, fp)`` for the
    evaluated threshold closest to the target. Raises if the target is not
    bracketed by ``fp_of(lo)`` and ``fp_of(hi)``.
    """
    f_lo, f_hi = fp_of(lo), fp_of(hi)
    if not (f_hi <= target <= f_lo):
        raise ValueError(
            f"target {target} not bracketed: fp({lo})={f_lo}, fp({hi})={f_hi}"
        )
    best = min(((lo, f_lo), (hi, f_hi)), key=lambda p: abs(p[1] - target))
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = fp_of(mid)
        if abs(f_mid - target) < abs(best[1] - target):
            best = (mid, f_mid)
        if abs(f_mid - target) <= tol:
            break
        if f_mid > target:
            lo = mid
        else:
            hi = mid
    return best
