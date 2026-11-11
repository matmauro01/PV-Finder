"""Alternative position estimators for a reconstructed vertex.

PV-Finder's production position is the local centroid of the output histogram
(``peak_finding_fast``, +-3 bins of 0.04 mm). These estimators are used to
test whether the hard-scatter (HS) resolution is limited by that read-out:

* :func:`parabolic_peak`: three-point interpolation around the maximum bin,
  on the bin heights or on their logarithm (exact for a Gaussian peak);
* :func:`weighted_z0_mean`: inverse-variance mean of the z0 of the tracks
  assigned to the vertex (the GNN assignment in the PV-Finder chain);
* :func:`robust_weighted_z0_mean`: the same, iteratively dropping tracks more
  than ``n_sigma`` standard deviations from the current mean.

Histogram positions follow the peak finder convention: bin ``i`` is centred
at ``Z_MIN + i * BIN_WIDTH``.
"""

from __future__ import annotations

import numpy as np

from pv_finder.utils.constants import Z_MAX, Z_MIN

BIN_WIDTH = (Z_MAX - Z_MIN) / 12000  # 0.04 mm


def parabolic_peak(hist: np.ndarray, peak_bin: int, log: bool = False) -> float:
    """Vertex z [mm] from a parabola through the maximum bin and its neighbours.

    With ``log=True`` the parabola is fitted to ln(height), which is exact for
    a sampled Gaussian. Falls back to the bin centre at the histogram edge, when
    a neighbour is not positive (log only) or when the three points are not
    concave.
    """
    i = int(peak_bin)
    centre = Z_MIN + i * BIN_WIDTH
    if i <= 0 or i >= len(hist) - 1:
        return float(centre)
    y = np.asarray(hist[i - 1 : i + 2], dtype=np.float64)
    if log:
        if np.any(y <= 0.0):
            return float(centre)
        y = np.log(y)
    denom = y[0] - 2.0 * y[1] + y[2]
    if not denom < 0.0:
        return float(centre)
    delta = 0.5 * (y[0] - y[2]) / denom
    return float(centre + np.clip(delta, -0.5, 0.5) * BIN_WIDTH)


def weighted_z0_mean(z0: np.ndarray, z0_err: np.ndarray) -> float:
    """Inverse-variance weighted mean of ``z0``; NaN for no usable track."""
    z0 = np.asarray(z0, dtype=np.float64)
    err = np.asarray(z0_err, dtype=np.float64)
    ok = np.isfinite(z0) & np.isfinite(err) & (err > 0.0)
    if not ok.any():
        return float("nan")
    w = 1.0 / err[ok] ** 2
    return float(np.sum(w * z0[ok]) / np.sum(w))


def robust_weighted_z0_mean(
    z0: np.ndarray,
    z0_err: np.ndarray,
    n_sigma: float = 3.0,
    max_iter: int = 5,
    seed: float | None = None,
) -> float:
    """:func:`weighted_z0_mean` after iteratively removing outlier tracks.

    A track is removed when ``|z0 - mean| > n_sigma * z0_err``. Iteration stops
    when the kept set does not change, after ``max_iter`` passes, or when no
    track would remain (the previous mean is then returned).

    With ``seed`` (e.g. the histogram peak position), the first selection is
    made around the seed instead of around the plain mean, so a few far
    outliers cannot pull the starting point; if no track is compatible with
    the seed, the seed is returned.
    """
    z0 = np.asarray(z0, dtype=np.float64)
    err = np.asarray(z0_err, dtype=np.float64)
    keep = np.isfinite(z0) & np.isfinite(err) & (err > 0.0)
    if seed is not None:
        keep &= np.abs(z0 - seed) <= n_sigma * err
        if not keep.any():
            return float(seed)
    mean = weighted_z0_mean(z0[keep], err[keep])
    for _ in range(max_iter):
        if not np.isfinite(mean):
            break
        new_keep = keep & (np.abs(z0 - mean) <= n_sigma * err)
        if not new_keep.any() or np.array_equal(new_keep, keep):
            break
        keep = new_keep
        mean = weighted_z0_mean(z0[keep], err[keep])
    return mean
